"""
10_agent_memory_long.py — Persistent Memory with LangGraph + Postgres
======================================================================
Extends 09 with a PostgresSaver checkpointer — agent state persists
across runs, enabling multi-turn conversations with full memory.
 
Key design decisions:
* PostgresSaver checkpointer — swappable to SqliteSaver with one line
* thread_id: every conversation is a named thread — same thread_id = same memory
* AgentState extended with messages: list[dict] — full conversation history
* memory_node: reads history from state, summarises if > MAX_HISTORY turns
* Supervisor routes to memory_node first — context always loaded before research
* graph compiled with checkpointer=checkpointer — LangGraph handles persistence
* setup_checkpointer() creates tables on first run — idempotent
* DATABASE_URL from .env — no hardcoded credentials
* Swapping to Postgres in prod: change SqliteSaver to PostgresSaver, done
"""

from __future__ import annotations

import os
import time
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

load_dotenv()

# ── Optional RAG tool ─────────────────────────────────────────────────────────
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "tools",
        os.path.join(os.path.dirname(__file__), "tools.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    rag_search = _mod.rag_search
    print("[info] RAG tool loaded from tools.py")
except Exception as e:
    print(f"[warn] RAG unavailable ({e}), using mock")
 
    def rag_search(query: str) -> str:
        knowledge = {
            "faiss": "FAISS performs ANN search via IVF or HNSW indexing of dense vectors.",
            "bm25": "BM25 is a sparse retrieval method based on TF-IDF term weighting.",
            "langgraph": "LangGraph models agent workflows as state machines with typed state.",
            "rag": "RAG combines retrieval and generation — retrieved context grounds the LLM output.",
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return value
        return "I don't have enough context to answer this."
 

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DB_URI = os.environ.get(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/agents"
)
MAX_HISTORY = 10 #summarise after this many turns

NO_ANSWER_STRINGS = {
    "I don't have enough context to answer this.",
    "No relevant information found in knowledge base.",
    "No relevant information found.",
    "No relevant information found",      # without period
    "I don't know.",
    "I don't know",
}

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MODEL = "llama-3.3-70b-versatile"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────
 
class AgentState(TypedDict):
    question: str
    research: str
    answer: str
    has_failure: bool
    failure_reason: str
    eval_scores: dict
    latency_ms: dict
    next: str
    # Persistent fields — survive across turns via checkpointer
    messages: Annotated[list[dict], operator.add]   # full conversation history
    summary: str                                     # running summary if history > MAX_HISTORY
    _messages_reset : list[dict]

# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer setup
# ─────────────────────────────────────────────────────────────────────────────
 
def setup_checkpointer():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
    # Re-open a normal connection for runtime use
    conn = psycopg.connect(DB_URI)
    return PostgresSaver(conn)
# ─────────────────────────────────────────────────────────────────────────────
# Node: Memory
# ─────────────────────────────────────────────────────────────────────────────

SUMMARISER_SYSTEM = """\
You are a memory summariser. Given a conversation history, produce a concise
summary that preserves all key facts, decisions and context needed to continue
the conversation. Be bried 3-5 sentences maximum."""

def memory_node(state:AgentState) ->AgentState:
    """
    Reads conversation history. If history exceeds MAX_HISTORY turns,
    summarises older turns and replaces them with the summary.
    Always runs first — ensures Researcher and Writer have full context.
    """
    t0 = time.time()
    messages = state.get("messages" ,[])
    summary = state.get("summary", "")

    if len(messages) > MAX_HISTORY:
        # Summarise older turns, keep last 4
        to_summarise = messages[:-4]
        recent = messages[-4:]

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_summarise
        )
        try:
            if summary:
                history_text = f"PREVIOUS SUMMARY:\n{summary}\n\nNEW HISTORY:\n{history_text}"

                resp = client.chat.completions.create(
                    model = MODEL,
                    messages=[
                        {"role" : "system" , "content" : SUMMARISER_SYSTEM },
                        {"role" : "user" , "content" : history_text},
                    ],
                    temperature=0.3,
                    max_tokens=300,
                )
                summary = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Memory] summarisation failed ({e}), keeping existing summary")

        print(f"  [Memory] history condensed → summary ({len(summary)} chars)")

        latency = state.get("latency_ms", {})
        latency["memory_ms"] = int((time.time() - t0) * 1000)
 
        # Return _messages_reset (plain field) to overwrite messages cleanly.
        # Also return messages=[] so operator.add adds nothing new this turn.
        return {
            "summary": summary,
            "messages": [],           # operator.add: append nothing
            "_messages_reset": recent, # plain field: overwrites stored list
            "latency_ms": latency,
        }
    else:
        print(f"  [Memory] history={len(messages)} entries — no summarisation needed")
        
        latency = state.get("latency_ms", {})
        latency["memory_ms"] = int((time.time() - t0) * 1000)
 
        return {
            "latency_ms": latency,
            "messages": [],            # append nothing
        }

    
# ─────────────────────────────────────────────────────────────────────────────
# Node: Supervisor
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    t0 = time.time()
 
    if state.get("answer"):
        decision = "end"
    elif state.get("has_failure"):
        decision = "fallback_writer"
    elif state.get("research"):
        decision = "writer"
    else:
        decision = "researcher"
 
    latency = state.get("latency_ms", {})
    latency["supervisor_ms"] = int((time.time() - t0) * 1000)
 
    print(f"  [Supervisor] → {decision}")
 
    return {"next": decision, "latency_ms": latency}

# ─────────────────────────────────────────────────────────────────────────────
# Node: Researcher
# ─────────────────────────────────────────────────────────────────────────────
 
RESEARCHER_SYSTEM = """\
You are a research agent. Gather relevant information to answer the question.
Use the retrieved context and expand on it with your knowledge.
If conversation history or a summary is provided, use it for context continuity.
Return ONLY research findings — no final answer, no preamble."""
 
 
def researcher_node(state: AgentState) -> dict:
    t0 = time.time()
    question = state["question"]
 
    retrieved = rag_search(question)
    has_failure = retrieved.strip() in NO_ANSWER_STRINGS
 
    if not has_failure:
        # Build context — prefer summary over raw history (more token-efficient)
        context_parts = [f"QUESTION :{question}", f"RETRIEVED:\n{retrieved}"]
        if state.get("summary"):
            context_parts.insert(0,f"CONVERSATION SUMMARY:\n{state['summary']}")
        elif state.get("messages"):
            recent = state["messages"][-4:]
            history ="\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
            context_parts.insert(0,f"RECENT HISTORY:\n{history}")
 
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": RESEARCHER_SYSTEM},
                    {"role": "user", "content": "\n\n".join(context_parts)},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            research = resp.choices[0].message.content.strip()
            failure_reason = ""
        except Exception as e:
            research = ""
            has_failure = True
            failure_reason = f"Researcher LLM call failed: {e}"
    else:
        research = ""
        failure_reason = f"RAG returned no relevant result for: '{question}'"
 
    latency = state.get("latency_ms", {})
    latency["researcher_ms"] = int((time.time() - t0) * 1000)
 
    print(f"  [Researcher] has_failure={has_failure}  ({latency['researcher_ms']}ms)")
    if has_failure:
        print(f"  [Researcher] {failure_reason}")
 
    return {
        "research": research,
        "has_failure": has_failure,
        "failure_reason": failure_reason,
        "latency_ms": latency,
        "messages": [],   # researcher doesn't add to history
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Node: Writer (grounded)
# ─────────────────────────────────────────────────────────────────────────────
 
WRITER_SYSTEM = """\
You are a technical writer. Given a question and research findings, write a
clear, accurate, well-structured answer. Be concise but complete.
If a conversation summary or history is provided, maintain continuity."""
 
 
def writer_node(state: AgentState) -> dict:
    t0 = time.time()
    question = state["question"]
    research = state["research"]
 
    context_parts = [f"QUESTION:\n{question}", f"RESEARCH FINDINGS:\n{research}"]
    if state.get("summary"):
        context_parts.insert(0, f"CONVERSATION SUMMARY:\n{state['summary']}")
    elif state.get("messages"):
        recent = state["messages"][-4:]
        history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
        context_parts.insert(0, f"RECENT HISTORY:\n{history}")
 
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": WRITER_SYSTEM},
                {"role": "user", "content": "\n\n".join(context_parts)},
            ],
            temperature=0.5,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Writer failed to generate a response: {e}"
 
    latency = state.get("latency_ms", {})
    latency["writer_ms"] = int((time.time() - t0) * 1000)
 
    eval_scores = {
        "grounded": True,
        "research_len": len(research),
        "answer_len": len(answer),
        "faithfulness": None,
        "relevance": None,
    }
 
    # FIX: Only return the NEW messages for this turn.
    # operator.add will append them to existing history automatically.
    # Do NOT return the full messages list — that caused overwrite bugs before.
    new_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
 
    print(f"  [Writer]     grounded=True  ({latency['writer_ms']}ms)")
 
    return {
        "answer": answer,
        "eval_scores": eval_scores,
        "latency_ms": latency,
        "messages": new_messages,   # operator.add appends these
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Node: Fallback Writer
# ─────────────────────────────────────────────────────────────────────────────
 
WRITER_FALLBACK_SYSTEM = """\
You are a technical writer. The research agent could not find relevant context.
Answer using your own knowledge. Be transparent that this answer is not grounded
in retrieved documents. Clearly note any uncertainty.
If a conversation summary or history is provided, maintain continuity."""
 
 
def fallback_writer_node(state: AgentState) -> dict:
    t0 = time.time()
    question = state["question"]
 
    context_parts = [f"QUESTION:\n{question}\n\nNote: No retrieved context available."]
    if state.get("summary"):
        context_parts.insert(0, f"CONVERSATION SUMMARY:\n{state['summary']}")
    elif state.get("messages"):
        recent = state["messages"][-4:]
        history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
        context_parts.insert(0, f"RECENT HISTORY:\n{history}")
 
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": WRITER_FALLBACK_SYSTEM},
                {"role": "user", "content": "\n\n".join(context_parts)},
            ],
            temperature=0.5,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Fallback writer failed to generate a response: {e}"
 
    latency = state.get("latency_ms", {})
    latency["fallback_writer_ms"] = int((time.time() - t0) * 1000)
 
    eval_scores = {
        "grounded": False,
        "research_len": 0,
        "answer_len": len(answer),
        "faithfulness": None,
        "relevance": None,
    }
 
    new_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
 
    print(f"  [FallbackWriter] grounded=False  ({latency['fallback_writer_ms']}ms)")
 
    return {
        "answer": answer,
        "eval_scores": eval_scores,
        "latency_ms": latency,
        "messages": new_messages,   # operator.add appends these
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────
 
def route_from_supervisor(state: AgentState) -> str:
    return state["next"]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(checkpointer: PostgresSaver):
    graph = StateGraph(AgentState)
 
    graph.add_node("memory", memory_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("fallback_writer", fallback_writer_node)
 
    graph.set_entry_point("memory")
    graph.add_edge("memory", "supervisor")
 
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "fallback_writer": "fallback_writer",
            "end": END,
        },
    )
 
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("fallback_writer", "supervisor")

    return graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────────────────────────
# Run helper
# ─────────────────────────────────────────────────────────────────────────────

def run(app, question: str, thread_id: str) -> AgentState:
    """
    thread_id: identifies the conversation.
    Same thread_id across calls = persistent memory (checkpointer merges state).
    Different thread_id = fresh conversation.
 
    NOTE: initial_state fields like messages=[] are safe to pass every call.
    Because messages uses operator.add, LangGraph APPENDS [] (nothing) to the
    existing checkpoint history — it does NOT reset it.
    """
    print(f"\n{'═'*60}")
    print(f"THREAD : {thread_id}")
    print(f"QUESTION: {question}")
    print(f"{'─'*60}")

    config = {"configurable": {"thread_id" : thread_id}}

    initial_state: AgentState = {
        "question": question,
        "research": "",
        "answer": "",
        "has_failure": False,
        "failure_reason": "",
        "eval_scores": {},
        "latency_ms": {},
        "next": "",
        "messages": [],       # safe: operator.add appends nothing to checkpoint
        "summary": "",
        "_messages_reset": [],
    }
 
    result = app.invoke(initial_state, config=config)
 
    # Resolve effective messages: if memory_node condensed history, use reset list
    effective_messages = result.get("_messages_reset") or result.get("messages", [])
 
    total_ms = sum(result["latency_ms"].values())
    print(f"{'─'*60}")
    print(f"ANSWER:\n{result['answer']}")
    print(f"{'─'*60}")
    print(f"has_failure    : {result['has_failure']}")
    print(f"grounded       : {result['eval_scores'].get('grounded')}")
    print(f"history_turns  : {len(effective_messages) // 2}")
    print(f"summary        : {result['summary'][:80] + '...' if result['summary'] else 'none'}")
    print(f"total_latency  : {total_ms}ms")
    print(f"{'═'*60}\n")
 
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("10 — PERSISTENT MEMORY: LangGraph + Postgres")
    print("Same thread_id = memory persists across runs")
    print("═" * 60)

    checkpointer = setup_checkpointer()
    app = build_graph(checkpointer)

    # Thread A — multi-turn conversation, memory carries forward
    thread_a = "thread_nlp_conceps"
    run(app, "What is BM25 and how does it work?" , thread_a)
    run(app, "How does it compare to dense retrieval?", thread_a)   # remembers BM25 context

    #Thread B - separate conversation, no shared memory with A

    thread_b = "thread_faiss"
    run(app, "What is FAISS used for", thread_b)