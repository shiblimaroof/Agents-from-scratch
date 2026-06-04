"""
09_multi_agent.py — Multi-Agent Collaboration via LangGraph
============================================================
Two agents — Researcher and Writer — collaborate through a
LangGraph state machine.
 
Key design decisions:
* AgentState: shared TypedDict — single source of truth across nodes
* supervisor_node: LLM call that reads state and sets state["next"] to route the graph
* Researcher node: calls rag_search, detects has_failure, returns to supervisor
* writer_node: grounded path only — research context always present
* fallback_writer_node: no-context path only — parametric knowledge, flags uncertainty
* route_from_supervisor: just returns state["next"] — supervisor owns all routing logic
* eval_scores written into state — readable directly by 12_ragas_langsmith.py
* has_failure carried forward from 07/08 — same property, same contract
* No LangChain dependency — pure LangGraph + Groq
* Graph compiled once, reused across all queries (no re-compilation overhead)
"""

from __future__ import annotations

import os
import time
from typing import TypedDict

from groq import Groq
from langgraph.graph import StateGraph , END

# RAG tool from 02

try:
    import importlib.util, sys
    _spec = importlib.util.spec_from_file_location(
        "tools",
        os.path.join(os.path.dirname(__file__),"tools.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    rag_search = _mod.rag_search
    RAG_AVAILABLE = True
    print("[info] RAG tool loaded from tools.py")
except Exception as e:
    RAG_AVAILABLE =False
    print(f"[warn] RAG unavailable ({e}), using mock")


    def rag_search(query : str) -> str:
        knowledge = {
            "faiss": "FAISS performs ANN search via IVF or HNSW indexing of dense vectors.",
            "bm25": "BM25 is a sparse retrieval method based on TF-IDF term weighting.",
            "langgraph": "LangGraph models agent workflows as state machines with typed state.",
            "react": "ReAct alternates between reasoning (Thought) and acting (Action/Observation).",
            "rag": "RAG combines retrieval and generation — retrieved context grounds the LLM output.",

        }
        for key, value in knowledge.items():
            if key in query.lower():
                return value
        return "No relevant information found in knowledge base"
    
# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str           # original user question
    research: str           # Researcher's retrieved context
    answer: str             # Writer's final answer
    has_failure: bool       # True if Researcher hit a failure
    failure_reason: str     # human-readable failure description
    eval_scores: dict       # populated for 12_ragas_langsmith.py
    latency_ms: dict        # per-node timing
    next: str               # supervisor sets this to route the graph

# ─────────────────────────────────────────────────────────────────────────────
# Groq client (shared)
# ─────────────────────────────────────────────────────────────────────────────

client = Groq(api_key= os.environ.get("GROQ_API_KEY", ""))
MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Node: Supervisor
# ─────────────────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """\
You are a supervisor orchestrating a multi-agent pipeline. Decide which agent
should act next based on the current state.
 
Rules (check in this exact order):
1. If answer is populated → route to "end"
2. If has_failure is True → route to "fallback_writer"
3. If research is populated → route to "writer"
4. If research is empty → route to "researcher"
 
Respond with ONLY one word: researcher, writer, fallback_writer, or end."""

def supervisor_node(state : AgentState) -> AgentState:

    t0 = time.time()

    user_content = (
        f"question : {state['question']}\n"
        f"research populated : {bool(state['research'])}"
        f"has_failure : {state['has_failure']}\n"
        f"answer populated : {bool(state['answer'])}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages= [
            {"role" : "system" , "content" : SUPERVISOR_SYSTEM},
            {"role" : "user" , "content" : user_content},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    decision = resp.choices[0].message.content.strip().lower()

    # Sanitise — only accept known routes

    valid = {"researcher" , "writer", "fallback_writer", "end"}
    if decision not in valid:
        decision = "fallback_writer" if state["has_failure"] else(
            "writer" if state["research"] else "researcher"
        )
    latency = state.get("latency_ms", {})
    latency["supervisor_ms"] = int((time.time() - t0) *1000)

    print(f" [Supervisor] -> {decision}")
    
    return{**state, "next" : decision, "latency_ms" : latency}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Researcher
# ─────────────────────────────────────────────────────────────────────────────

RESEARCCHER_SYSTEM = """\
You are a research agent. Your job is to gather relevant information to answer a question.
Use the retrieved context provided and expand on it with your knowledge.
Be factual, thorough, and cite what you know confidently.
Return ONLY  the research findings - no final answer, no preamble.
"""

def researcher_node(state:AgentState) -> AgentState:
    t0 = time.time()
    question = state["question"]

    # Retrieve from RAG
    retrieved = rag_search(question)
    NO_ANSWER_STRINGS = {
        "I don't have enough context to answer this.",
        "No relevant information found in knowledge base.",
        "No relevant information found.",
    }
    has_failure = retrieved.strip() in NO_ANSWER_STRINGS


    if not has_failure:
    # Expand retrieved context with LLM
        resp = client.chat.completions.create(
            model = MODEL,
            messages= [
                {"role" : "system" , "content" : RESEARCCHER_SYSTEM},
                {"role" : "user" , "content" : f"QUESTION : {question}\n\nRETRIEVED:\n{retrieved}"},
            ],
            temperature= 0.3,
            max_tokens=600,
        )
        research = resp.choices[0].message.content.strip()
        failure_reason = ""
    else:
        research = ""
        failure_reason = f"RAG returned no relevant result for : {question}"

    
    latency = state.get("latency_ms", {})
    latency["researcher_ms"] = int((time.time()- t0) *1000)

    print(f"  [Researcher] has_failure={has_failure}  ({latency['researcher_ms']}ms)")
    if has_failure:
        print(f" [Researcher] failure_reason : {failure_reason}")

    return {
        **state,
        "research" : research,
        "has_failure" : has_failure,
        "failure_reason" : failure_reason,
        "latency_ms" : latency,
    }
# ─────────────────────────────────────────────────────────────────────────────
# Node: Writer (grounded — research context present)
# ─────────────────────────────────────────────────────────────────────────────

WRITER_SYSTEM = """\
You are a technical writer. Given a question and research findings, write a
clear, accurate, well-structure answer .Be concise but complete."""

def writer_node(state : AgentState) -> AgentState:

    t0 = time.time()
    question = state["question"]
    research = state["research"]

    resp = client.chat.completions.create(
        model= MODEL,
        messages= [
            {"role": "system" , "content" : WRITER_SYSTEM },
            {"role":"user" , "content" : f"QUESTION:\n{question}\n\nRESEARCH FINDINGS\n{research}"},
        ],
        temperature=0.5,
        max_tokens= 800,
    )
    answer = resp.choices[0].message.content.strip()

    latency = state.get("latency_ms", {})
    latency["writer_ms"] = int((time.time() - t0) *1000)

    eval_scores = {
        "grounded" : True,
        "research_len" : len(research),
        "answer_len" : len(answer),
        "faithfulness" : None,           #filled by 12
        "relevance" : None,            #filled by 12
    }

    print(f"[Writer] grounded =True ({latency['writer_ms']}ms)")

    return {**state, "answer" : answer , "eval_scores" : eval_scores , "latency_ms": latency}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Fallback Writer (no retrieved context — parametric knowledge only)
# ─────────────────────────────────────────────────────────────────────────────

WRITER_FALLBACK_SYSTEM = """\
You are a technical writer. The research agent could not find relevant context
for this question. Answer using own knowledge. Be transparent that this answer
is not grounded in retrieved documents - rely on parametric knowledge only.
Clealy note any uncertainty."""

def fallback_writer_node(state:AgentState) -> AgentState:
    t0 = time.time()
    question = state["question"]

    resp = client.chat.completions.create(
        model= MODEL,
        messages= [
            {"role" : "system" , "content" : WRITER_FALLBACK_SYSTEM},
            {"role" : "user" , "content" : f"QUESTION:\n{question}\n\nNote: No retrieved context available."}  
        ],
        temperature=0.5,
        max_tokens= 800,
    )
    answer = resp.choices[0].message.content.strip()

    latency = state.get("latency_ms" , {})
    latency["fallback_writer_ms"] = int((time.time() -t0) *1000)

    eval_scores = {
        "grounded" : False,
        "research_len" : 0,
        "answer_len" : len(answer),
        "faithfulness" : None,          #filled by 12
        "relevance" : None,             #filled by 12
    }

    print(f" [FallbackWriter] grounded =False ({latency['fallback_writer_ms']}ms)")

    return{**state, "answer" : answer, "eval_scores" : eval_scores , "latency_ms" : latency}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def route_from_supervisor(state: AgentState) ->str:
    """Supervisor owns all routing logic — this just reads state['next']."""
    return state["next"]

# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor" , supervisor_node)
    graph.add_node("researcher" , researcher_node)
    graph.add_node("writer" , writer_node)
    graph.add_node("fallback_writer" , fallback_writer_node)

    graph.set_entry_point("supervisor")

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

    graph.add_edge("researcher" , "supervisor")
    graph.add_edge("writer" , "supervisor")
    graph.add_edge("fallback_writer" , "supervisor")

    return graph.compile()

# Compile once — reuse across all queries
app = build_graph()

# ─────────────────────────────────────────────────────────────────────────────
# Run helper
# ─────────────────────────────────────────────────────────────────────────────

def run(question : str) -> AgentState:
    print(f"\n{'═'*60}")
    print(f"QUESTION: {question}")
    print(f"{'─'*60}")

    initial_state : AgentState = {
        "question" : question,
        "research" : "",
        "answer" : "",
        "has_failure" : False,
        "failure_reason" : "",
        "eval_scores" : {},
        "latency_ms":{},
        "next" : "",
    }

    result = app.invoke(initial_state)

    total_ms = sum(result["latency_ms"].values())
    print(f"{'─'*60}")
    print(f"ANSWER:\n{result['answer']}")
    print(f"{'─'*60}")
    print(f"has_failure : {result['has_failure']}")
    if result["failure_reason"]:
        print(f"failure_reason : {result['failure_reason']}")
    print(f"eval_scores : {result['eval_scores']}")
    print(f"total_latency : {total_ms}ms")
    print(f"{'═'*60}\n")
 
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    # RAG should hit — triggers full Researcher → Writer path
    "How does FAISS perform approximate nearest-neighbor search?",
    "What are the trade-offs between BM25 and dense retrieval in RAG?",
    # RAG will miss — triggers has_failure=True → Writer fallback path
    "What is the capital of France?",

]

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("09 — MULTI-AGENT: RESEARCHER + WRITER (LangGraph)")
    print("Phase 3 begins.")
    print("═" * 60)

    for q in DEMO_QUESTIONS:
        run(q)

