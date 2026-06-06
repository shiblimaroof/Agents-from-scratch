"""
11_tool_calling.py — Structured Tool Use with LangGraph + Groq Function Calling
=================================================================================
Demonstrates production-grade tool calling:
  - Tools defined as typed schemas via @tool decorator
  - LLM emits structured JSON tool calls (Groq function calling API)
  - LangGraph ToolNode handles dispatch + result injection automatically
  - Parallel tool calls: LLM can fire multiple tools in one turn
  - Loop runs until LLM stops emitting tool_calls → END
 
Key design decisions:
* bind_tools() attaches JSON schemas to the ChatGroq LLM — no manual schema writing
* ToolNode replaces manual if/else dispatch from 04_multi_tool_agent.py
* add_messages reducer (not operator.add) — handles HumanMessage / AIMessage / ToolMessage types
* Routing: AIMessage.tool_calls present → ToolNode, else → END
* Parallel tool calls supported natively — LLM decides how many tools to fire per turn
* Tools are stateless functions — pure input → output, no side effects
* 04_multi_tool_agent.py: manual dispatch, raw strings. 11: typed schemas, graph-native dispatch
"""

from __future__ import annotations

import os
import math
import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


load_dotenv()

#Optional RAG

try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "tools",
        os.path.join(os.path.dirname(__file__), "tools.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _rag_search = _mod.rag_search
    print("[info] RAG tool loaded from tools.py")
except Exception as e:
    print(f"[warn] RAG unavailable ({e}), using mock")
    def _rag_search(query: str) -> str:
        knowledge = {
            "bm25":      "BM25 is a sparse retrieval method based on TF-IDF term weighting.",
            "faiss":     "FAISS performs ANN search via IVF or HNSW indexing of dense vectors.",
            "langgraph": "LangGraph models agent workflows as state machines with typed state.",
            "rag":       "RAG combines retrieval and generation — retrieved context grounds the LLM output.",
            "dense":     "Dense retrieval uses bi-encoder models to embed queries into shared vector spaces.",
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return value
        return "No relevant information found."

# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions — @tool decorator generates the JSON schema automatically
# ─────────────────────────────────────────────────────────────────────────────
 
@tool
def rag_search(query: str) -> str:
    """Search the RAG knowledge base for information about NLP, LLMs, and retrieval systems.
    Use this tool when the user asks about technical concepts like BM25, FAISS, RAG, LangGraph,
    dense retrieval, transformers, or similar topics."""

    try:
        return _rag_search(query)
    except Exception as e:
        return f"RAG search error : {e}"
    
@tool
def calculator(expression : str) -> str:
    """Evaluate a mathematical expression and return the result.
    Supports arithmetic, powers, square roots, and standard math functions.
    Examples: '2 + 2', 'sqrt(144)', '2 ** 10', '(3 + 4) * 5'"""

    try:
        # Safe eval: only math functions, no builtins
        allowed = {k : getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        result = eval(expression , {"__builtins__" : {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error : {e}"
    
@tool
def web_search(query: str) -> str:
    """Search the web for current information not available in the knowledge base.
    Use this for recent events, news, or topics outside the NLP/ML domain."""
    try:
      from duckduckgo_search import DDGS
      results = DDGS().text(query, max_results= 3)
      if not results:
          return "No results found."
      return "\n".join(f"{r['title']}:{r['body']}"for r in results)
    except Exception as e:
        return f"Web search error : {e}"
    
# ─────────────────────────────────────────────────────────────────────────────
# Tools registry + LLM with bound tools
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [rag_search, calculator, web_search]

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key= os.environ.get("GROQ_API_KEY", ""),
    temperature= 0.3,
)

# bind_tools: attaches JSON schemas to LLM — it now knows tool names, descriptions, arg types
llm_with_tools = llm.bind_tools(TOOLS)

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
 
class AgentState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    # add_messages reducer: merges HumanMessage / AIMessage / ToolMessage correctly
    # Unlike operator.add, it handles message ID deduplication and tool result injection

# ─────────────────────────────────────────────────────────────────────────────
# Node: Agent
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a technical assistant with access to three tools:
- rag_search: search the knowledge base for NLP/ML/retrieval concepts (BM25, FAISS, RAG, LangGraph)
- calculator: evaluate mathematical expressions
- web_search: search the web for companies, products, news, or anything outside the NLP/ML knowledge base

Use tools when needed. You may call multiple tools in parallel if the question
requires information from more than one source. When you have enough information,
provide a clear, accurate answer without calling any more tools."""

def agent_node(state: AgentState) -> dict:
    t0 = time.time()
    
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke(state["messages"])
            break
        except Exception as e:
            if "tool_use_failed" in str(e) and attempt < 2:
                print(f"  [Agent]    tool_use_failed, retrying ({attempt + 1}/3)...")
                time.sleep(0.5)
            else:
                raise
    
    tool_calls = getattr(response, "tool_calls", [])
    latency = int((time.time() - t0) * 1000)

    if tool_calls:
        tool_names = [tc["name"] for tc in tool_calls]
        print(f"  [Agent]    tool_calls={tool_names}  ({latency}ms)")
    else:
        print(f"  [Agent]    final answer  ({latency}ms)")

    return {"messages": [response]}
# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────
 
def route_after_agent(state : AgentState) -> str:
  """
    Inspect the last message. If the LLM emitted tool_calls → dispatch to ToolNode.
    Otherwise the LLM produced a final answer → END."""
  
  last = state["messages"][-1]
  if isinstance(last, AIMessage) and getattr(last, "tool_calls" ,[]):
      return "tools"
  return END

# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────
 
def build_graph():
    
  graph = StateGraph(AgentState)

  graph.add_node("agent" , agent_node)
  graph.add_node("tools", ToolNode(TOOLS))
  # ToolNode: receives AIMessage with tool_calls, runs each tool,
  # injects ToolMessage results back into state["messages"] automatically

  graph.set_entry_point("agent")

  graph.add_conditional_edges(
      "agent",
      route_after_agent,
      {"tools" : "tools" , END:END}
  )

  graph.add_edge("tools", "agent")

  return graph.compile()

app = build_graph()
print("[info] Graph compiled with ToolNode dispatch")

# ─────────────────────────────────────────────────────────────────────────────
# Run helper
# ─────────────────────────────────────────────────────────────────────────────
 
def run(question: str) -> str:
    print(f"\n{'═'*60}")
    print(f"QUESTION: {question}")
    print(f"{'─'*60}")

    t0 = time.time()
    from langchain_core.messages import SystemMessage
    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    result = app.invoke({"messages" : initial_messages})
    total_ms = int((time.time() -t0)*1000)

    final_answer = result["messages"][-1].content

    # Count tool calls made across all turns
    tool_call_count = sum(
        len(getattr(m, "tool_calls", []))
        for m in result["messages"]
        if isinstance(m, AIMessage)
    )

    print(f"{'─'*60}")
    print(f"ANSWER:\n{final_answer}")
    print(f"{'─'*60}")
    print(f"tool_calls_made : {tool_call_count}")
    print(f"total_latency   : {total_ms}ms")
    print(f"{'═'*60}\n")
 
    return final_answer
  
# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
 
 
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("11 — STRUCTURED TOOL CALLING: Groq Function Calling + LangGraph ToolNode")
    print("═" * 60)

  #Single tool - RAG

    run("what is BM25 and how does it work.?")

    # Single tool — calculator
    run("what is the square root of 1764 plus 2 to the power of 8.?")

    # Parallel tool calls — LLM fires rag_search + web_search in one turn
    run("compare FAISS with what Groq does - one is for vectors, one for LLM  inference.")

    # Multi-turn tool use — needs RAG then calculator
    run("if BM25 uses k1=1.5 and b=0.75, what is k1 divided by b?")

    

        
