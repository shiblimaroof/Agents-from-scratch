"""
13_mcp.py — Model Context Protocol (MCP)
=========================================
Server : FastMCP over HTTP/SSE on localhost:8000
Tools  : rag_search · web_search · calculator
Client : MultiServerMCPClient + LangGraph ToolNode agent (Groq)
Tracing: LangSmith
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import importlib
import types
from textwrap import shorten
import httpx

from dotenv import load_dotenv

load_dotenv()

# ── reproducible importlib loader for numbered filenames ──────────────────────
def _load_module(path: str, name : str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MCP SERVER
# ══════════════════════════════════════════════════════════════════════════════
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("agents-from-scratch")
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── import tools directly from file 11 — no duplication ──────────────────────

print("[server] Loading tools from 11_tool_calling.py...")
_mod = _load_module(os.path.join(_HERE, "11_tool_calling.py"), "tool_calling")
print("[server]" "Tools ready")

@mcp.tool()
def rag_search(query :str) ->str:
    """Search the local knowledge base using hybrid BM25 + dense retrieval."""
    print(f"\n  [MCP tool called] rag_search({shorten(query, 60)!r})")
    return _mod.rag_search.invoke(query)

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for current news and prices."""
    print(f"\n  [MCP tool called] web_search({shorten(query, 60)!r})")
    return _mod.web_search.invoke(query)

@mcp.tool()
def calculator(expression :str)->str:
    """Evaluate a mathematical expression and return the result."""
    print(f"\n  [MCP tool called] calculator({expression!r})")
    return _mod.calculator.invoke(expression)

def _run_server():
    import uvicorn
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def _wait_for_server(url: str, timeout: int = 30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.post(url, json={}, timeout=1)
            if r.status_code != 404:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start within {timeout}s")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LANGGRAPH AGENT via MCP CLIENT
# ══════════════════════════════════════════════════════════════════════════════

from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage

SYSTEM_PROMPT = """\
You are a technical assistant with access to three tools:
- rag_search: search the knowledge base for NLP/ML concepts (BM25, FAISS, RAG, dense retrieval)
- calculator: evaluate mathematical expressions
- web_search: search the web for current information outside the knowledge base
Use tools when needed. When you have enough information, provide a clear answer."""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

def _build_graph(tools):
    llm = ChatGroq(
        model = "llama-3.3-70b-versatile",
        temperature= 0,
        api_key= os.environ["GROQ_API_KEY"]
    ).bind_tools(tools)

    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        try:
            response = llm.invoke(messages)
        except Exception as e:
            if "tool_use_failed" in str(e):
                # llama failed to format tool call — answer from parametric memory
                no_tools_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    api_key=os.environ["GROQ_API_KEY"],
                )
                response = no_tools_llm.invoke(messages)
            else:
                raise
        return {"messages": [response]}
    
    def should_continue(state: AgentState):          # ← add this
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools" : "tools", END:END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_agent(question :str , app) ->str:
    state = {"messages" : [HumanMessage(content=question)]}
    result = await app.ainvoke(state)
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVAL
# ══════════════════════════════════════════════════════════════════════════════

EVAL_QUESTIONS = [
    ("rag_search",  "What is BM25 and how does it differ from dense retrieval?"),
    ("web_search", "What is the current price of gold today?"),
    ("calculator",  "What is 1234 multiplied by 5678?"),
]
 
DIVIDER = "═" * 60

async def main():
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "agents-from-scratch")

    print(f"\n{DIVIDER}")
    print("13 — MODEL CONTEXT PROTOCOL (MCP)")
    print(f"Server : FastMCP / HTTP-SSE / localhost:8000")
    print(f"Tools  : rag_search · web_search · calculator")
    print(f"{DIVIDER}\n")

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    print("[main] Server thread started — waiting for it to come up...")
    _wait_for_server("http://127.0.0.1:8000/mcp")
    print("[main] Server ready.")


    # ── connect client ────────────────────────────────────────────────────────
    client = MultiServerMCPClient(
        connections={
            "agents-from-scratch": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp",
            }
        }
    )
    tools = await client.get_tools()
    print(f"[client] Discovered {len(tools)} tools: {[t.name for t in tools]}\n")
    for tool in tools:
        print(f"\n{tool.name}")
        print(f"  description: {tool.description}")
        print(f"  schema: {tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else tool.args_schema}")

    app = _build_graph(tools)  # ← add app =
    # ── run eval ──────────────────────────────────────────────────────────────
    print(f"{DIVIDER}")
    print("EVAL")
    print(f"{DIVIDER}")

    for expected_tool, question in EVAL_QUESTIONS:
        print(f"\n  Question : {question}")
        print(f"  Expects  : {expected_tool}")
        t0 = time.perf_counter()
        answer = await run_agent(question, app)
        latency = int((time.perf_counter() - t0) * 1000)
        print(f"  Answer   : {shorten(answer, 120)}")
        print(f"  Latency  : {latency}ms")
        print(f"  {'─'*56}")

    print(f"\n{DIVIDER}")
    print(f"LangSmith: https://smith.langchain.com")
    print(f"Project  : agents-from-scratch")
    print(f"{DIVIDER}\n")


if __name__ == "__main__":
    asyncio.run(main())