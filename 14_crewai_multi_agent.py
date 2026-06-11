"""
14_crewai_multi_agent.py — CrewAI Multi-Agent System
======================================================
Three specialised agents collaborate in a sequential crew:
  Researcher  — retrieves facts via rag_search + web_search
  Analyst     — runs numbers via calculator
  Writer      — synthesises a final answer from context
 
Key design decisions:
* CrewAI LLM wrapper points at Groq (groq/llama-3.3-70b-versatile)
* Tools imported from tools.py — no duplication
* Process.sequential: Researcher → Analyst → Writer in order
* Each agent gets only the tools it needs (principle of least privilege)
* Writer receives Researcher + Analyst outputs via task context chaining
* LangSmith tracing enabled via LANGCHAIN_TRACING_V2
* 04_multi_tool_agent.py: single agent, manual dispatch
  14: three specialised agents, CrewAI orchestration
"""

from __future__ import annotations

import os
import sys
import time
import importlib
import importlib.util
from textwrap import shorten

from dotenv import load_dotenv

import litellm
litellm.set_verbose = True

from crewai import LLM

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool as crewai_tool
from ddgs import DDGS

load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "agents-from-scratch")


import litellm
from litellm import completion as _orig_completion

def _patched_completion(**kwargs):
    """Strip cache_breakpoint from messages before sending to Groq."""
    for msg in kwargs.get("messages", []):
        if not isinstance(msg, dict):
            continue
        msg.pop("cache_breakpoint", None)
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_breakpoint", None)
    return _orig_completion(**kwargs)

litellm.completion = _patched_completion

# ── importlib loader for numbered filenames ───────────────────────────────────

def _load_module(path : str, name: str):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name]= mod
    spec.loader.exec_module(mod)
    return mod

DIVIDER = "=" * 60
_HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOOLS  (imported from tools.py, wrapped for CrewAI)
# ══════════════════════════════════════════════════════════════════════════════

print("[tools] Loading RAG pipeline from tools.py")
_tools_mod = _load_module(os.path.join(_HERE, "tools.py"),"tools")
_rag =  _tools_mod.rag_search
print("[tools] RAG pipleline ready")

# Load web_search + calculator from file 11 to avoid duplication
print("[tools] loading web_search + calculator from 11_tool_calling.py")
_mod11 = _load_module(os.path.join(_HERE, "11_tool_calling.py"), "tool_calling")
print("[tools] Done")

@crewai_tool("rag_search")
def rag_search(query: str) -> str:
    """Search the local knowledge base for NLP, ML, and retrieval concepts
    such as BM25, FAISS, RAG, dense retrieval, LangGraph, and transformers."""

    print(f"\n  [tool] rag_search({shorten(query,60)!r})")
    result = _rag(query)
    if not result:
        return "No relevant information found."
    return result

@crewai_tool("web_search")
def web_search(query:str) ->str :
    """Search the web for current information, news, and topics outside
    the local knowledge base."""
    print(f"\n  [tool] web_search({shorten(query, 60)!r})")
    return _mod11.web_search.invoke(query)

@crewai_tool("calculator")
def calculator(expression : str)-> str:
    """Evaluate a mathematical expression.
    Examples: '2 + 2', 'sqrt(144)', '1234 * 5678', '(3 + 4) * 5'"""
    print(f"\n  [tool] calculator({expression!r})")
    return _mod11.calculator.invoke(expression)

 
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LLM
# ══════════════════════════════════════════════════════════════════════════════
os.environ["LITELLM_LOG"] = "WARNING"   # or "ERROR" to silence it entirely
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY", ""),
    temperature= 0.3,
    max_tokens=1024,

)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AGENTS
# ══════════════════════════════════════════════════════════════════════════════

researcher = Agent(
    role ="Research Specialist",
    goal = "Retrieve accurate, relevant information from the knowledge base "
         "and web to answer the user's question thoroughly.",

    backstory =(
        "You are an expert research analyst with deep knowledge of NLP and "
        "information retrieval systems. You always ground your findings in "
        "retrieved evidence rather than assumptions."
    ),
    tools = [rag_search, web_search],
    llm = groq_llm,
    verbose = True,
    allow_delegation = False, 
    max_iter = 3,
    max_retry_limit = 2, 
)

analyst = Agent(
    role = "Data Analyst",
    goal = "Perform any calculations or quantitative analysis required by the" 
            "research findings.",
    backstory = "You are a precise data analyst who loves turning raw numbers into "
            "clear insights. You use the calculator tool for all arithmetic to "
            "ensure accuracy.",
    tools = [calculator],
    llm = groq_llm,
    verbose = True,
    allow_delegation = False, 
    max_iter = 2,
    max_retry_limit = 2, 
)

writer = Agent(
    role = "Technical Writer",
    goal = "Synthesise the research and analysis into a clear, concise, well-structure answer for the user.",
    backstory = (
        "You are a skilled technical writer who distills complex findings "
        "into accessible explanations. You rely only on the context provided "
        "by the Researcher and Analyst — never fabricate information."
    ),
    tools = [],
    llm = groq_llm,
    verbose = True,
    allow_delegation = False,
    max_iter = 2,
    max_retry_limit = 2,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CREW BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_crew(question :str) -> Crew:
    research_tast = Task(
        description = (
        "Research the following question thoroughly using tools :\n\n"
        f"QUESTION : {question}\n\n"
        "User rag_search for NLP/ML/retrieval topics."
        "Use web_searh for current events or topics outside the knowledge."
        "Return all relevant facts you find."
    ),
    expected_output = (
        "A concise summary (max 200 words) of findings with key facts "
        "and evidence retrieved from tools. No repetition."
    ),
    agent=  researcher,
)
    
    analysis_task = Task(
        description= (
            "Review the research findings from the Researcher."
            "If the question or findings involve any number, calculation,"
            "or quantitative comparison, use the calculator tool to verify"
            "or compute them. If no calculations are needed, summarise the"
            "key quantitative points from the research."
        ),
        expected_output= (
                "One or two sentences: the computed result (if any) and what it means. "
                "No repetition of research findings."
            ),
        agent = analyst,
        context = [research_tast],
    )

    writing_tast = Task(
        description= (
            "Using the outputs from the Researcher and Analyst, write a "
            "clear, accurate , and well-structured answer to :\n\n"
            f"QUESTION : {question}\n\n"
            "Do not use any information beyond what was provided in context"
        ),
        expected_output= (
            "A polished, concise answer (3-5 paragraphs) that directly "
            "addresses the question using only the researched and analysed findings"
        ),
        agent = writer,
        context= [research_tast, analysis_task]

    )

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_tast, analysis_task, writing_tast],
        process= Process.sequential,
        verbose= True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RUN
# ══════════════════════════════════════════════════════════════════════════════

def run(question :str) -> str:
    print(f"\n{DIVIDER}")
    print(f"QUESTION: {question}")
    print(f"{DIVIDER}")

    crew = build_crew(question)
    t0 = time.perf_counter()
    result = crew.kickoff()
    latency = int((time.perf_counter() - t0) *1000)

    answer = result.raw if hasattr(result, "raw") else str(result)

    print(f"\n{DIVIDER}")
    print(f"FINAL ANSWER:\n{answer}")
    print(f"{'─'*60}")
    print(f"total_latency : {latency}ms")
    print(f"{DIVIDER}\n")
 
    return answer

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DEMO
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    print(f"\n{DIVIDER}")
    print("14 — CREWAI MULTI-AGENT")
    print("Agents : Researcher · Analyst · Writer")
    print("Process: Sequential")
    print(f"LLM    : groq/llama-3.3-70b-versatile")
    print(f"{DIVIDER}")


    run("Explain how hybrid search combines BM25 and dense retrieval,"
        "and what are the typical hyperparameters involved")

    import time
    time.sleep(30)   # ← let the TPM window reset before Q2

    run("If BM25 uses k1 = 1.5 and b=0.75, what is k1 divided by b,"
        "and what does each parameter control?")
