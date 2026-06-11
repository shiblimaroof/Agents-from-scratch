# LLM Agent Engineering — From Scratch to Production

Building production-grade LLM agents from first principles.  
No shortcuts. Every concept implemented by hand before reaching for a framework.

**Stack:** Python · Groq (llama-3.3-70b-versatile) · sentence-transformers · FAISS · BM25 · LangGraph · CrewAI · FastAPI

---

## Philosophy

Most agent tutorials hand you a framework and call it learning.  
This repo goes the other direction — every pattern is built in pure Python first, so the abstractions actually mean something when you reach for LangGraph or CrewAI.

The progression is deliberate:

```
Pure Python → Advanced Python → LangGraph → Ecosystem
```

By the time you hit Phase 3, you've already built the thing the framework is abstracting.

---

## Structure

### Phase 1 — Pure Python Agents ✅

| File | What it builds |
|------|----------------|
| `01_agent_basics.py` | ReAct loop from scratch — think / act / observe |
| `02_tools.py` | RAG pipeline wrapped as an agent tool |
| `03_memory.py` | Short-term conversation memory |
| `04_multi_tool_agent.py` | RAG + web search + calculator in one agent |
| `agent_evaluation.py` | Eval framework — faithfulness, relevance, trajectory scoring |
| `06_agent_failures.py` | 5 failure modes, guardrail suite, 3 recovery strategies |

### Phase 2 — Advanced Pure Python ✅

| File | What it builds |
|------|----------------|
| `07_planning_agent.py` | Static + dynamic multi-step task planning |
| `08_reflection_agent.py` | Agent critiques and revises its own output |

### Phase 3 — LangGraph ✅

| File | What it builds |
|------|----------------|
| `09_multi_agent.py` | Researcher + Writer agents with supervisor routing via LangGraph |
| `10_agent_memory_long.py` | Persistent memory with LangGraph + Postgres checkpointer |
| `11_tool_calling.py` | Structured tool use with Groq function calling + ToolNode |
| `12_ragas_langsmith.py` | RAGAS + LangSmith eval pipeline — 9 runs across agents 09, 10, 11 |

### Phase 4 — Ecosystem 🔄

| File | What it builds |
|------|----------------|
| `13_mcp.py` | Model Context Protocol — FastMCP server + client, 3 tools discovered ✅ |
| `14_crewai_multi_agent.py` | CrewAI sequential crew: Researcher → Analyst → Writer ✅ |
| `15_agent_api.py` | FastAPI wrapper — `/ask`, `/ask/stream`, `/eval`, `/health` 🔄 |
| `16_agent_deployment.py` | Deployment to HuggingFace Spaces ⬜ |

---

## Key Design Decisions

**Evaluation (`agent_evaluation.py`)**
- No separate `evaluate()` wrapper — methods called directly from orchestrator
- `faithfulness()` returns `dict` with score, details, and unsupported claims
- `TokenCostTracker` wraps Groq client via monkey-patch
- `EvalResult` carries `faithfulness_details` and `unsupported_claims` as list fields
- `_pass_check` returns `tuple[bool, list[str]]` — failed metric names included
- Renamed from `05_agent_evaluation.py` → `agent_evaluation.py` for clean imports in 06+

**Failure Modes & Recovery (`06_agent_failures.py`)**
- 5 failure types: infinite loop, tool hallucination, empty thought, observation ignored, early termination
- Guardrail suite: `LoopGuard`, `ToolValidator`, `MaxStepsGuard`, `ThoughtGuard`, `FinalAnswerGuard`
- 3 recovery strategies: retry rephrased, fallback tool, partial answer
- Before/after eval scores per failure — shows exactly what each recovery improved

**Planning Agent (`07_planning_agent.py`)**
- `StaticPlanner` generates full plan upfront via a single LLM call
- `DynamicReplanner` revises only pending steps after a failure — completed work never thrown away
- `plan_adherence_score` — low adherence + high faithfulness = replanning worked
- `fallback_map` dict pattern instead of if/elif — swappable, scalable

**Reflection Agent (`08_reflection_agent.py`)**
- `ReflectionCritic` scores on 4 axes: accuracy, completeness, clarity, relevance (0–10)
- `ImprovementDirective` converts critique into structured fix instructions
- `reflection_improvement_score` measures score delta across rounds

**LangGraph Multi-Agent (`09_multi_agent.py`)**
- Researcher + Writer with supervisor routing pattern
- `NO_ANSWER_STRINGS` detection for explicit routing — not length-based thresholds
- LangSmith tracing live; Agent 09 achieved faithfulness 0.952 in RAGAS eval

**Persistent Memory (`10_agent_memory_long.py`)**
- `PostgresSaver.from_conn_string()` with `autocommit=True` — fixes `CREATE INDEX CONCURRENTLY` bug
- Thread-based memory isolation via `thread_id`

**Tool Calling (`11_tool_calling.py`)**
- `@tool` decorator + `bind_tools()` + LangGraph `ToolNode`
- Real DuckDuckGo web search via `ddgs`
- Retry logic for Groq `tool_use_failed` errors

**RAGAS + LangSmith (`12_ragas_langsmith.py`)**
- RAGAS pinned to 0.2.15; `ragas/llms/base.py` patched for VertexAI import errors
- 9 eval runs pushed to LangSmith across agents 09, 10, 11
- Agent 09 led: faithfulness 0.952

**MCP (`13_mcp.py`)**
- `mcp.server.fastmcp.FastMCP` (not third-party `fastmcp`)
- `streamable_http_app()` over deprecated `http_app()`
- `127.0.0.1` instead of `localhost` — fixes IPv6/IPv4 conflict
- `_wait_for_server()` polling ensures client connects after server is ready
- 3 tools discovered end-to-end ✅

**CrewAI Multi-Agent (`14_crewai_multi_agent.py`)**
- Sequential crew: Researcher (rag_search + web_search) → Analyst (calculator) → Writer (no tools)
- Principle of least privilege — each agent gets only the tools it needs
- Writer synthesises purely from `context=` chaining — no tool access
- `litellm` `cache_breakpoint` patch for Groq compatibility
- Tools imported from `tools.py` + `11_tool_calling.py` — no duplication

**FastAPI Wrapper (`15_agent_api.py`)**
- RAG pipeline loaded once at startup via FastAPI `lifespan` — not per request
- `POST /ask` — sync response with answer + latency
- `POST /ask/stream` — Server-Sent Events streaming via `StreamingResponse`
- `POST /eval` — per-request RAGAS scoring tied to file 12
- `GET /health` — readiness probe with `agent_loaded` + `memory_backend` fields
- Postgres memory optional — falls back to in-memory if `DATABASE_URL` not set

---

## Findings Worth Noting

**On faithfulness scores (`agent_evaluation.py`)**

> A short, vague answer can score higher on faithfulness than a correct, detailed one.

Faithfulness only asks *"are the claims grounded in context?"*  
A generic answer makes fewer claims — less surface area to fail on.  
This is why faithfulness alone is misleading. Relevance, trajectory quality, and tool-use metrics are all needed together.

**On reflection convergence (`08_reflection_agent.py`)**

> Hitting threshold on Round 1 doesn't mean the reflection loop added no value — it means the generator was already strong.

Set `threshold=9.5` to force multiple rounds and observe the improvement loop actually fire.  
`reflection_improvement_score` is 0.0 when the agent converges immediately — that's correct behaviour, not a bug.

**On multi-agent tool assignment (`14_crewai_multi_agent.py`)**

> Giving every agent every tool doesn't make the crew smarter — it makes it less predictable.

The Analyst with only a calculator can't hallucinate a web search result.  
The Writer with no tools can only synthesise what it was given.  
Constraint is a feature.

---

## Setup

```bash
git clone https://github.com/shiblimaroof/agents-from-scratch
cd agents-from-scratch
pip install groq sentence-transformers faiss-cpu rank-bm25 langchain langgraph
pip install crewai litellm fastapi uvicorn ragas==0.2.15
export GROQ_API_KEY=your_key_here
export LANGCHAIN_API_KEY=your_key_here   # for LangSmith tracing
```

Run any file directly:

```bash
python 01_agent_basics.py
python 09_multi_agent.py
python 14_crewai_multi_agent.py

# API server
uvicorn 15_agent_api:app --reload --port 8000
```

---

## Progress

- [x] Phase 1 — Pure Python Agents (6/6)
- [x] Phase 2 — Advanced Pure Python (2/2)
- [x] Phase 3 — LangGraph (4/4)
- [ ] Phase 4 — Ecosystem (2/4) — 15 in progress, 16 next

---

LangSmith project: `agents-from-scratch` · 9 eval runs logged  
RAG pipeline deployed on HuggingFace Spaces → [link coming soon]
