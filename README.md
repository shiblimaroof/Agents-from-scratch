# LLM Agent Engineering — From Scratch to Production

Building production-grade LLM agents from first principles.  
No shortcuts. Every concept implemented by hand before reaching for a framework.

**Stack:** Python · Groq (llama-3.3-70b-versatile) · sentence-transformers · FAISS · BM25 · LangGraph · CrewAI · FastAPI

---

## Structure

### Phase 1 — Pure Python Agents

| File | What it builds |
|------|---------------|
| `01_agent_basics.py` | ReAct loop from scratch — think / act / observe |
| `02_tools.py` | RAG pipeline as an agent tool |
| `03_memory.py` | Short-term conversation memory |
| `04_multi_tool_agent.py` | RAG + web search + calculator in one agent |
| `agent_evaluation.py` | Eval framework — faithfulness, relevance, trajectory scoring |
| `06_agent_failures.py` | 5 failure modes, guardrail suite, 3 recovery strategies |

### Phase 2 — Advanced Pure Python

| File | What it builds |
|------|---------------|
| `07_planning_agent.py` | Static + dynamic multi-step task planning |
| `08_reflection_agent.py` | Agent critiques and revises its own output |

### Phase 3 — LangGraph

| File | What it builds |
|------|---------------|
| `09_multi_agent.py` | Two agents collaborating via LangGraph |
| `10_agent_memory_long.py` | Persistent memory with LangGraph + Postgres |
| `11_tool_calling.py` | Structured tool use |
| `12_ragas_langsmith.py` | RAGAS + LangSmith eval — covers 09, 10, 11 |

### Phase 4 — Ecosystem

| File | What it builds |
|------|---------------|
| `13_mcp.py` | Model Context Protocol |
| `14_crewai_multi_agent.py` | CrewAI multi-agent system |
| `15_agent_api.py` | FastAPI wrapper around agents |
| `16_agent_deployment.py` | Deployment to HuggingFace Spaces |

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
- Guardrail suite: LoopGuard, ToolValidator, MaxStepsGuard, ThoughtGuard, FinalAnswerGuard
- 3 recovery strategies: retry rephrased, fallback tool, partial answer
- Before/after eval scores per failure — shows exactly what each recovery improved

**Planning Agent (`07_planning_agent.py`)**
- `StaticPlanner` generates full plan upfront via a single LLM call
- `DynamicReplanner` revises only pending steps after a failure — completed work is never thrown away
- `plan_adherence_score` — new metric: low adherence + high faithfulness = replanning worked; low + low = replanning also failed
- `summarizer_tool` added to condense intermediate results between steps

---

## A Finding Worth Noting

While building the eval framework I ran into a counterintuitive result:

> A short, vague answer can score **higher** on faithfulness than a correct, detailed one.

Faithfulness only asks *"are the claims grounded in context?"*  
A generic answer makes fewer claims — less surface area to fail on.

This is why faithfulness alone is misleading. Relevance, trajectory quality, and tool-use metrics are all needed together.

---

## Setup

```bash
git clone https://github.com/yourusername/llm-agent-engineering
cd llm-agent-engineering
pip install groq sentence-transformers faiss-cpu rank-bm25
export GROQ_API_KEY=your_key_here
```

Run any file directly:

```bash
python 01_agent_basics.py
python 06_agent_failures.py
python 07_planning_agent.py
```

---



RAG pipeline deployed on HuggingFace Spaces → [link coming soon]
