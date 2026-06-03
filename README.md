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

### Phase 1 — Pure Python Agents

| File | What it builds |
|------|----------------|
| `01_agent_basics.py` | ReAct loop from scratch — think / act / observe |
| `02_tools.py` | RAG pipeline wrapped as an agent tool |
| `03_memory.py` | Short-term conversation memory |
| `04_multi_tool_agent.py` | RAG + web search + calculator in one agent |
| `agent_evaluation.py` | Eval framework — faithfulness, relevance, trajectory scoring |
| `06_agent_failures.py` | 5 failure modes, guardrail suite, 3 recovery strategies |

### Phase 2 — Advanced Pure Python

| File | What it builds |
|------|----------------|
| `07_planning_agent.py` | Static + dynamic multi-step task planning |
| `08_reflection_agent.py` | Agent critiques and revises its own output |

### Phase 3 — LangGraph

| File | What it builds |
|------|----------------|
| `09_multi_agent.py` | Two agents collaborating via LangGraph |
| `10_agent_memory_long.py` | Persistent memory with LangGraph + Postgres |
| `11_tool_calling.py` | Structured tool use |
| `12_ragas_langsmith.py` | RAGAS + LangSmith eval — covers 09, 10, 11 |

### Phase 4 — Ecosystem

| File | What it builds |
|------|----------------|
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
- Guardrail suite: `LoopGuard`, `ToolValidator`, `MaxStepsGuard`, `ThoughtGuard`, `FinalAnswerGuard`
- 3 recovery strategies: retry rephrased, fallback tool, partial answer
- Before/after eval scores per failure — shows exactly what each recovery improved

**Planning Agent (`07_planning_agent.py`)**
- `StaticPlanner` generates full plan upfront via a single LLM call
- `DynamicReplanner` revises only pending steps after a failure — completed work is never thrown away
- `plan_adherence_score` — new metric: low adherence + high faithfulness = replanning worked; low + low = replanning also failed
- `summarizer_tool` condenses intermediate results between steps
- `fallback_map` dict pattern instead of if/elif — swappable, scalable

**Reflection Agent (`08_reflection_agent.py`)**
- `ReflectionCritic` scores output on 4 axes: accuracy, completeness, clarity, relevance (0–10)
- `ImprovementDirective` converts critique into structured fix instructions — not raw critique text passed back into the next generation
- Loop runs until `score >= threshold` or `max_reflections` hit — full `ReflectionRound` audit trail
- `reflection_improvement_score` measures score delta across rounds — only meaningful when the agent needs 2+ rounds
- `has_failure` property carried forward for Phase 3 multi-agent compatibility

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
python 07_planning_agent.py
python 08_reflection_agent.py
```

---

## Progress

- [x] Phase 1 — Pure Python Agents (6/6)
- [x] Phase 2 — Advanced Pure Python (2/2)
- [ ] Phase 3 — LangGraph (0/4)
- [ ] Phase 4 — Ecosystem (0/4)

---

RAG pipeline deployed on HuggingFace Spaces → [link coming soon]
