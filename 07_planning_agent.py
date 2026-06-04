"""
07_planning_agent.py
====================
Multi-Step Task Planning Agent
Stack: Python · Groq · sentence-transformers
 
Covers:
  - PlanStep     (single step with goal, tool, depends_on, status)
  - Plan         (ordered steps with status tracking)
  - StaticPlanner  (LLM generates full plan upfront)
  - DynamicReplanner (revises remaining steps after a failure)
  - PlanExecutor   (walks plan, calls tools, triggers replanning)
  - Plan adherence score (did agent follow its own plan?)
  - Before/after eval using agent_evaluation.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────

import os
import re 
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from groq import Groq

# ── local ─────────────────────────────────────────────────────────────────────

from agent_evaluation import(
    AgentStep,
    AgentTrace,
    AgentEvaluator,
    EvalResult,
    THRESHOLDS,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 10
MAX_RETRIES = 2

AVAILABLE_TOOLS : set[str] = {
    "retrieval_tool",
    "web_search",
    "calculator",
    "summarizer_tool",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PLAN DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class StepStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class PlanStep:
    """
    A single step inside a Plan.
 
    depends_on  : list of step_ids this step needs to complete first
    tool        : which tool to call
    goal        : human-readable description of what this step achieves
    result      : populated after execution
    """

    step_id     : str
    goal        : str
    tool        : str
    depends_on  : list[str]         = field(default_factory=list)
    status      : StepStatus        = StepStatus.PENDING
    result      : Optional[str]     = None
    error       : Optional[str]     = None

    def mark_running(self) ->None:
        self.status = StepStatus.RUNNING
    
    def mark_done(self, result : str) -> None:
        self.status = StepStatus.DONE
        self.result = result

    def mark_failed(self, error : str) -> None:
        self.status = StepStatus.FAILED
        self.error = error

    def mark_skipped(self) -> None:
        self.status = StepStatus.SKIPPED


@dataclass
class Plan:
    """
    Ordered list of PlanSteps with helpers for status tracking.
    """
    query : str
    steps  : list[PlanStep] = field(default_factory=list)

    
    # ── status helpers ────────────────────────────────────────────────────────

    @property
    def is_complete(self) ->bool:
        return all(s.status in (StepStatus.DONE , StepStatus.SKIPPED)
                   for s in self.steps)
    
    @property
    def has_failure(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in  self.steps)    #we will use this in later files
    

    @property
    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]
    
    @property
    def done_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.DONE]
    
    def get_step(self, step_id : str) -> Optional[PlanStep]:
        return next ((s for s in self.steps if s.step_id == step_id), None)
    
    def dependencies_met(self, step : PlanStep) -> bool:
        """True if all steps this step depends on are DONE."""
        return all (
            self.get_step(dep_id) is not None
            and self.get_step(dep_id).status == StepStatus.DONE
            for dep_id in step.depends_on
        )
    
    def print(self, title: str = "Plan") -> None:
        print(f"\n  {'─'*8} {title} {'─'*8}")
        for s in self.steps:
            deps = f"  (needs: {', '.join(s.depends_on)})" if s.depends_on else ""
            print(f"    [{s.step_id}] {s.goal}")
            print(f"           tool={s.tool}  status={s.status.name}{deps}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TOOL REGISTRY  —  simple deterministic tools (no real LLM/API calls)
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE : list[str] = [
    "RAG combines retrieval and generation to ground LLM outputs in external knowledge.",
    "Dense retrieval with FAISS enables fast approximate nearest-neighbour search.",
    "BM25 is a sparse retrieval method effective for keyword-heavy queries.",
    "Reranking with cross-encoders improves retrieval precision after initial recall.",
    "LangGraph enables stateful multi-agent workflows using a graph abstraction.",
    "Hallucination is reduced when the LLM is conditioned on retrieved documents.",
    "Sentence-transformers produce dense embeddings for semantic similarity search.",

]

def retrieval_tool(query : str, **_) -> str:
    """Returns the two most relevant docs (keyword overlap heuristic)."""
    q_words = set(query.lower().split())
    scored = sorted (KNOWLEDGE_BASE, 
                     key= lambda doc : len(set(doc.lower().split()) &q_words), reverse=True)
    return " | ".join(scored[:2])

def web_search(query: str, **_) -> str:
    """Simulated web search — returns a plausible result string."""
    return (
        f"[Web] Search results for '{query}': "
        "Recent benchmarks show RAG pipelines outperform vanilla LLMs on "
        "knowledge-intensive tasks by 15–30% on faithfulness metrics."
    )

def calculator(expression : str, **_)->str:
    """Evaluates a safe arithmetic expression."""
    #allow only number and basic operators
    try:
        safe = re.sub(r"[^0-9+\-*/().\s]", "", expression)
        result = eval(safe,{"__builtins__": {}})
        return str(round(result,4))
    except Exception as exc:
        return f"ERROR : {exc}"
    

def summarizer_tool(text: str, **_) -> str:
    """
    NEW in 07 — condenses a long intermediate result into a 1-2 sentence summary.
    In production this would call an LLM; here we truncate + label.
    """
    sentences = text.split(". ")

    summary = ". ".join(s.strip() for s in sentences[:2])

    if not summary.endswith("."):
        summary += "."

    return f"[Summary] {summary}"
    
TOOL_REGISTRY : dict[str, Any] = {
    "retrieval_tool" : retrieval_tool,
    "web_search"     : web_search,
    "calculator"     : calculator,
    "summarizer_tool": summarizer_tool,
}

def call_tool(tool_name : str, tool_input:dict) -> str:
    """Dispatch a tool call; returns error string on unknown tool."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return f"ERROR : unknown tool '{tool_name}'"
    return fn(**tool_input)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  STATIC PLANNER  —  LLM generates full plan upfront
# ─────────────────────────────────────────────────────────────────────────────
 
STATIC_PLAN_PROMPT = """\
You are a planning agent. Given a user query, produce a JSON plan.
 
Available tools: retrieval_tool, web_search, calculator, summarizer_tool
 
Rules:
- Break the query into 3-5 logical steps.
- Each step must have: step_id (s1, s2 ...), goal (string), tool (one of the available tools), depends_on (list of step_ids or empty list).
- The LAST step must always use summarizer_tool to condense the final answer.
- depends_on must only reference earlier step_ids.
- Return ONLY valid JSON — no markdown, no explanation.
 
Format:
[
  {{"step_id": "s1", "goal": "...", "tool": "retrieval_tool", "depends_on": []}},
  {{"step_id": "s2", "goal": "...", "tool": "web_search",     "depends_on": ["s1"]}},
  {{"step_id": "s3", "goal": "...", "tool": "summarizer_tool","depends_on": ["s1","s2"]}}
]
 
Query: {query}
"""
 
class StaticPlanner : 
    """
    Calls the LLM once to generate the full plan upfront.
    Falls back to a hardcoded plan if the LLM is unavailable.
    """
    def __init__(self, client : Groq, use_llm : bool = True):
        self.client = client
        self.use_llm = use_llm

    def plan(self, query:str) -> Plan:
        if self.use_llm:
            return self._llm_plan(query)
        return self._fallback_plan(query)
    

    def _llm_plan(self, query:str) -> Plan:
        prompt = STATIC_PLAN_PROMPT.format(query = query)
        
        try:
            resp = self.client.chat.completions.create(
                model=  GROQ_MODEL,
                messages=  [{"role" : "user" , "content" : prompt}],
                max_tokens  = 512,
                temperature= 0.2,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            steps = [
                PlanStep(
                    step_id=    d["step_id"],
                    goal =      d["goal"],
                    tool =      d["tool"],
                    depends_on= d.get("depends_on" , []),
                )
                for d in data
            ]
            return Plan(query =query, steps = steps)
        except Exception as exc:
            print(f"[StaticPlanner] LLM plan failed ({exc}) - using fallback.")
            return self._fallback_plan(query)
        
    def _fallback_plan(self, query: str) -> Plan:

        return Plan(
            query=query,
            steps=[
                PlanStep(
                    "s1",
                    "Retrieve relevant documents from knowledge base",
                    "retrieval_tool",
                    []
                ),

                PlanStep(
                    "s2",
                    "Search the web for recent information",
                    "web_search",
                    []
                ),

                PlanStep(
                    "s3",
                    "Calculate any numeric values if needed",
                    "calculator",
                    ["s1", "s2"]
                ),

                PlanStep(
                    "s4",
                    "Combine all findings into a final answer",
                    "summarizer_tool",
                    ["s1", "s2", "s3"]
                )
            ]
        )

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DYNAMIC REPLANNER  —  revises remaining steps after a failure
# ─────────────────────────────────────────────────────────────────────────────

REPLAN_PROMPT = """\
A step in the agent plan has FAILED.

Original query  :  {query}
Failed step     :  {failed_step_id} - {failed_goal} (tool :{failed_tool})
Error           :  {error}
Completed steps : {completed}

Revise ONLY the remaining pending steps.
Available tools : retrieval_tool, web_search, calculator, summarizer_tool
Return ONLY valid JSON (same format as before). No markdown.
"""


class DynamicReplanner:
    """
    Called mid-execution when a step fails.
    Asks the LLM to revise only the remaining PENDING steps.
    Falls back to a simple substitution strategy if LLM unavailable.
    """

    def __init__(self, client : Groq, use_llm:bool = True):
        self.client = client
        self.use_llm = use_llm

    def replan(self, plan:Plan, failed_step : PlanStep) ->Plan:
        if self.use_llm:
            return self._llm_replan(plan, failed_step)
        return self._fallback_replan(plan, failed_step)
    
    def _llm_replan(self, plan :Plan, failed_step : PlanStep) -> Plan:
        completed = [
            f"{s.step_id} : {s.goal}" for s in plan.done_steps
        ]
        prompt = REPLAN_PROMPT.format(
            query               =  plan.query,
            failed_step_id      =  failed_step.step_id,
            failed_goal         =  failed_step.goal,
            failed_tool         =  failed_step.tool,
            error               =  failed_step.error or "unknown error",
            completed           =  ", ".join(completed) or "none",
        )
        try:
            resp = self.client.chat.completions.create(
                model = GROQ_MODEL,
                messages=[{"role" : "user", "content" : prompt}],
                max_tokens=512,
                temperature= 0.2,
            )
            raw = resp.choices[0].message.content.strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            new_steps = [
                PlanStep(
                    step_id=    d["step_id"],
                    goal=       d["goal"],
                    tool=   d["tool"],
                    depends_on= d.get("depends_on" ,[]),

                )
                for d in data
            ]
            # keep completed steps, replace pending with revised ones
            revised = Plan(query = plan.query, steps=plan.done_steps + new_steps)
            return revised
        except Exception as exc:
            print(f"  [DynamicReplanner] LLM replan failed ({exc}) — using fallback.")
            return self._fallback_replan(plan, failed_step)
        
    def _fallback_replan(self, plan : Plan, failed_step : PlanStep) ->Plan:
        
        # choose fallback based on failed tool
        fallback_map = {
            "web_search": "retrieval_tool",
            "calculator": "retrieval_tool",
            "retrieval_tool": "web_search",
            "summarizer_tool": "retrieval_tool",
        }

        fallback_tool = fallback_map.get(
            failed_step.tool,
            "retrieval_tool"
        )

        new_pending = []

        for s in plan.pending_steps:
            new_pending.append(
                PlanStep(
                    step_id=  s.step_id + "_r",
                    goal = f"[Revised] {s.goal}",
                    tool = fallback_tool,
                    depends_on= [],
                )

            )
        # always ensure final summary exists
        if not any(
            s.tool == "summarizer_tool"
            for s in new_pending
        ):
            new_pending.append(
                PlanStep(
                    step_id= "s_final",
                    goal = "Summarise all available context",
                    tool = "summarizer_tool",
                    depends_on=[]
                )
            )
        return Plan(
            query= plan.query,
            steps = plan.done_steps + new_pending
        )
    
# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLAN EXECUTOR  —  walks the plan, calls tools, triggers replanning
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Full result of running a Plan to completion."""
    original_plan : Plan
    final_plan    : Plan            # may differ if replanning occur
    agent_steps   : list[AgentStep]
    final_answer  : str
    replanned     : bool = False
    adherence_score : float = 0.0 

class PlanExecutor:
    """
    Executes a Plan step by step.
    If a step fails, calls DynamicReplanner and continues.
    Builds an AgentTrace compatible with AgentEvaluator.
    """

    def __init__(self, replanner : DynamicReplanner):
        self.replanner = replanner
    
    def execute(self, plan : Plan , inject_failure_at : Optional[str] = None) -> ExecutionResult:
        """
        inject_failure_at: step_id to deliberately fail (for demo purposes).
        """
        original_plan = Plan(query=plan.query, steps=list(plan.steps))
        agent_steps : list[AgentStep] = []
        current_plan = plan
        replanned = False
        step_counter = 0
        context_parts : list[str] = []

        print(f"\n  Executing plan for: '{plan.query}'")

        while not current_plan.is_complete:
            # find first pending step whose dependencies are met
            ready =[
                s for s in current_plan.pending_steps
                if current_plan.dependencies_met(s)
            ]
            if not ready:
                print("  [Executor] No ready steps — possible dependency deadlock.")
                break

            step = ready[0]
            step.mark_running()
            step_counter += 1

            print(f"    -> [{step.step_id}]{step.goal}(tool : {step.tool})")

            # ── inject failure for demo ───────────────────────────────────────

            if step.step_id == inject_failure_at:
                error_msg = f"Simulated failure at step {step.step_id}"
                step.mark_failed(error_msg)
                print(f"     FAILED: {error_msg}")

                agent_steps.append(AgentStep(
                    step_index   =   step_counter,
                    thought      =   f"Attempting : {step.goal}",
                    action       =   step.tool,
                    action_input =  {"query" : current_plan.query},
                    observation  = f"ERROR : {error_msg}",
                ))

                #trigger replanning

                print(f" [Executor] Triggering DynamicReplaning")
                current_plan = self.replanner.replan(current_plan, step)
                replanned = True
                current_plan.print("Revised Plan")
                continue
            # ── normal execution ──────────────────────────────────────────────
            tool_input = {"query" : current_plan.query, "expression" : step.goal,
                          "text" : " ".join(context_parts) or current_plan.query}
            observation = call_tool(step.tool, tool_input)
            step.mark_done(observation)
            context_parts.append(observation)

            print(f"      {observation[:80]}{'…' if len(observation) > 80 else ''}")

            agent_steps.append(AgentStep(
                step_index      =   step_counter,
                thought         =   f"Executing step {step.step_id} : {step.goal}",
                action          =   step.tool,
                action_input    =   tool_input,
                observation     =   observation,
            ))     
        # ── build final answer from last DONE step ────────────────────────────
        done = current_plan.done_steps
        final_answer = done[-1].result if done else "No answer produced"

        # ── add final AgentStep ───────────────────────────────────────────────

        agent_steps.append(AgentStep(
            step_index=step_counter +1,
            thought= "Plan complete. Returning final answer",
            is_final = True,
        ))

        return ExecutionResult(
            original_plan = original_plan,
            final_plan    = current_plan,
            agent_steps   = agent_steps,
            final_answer  = final_answer,
            replanned     = replanned,
        )
    

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLAN ADHERENCE SCORE  —  new eval metric
# ─────────────────────────────────────────────────────────────────────────────

def plan_adherence_score(original_plan : Plan , final_plan : Plan) -> float:

    """
    Measures how closely execution followed the original plan.
 
    Score = (steps completed as originally planned) / (total original steps)
 
    - 1.0 : perfect adherence — every original step ran as planned
    - 0.5 : half the original steps ran; rest were replanned / skipped
    - 0.0 : nothing from original plan survived
 
    Why this matters:
      A low adherence score + high faithfulness = replanning worked well.
      A low adherence score + low faithfulness  = replanning failed.
    """

    original_ids = {s.step_id for s in original_plan.steps}
    final_done = {s.step_id for s in final_plan.done_steps}

    matched = original_ids & final_done
    score = len(matched) / len(original_ids) if original_ids else 0.0
    return round(score,2)
        
# ─────────────────────────────────────────────────────────────────────────────
# 7.  DEMO
# ─────────────────────────────────────────────────────────────────────────────

QUERY = "How does RAG  improve LLM accuracy and what retrieval methods does it use?"

GROUND_TRUTH = (
    "RAG improves LLM accuracy by retrieving relevant documents from a knowledge "
    "base and conditioning the model's output on them, reducing hallucinations. "
    "Common retrieval methods include dense retrieval with FAISS and sparse "
    "retrieval with BM25."

)

RETRIEVED_DOCS = [
    "RAG combines retrieval and generation to ground LLM outputs in external knowledge.",
    "Dense retrieval with FAISS enables fast approximate nearest-neighbour search.",
    "BM25 is a sparse retrieval method effective for keyword-heavy queries.",
]

def _divider(title: str = "") -> None:
    if title:
        print(f"\n{'─'*10} {title} {'─'*(47 - len(title))}")
    else:
        print("─" * 60)
 
 
def _print_eval(label: str, result: EvalResult) -> None:
    status = "PASSED" if result.passed else "FAILED"
    print(f"  {label:<22} │ "
          f"faith={result.faithfulness:.2f}  "
          f"rel={result.answer_relevance:.2f}  "
          f"traj={result.trajectory_score:.2f}  "
          f"→ {status}")

def main() -> None:
    print("=" * 60)
    print(" 07 · Multi-Step Task Planning Agent")
    print("=" * 60)

    api_key = os.getenv("GROQ_API_KEY", "")
    client = Groq(api_key=api_key) if api_key else Groq(api_key="dummy")
    use_llm = bool(api_key)
    use_judge = bool(api_key)

    evaluator = AgentEvaluator(groq_client=client, use_judge=use_judge)
    replanner = DynamicReplanner(client,use_llm=use_llm)
    executor = PlanExecutor(replanner)
    planner = StaticPlanner(client, use_llm=use_llm)

    # ── DEMO A: Static plan — no failures ────────────────────────────────────
    _divider("Static Plan  (no failure)")
    static_plan = planner.plan(QUERY)
    static_plan.print("Generated Plan")

    result_a = executor.execute(static_plan)
    adherence_a = plan_adherence_score(result_a.original_plan, result_a.final_plan)

    trace_a = AgentTrace(
        query           =   QUERY,
        ground_truth    =   GROUND_TRUTH,
        final_answer    =   result_a.final_answer,
        steps           =   result_a.agent_steps,
        retrieved_docs  =   RETRIEVED_DOCS,
        expected_tools  =   ["retrieval_tool" , "summarizer_tool"]   
    )
    evaluator.tracker.records.clear()
    eval_a = evaluator.evaluate_trace(trace_a)

    print(f"\n  Plan adherence : {adherence_a:.0%}")
    print(f"  Replanned      : {result_a.replanned}")
    _print_eval("Static (no failure)", eval_a)
 
    # ── DEMO B: Dynamic replan — inject failure at s2 ────────────────────────
    _divider("Dynamic Replan  (failure injected at s2)")
    dynamic_plan = planner.plan(QUERY)
    dynamic_plan.print("Original Plan")

    result_b = executor.execute(dynamic_plan,inject_failure_at="s2")
    adherence_b = plan_adherence_score(result_b.original_plan, result_b.final_plan)

    trace_b = AgentTrace(
        query                   =   QUERY,
        ground_truth            =   GROUND_TRUTH,
        final_answer            =   result_b.final_answer,
        steps                   =   result_b.agent_steps,
        retrieved_docs          =   RETRIEVED_DOCS,
        expected_tools          =   ["retrieval_tool", "summarizer_tool"],
    
    )

    evaluator.tracker.records.clear()
    eval_b = evaluator.evaluate_trace(trace_b)

    print(f"\n  Plan adherence : {adherence_b:.0%}")
    print(f"  Replanned      : {result_b.replanned}")
    _print_eval("Dynamic (with replan)", eval_b)

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    _divider()
    print(f"\n{'─'*68}")
    print(f"  {'Scenario':<28} │ {'Adherence':^10} │ {'Replanned':^10} │ {'Result':^10}")
    print(f"  {'─'*28}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")
    print(f"  {'Static (no failure)':<28} │ {adherence_a:^10.0%} │ {'No':^10} │ {'PASS' if eval_a.passed else 'FAIL':^10}")
    print(f"  {'Dynamic (with replan)':<28} │ {adherence_b:^10.0%} │ {'Yes':^10} │ {'PASS' if eval_b.passed else 'FAIL':^10}")
    print(f"{'─'*68}\n")
 
    # ── KEY INSIGHT ───────────────────────────────────────────────────────────
    print("  Key insight:")
    #print("  ┌─────────────────────────────────────────────────────────────┐")
    print("   Low adherence + high faithfulness = replanning worked well  ")
    print("   Low adherence + low  faithfulness = replanning also failed  ")
    #print("  └─────────────────────────────────────────────────────────────┘\n")
 
 
if __name__ == "__main__":
    main()






        