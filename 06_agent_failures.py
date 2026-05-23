"""
06_agent_failures.py
====================
Agent Failure Modes & Recovery
Stack: Python · Groq · sentence-transformers

Covers:
  - Failure mode simulation (loop, hallucination, empty thought,
                             observation ignored, early termination)
  - Guardrails  (loop detection, tool validation, max steps, thought check)
  - Recovery    (retry with rephrased prompt, fallback tool, partial answer)
  - Before/after eval scores using agent_evaluation.py
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

# ── third party ──────────────────────────────────────────────────────────────
from groq import Groq

# ── local ────────────────────────────────────────────────────────────────────
from agent_evaluation import (
    AgentStep,
    AgentTrace,
    AgentEvaluator,
    EvalResult,
    THRESHOLDS,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODEL  = "llama-3.3-70b-versatile"
MAX_STEPS   = 10
RETRY_LIMIT = 2

AVAILABLE_TOOLS: set[str] = {
    "retrieval_tool",
    "web_search",
    "calculator",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FAILURE TYPES
# ─────────────────────────────────────────────────────────────────────────────

class FailureType(Enum):
    INFINITE_LOOP       = auto()   # repeats the same action forever
    TOOL_HALLUCINATION  = auto()   # calls a tool that does not exist
    EMPTY_THOUGHT       = auto()   # acts without any reasoning
    OBSERVATION_IGNORED = auto()   # receives tool result but ignores it
    EARLY_TERMINATION   = auto()   # gives up before solving the task


@dataclass
class FailureConfig:
    """Controls how and where a failure is injected into a trace."""
    failure_type : FailureType
    loop_repeats : int = 4                    # repetitions for INFINITE_LOOP
    inject_at    : int = 0                    # step index where failure starts
    ghost_tool   : str = "llm_brain_tool"     # fake tool name for TOOL_HALLUCINATION


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FAULTY AGENT  —  injects a specific failure into an AgentTrace
# ─────────────────────────────────────────────────────────────────────────────

class FaultyAgent:
    """
    Produces a broken AgentTrace for each FailureType.
    No LLM calls — failures are injected deterministically so they are
    fully reproducible and cost-free.
    """

    def __init__(self, query: str, ground_truth: str, retrieved_docs: list[str]):
        self.query         = query
        self.ground_truth  = ground_truth
        self.retrieved_docs = retrieved_docs

    # ── public ────────────────────────────────────────────────────────────────

    def build_trace(self, cfg: FailureConfig) -> AgentTrace:
        builders = {
            FailureType.INFINITE_LOOP       : self._infinite_loop,
            FailureType.TOOL_HALLUCINATION  : self._tool_hallucination,
            FailureType.EMPTY_THOUGHT       : self._empty_thought,
            FailureType.OBSERVATION_IGNORED : self._observation_ignored,
            FailureType.EARLY_TERMINATION   : self._early_termination,
        }
        steps, answer = builders[cfg.failure_type](cfg)
        return AgentTrace(
            query          = self.query,
            ground_truth   = self.ground_truth,
            final_answer   = answer,
            steps          = steps,
            retrieved_docs = self.retrieved_docs,
            expected_tools = ["retrieval_tool"],
        )

    # ── failure builders ──────────────────────────────────────────────────────

    def _infinite_loop(self, cfg: FailureConfig) -> tuple[list[AgentStep], str]:
        """Same action repeated cfg.loop_repeats times — no progress."""
        steps = []
        for i in range(cfg.loop_repeats):
            steps.append(AgentStep(
                step_index   = i,
                thought      = "I should search for more information.",
                action       = "retrieval_tool",
                action_input = {"query": self.query},
                observation  = "Found 3 documents about the topic.",
            ))
        # no final step — agent is stuck
        return steps, "I need to search more to find the answer."

    def _tool_hallucination(self, cfg: FailureConfig) -> tuple[list[AgentStep], str]:
        """Calls a tool that does not exist in AVAILABLE_TOOLS."""
        steps = [
            AgentStep(
                step_index   = 0,
                thought      = "I will use my built-in reasoning engine to answer directly.",
                action       = cfg.ghost_tool,          # ← does not exist
                action_input = {"query": self.query},
                observation  = "ERROR: tool not found",
            ),
            AgentStep(
                step_index = 1,
                thought    = "I got an answer from my reasoning engine.",
                is_final   = True,
            ),
        ]
        return steps, "RAG is a technique that makes LLMs smarter by giving them memory."

    def _empty_thought(self, cfg: FailureConfig) -> tuple[list[AgentStep], str]:
        """Agent acts without providing any reasoning."""
        steps = [
            AgentStep(
                step_index   = 0,
                thought      = "",                      # ← empty
                action       = "retrieval_tool",
                action_input = {"query": self.query},
                observation  = "Found 3 relevant documents.",
            ),
            AgentStep(
                step_index = 1,
                thought    = "",                        # ← empty again
                is_final   = True,
            ),
        ]
        return steps, "RAG combines retrieval and generation."

    def _observation_ignored(self, cfg: FailureConfig) -> tuple[list[AgentStep], str]:
        """Agent receives a rich observation but ignores it entirely."""
        steps = [
            AgentStep(
                step_index   = 0,
                thought      = "Let me retrieve some documents.",
                action       = "retrieval_tool",
                action_input = {"query": self.query},
                observation  = (                        # rich result — ignored below
                    "RAG grounds LLM outputs in external knowledge. "
                    "It retrieves documents via FAISS or BM25 and conditions "
                    "the generation step on them, reducing hallucinations."
                ),
            ),
            AgentStep(
                step_index = 1,
                thought    = "I already know what RAG is from my training data.",
                is_final   = True,
            ),
        ]
        # answer is generic — ignores the specific observation above
        return steps, "RAG is a popular AI technique used in chatbots."

    def _early_termination(self, cfg: FailureConfig) -> tuple[list[AgentStep], str]:
        """Agent gives up immediately without attempting retrieval."""
        steps = [
            AgentStep(
                step_index = 0,
                thought    = "This question is too complex. I cannot answer it.",
                is_final   = True,
            ),
        ]
        return steps, "I don't have enough information to answer this question."


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GUARDRAILS  —  detect failures before / during execution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    triggered : bool
    reason    : str  = ""
    details   : dict = field(default_factory=dict)


class LoopGuard:
    """Flags traces where the same action appears more than threshold times."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold

    def check(self, steps: list[AgentStep]) -> GuardrailResult:
        from collections import Counter
        actions = [s.action for s in steps if s.action and not s.is_final]
        counts  = Counter(actions)
        loops   = {a: n for a, n in counts.items() if n >= self.threshold}
        if loops:
            return GuardrailResult(
                triggered = True,
                reason    = f"Action(s) repeated ≥ {self.threshold}×",
                details   = {"repeated_actions": loops},
            )
        return GuardrailResult(triggered=False)


class ToolValidator:
    """Flags calls to tools not in the allowed registry."""

    def __init__(self, available_tools: set[str] = AVAILABLE_TOOLS):
        self.available_tools = available_tools

    def check(self, steps: list[AgentStep]) -> GuardrailResult:
        ghost_calls = [
            s.action for s in steps
            if s.action and s.action not in self.available_tools and not s.is_final
        ]
        if ghost_calls:
            return GuardrailResult(
                triggered = True,
                reason    = "Call(s) to unregistered tool(s)",
                details   = {"ghost_tools": ghost_calls},
            )
        return GuardrailResult(triggered=False)


class MaxStepsGuard:
    """Flags traces that exceed the maximum allowed step count."""

    def __init__(self, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps

    def check(self, steps: list[AgentStep]) -> GuardrailResult:
        if len(steps) > self.max_steps:
            return GuardrailResult(
                triggered = True,
                reason    = f"Step count {len(steps)} exceeds max {self.max_steps}",
                details   = {"n_steps": len(steps), "max_steps": self.max_steps},
            )
        return GuardrailResult(triggered=False)


class ThoughtGuard:
    """Flags traces where any non-final step has an empty thought."""

    def check(self, steps: list[AgentStep]) -> GuardrailResult:
        empty = [s.step_index for s in steps if not s.thought.strip()]
        if empty:
            return GuardrailResult(
                triggered = True,
                reason    = "Empty thought(s) detected",
                details   = {"empty_at_steps": empty},
            )
        return GuardrailResult(triggered=False)


class FinalAnswerGuard:
    """Flags traces with no final step or more than one final step."""

    def check(self, steps: list[AgentStep]) -> GuardrailResult:
        finals = [s for s in steps if s.is_final]
        if len(finals) == 0:
            return GuardrailResult(triggered=True, reason="No final answer step found")
        if len(finals) > 1:
            return GuardrailResult(
                triggered = True,
                reason    = f"{len(finals)} final steps found — expected exactly 1",
            )
        return GuardrailResult(triggered=False)


@dataclass
class GuardrailReport:
    """Aggregated result from running all guardrails on a trace."""
    any_triggered : bool
    results       : dict[str, GuardrailResult]

    def print(self) -> None:
        status = "⚠  GUARDRAILS TRIGGERED" if self.any_triggered else "✓  All guardrails passed"
        print(f"\n  {status}")
        for name, gr in self.results.items():
            icon = "✗" if gr.triggered else "✓"
            line = f"    {icon} {name}"
            if gr.triggered:
                line += f" → {gr.reason}"
                if gr.details:
                    line += f"  {gr.details}"
            print(line)


class GuardrailSuite:
    """Runs all guardrails and returns a single aggregated report."""

    def __init__(
        self,
        loop_threshold : int       = 3,
        max_steps      : int       = MAX_STEPS,
        available_tools: set[str]  = AVAILABLE_TOOLS,
    ):
        self.guards = {
            "LoopGuard"       : LoopGuard(loop_threshold),
            "ToolValidator"   : ToolValidator(available_tools),
            "MaxStepsGuard"   : MaxStepsGuard(max_steps),
            "ThoughtGuard"    : ThoughtGuard(),
            "FinalAnswerGuard": FinalAnswerGuard(),
        }

    def run(self, trace: AgentTrace) -> GuardrailReport:
        results = {name: g.check(trace.steps) for name, g in self.guards.items()}
        return GuardrailReport(
            any_triggered = any(r.triggered for r in results.values()),
            results       = results,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RECOVERY AGENT  —  detects failure, applies strategy, returns new trace
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryStrategy(Enum):
    RETRY_REPHRASED  = auto()   # rephrase the query and try again
    FALLBACK_TOOL    = auto()   # switch to a different tool
    PARTIAL_ANSWER   = auto()   # return what we have with a confidence flag


@dataclass
class RecoveryResult:
    strategy_used  : RecoveryStrategy
    recovered      : bool
    trace          : AgentTrace
    confidence     : float = 1.0     # 0–1, lowered for partial answers
    recovery_note  : str   = ""


class RecoveryAgent:
    """
    Wraps a FaultyAgent trace, runs guardrails, picks a recovery
    strategy, and returns a corrected AgentTrace.
    """

    # Maps each failure type to the most appropriate recovery strategy
    STRATEGY_MAP: dict[FailureType, RecoveryStrategy] = {
        FailureType.INFINITE_LOOP       : RecoveryStrategy.RETRY_REPHRASED,
        FailureType.TOOL_HALLUCINATION  : RecoveryStrategy.FALLBACK_TOOL,
        FailureType.EMPTY_THOUGHT       : RecoveryStrategy.RETRY_REPHRASED,
        FailureType.OBSERVATION_IGNORED : RecoveryStrategy.RETRY_REPHRASED,
        FailureType.EARLY_TERMINATION   : RecoveryStrategy.PARTIAL_ANSWER,
    }

    def __init__(self, guardrails: GuardrailSuite):
        self.guardrails = guardrails

    def recover(self, faulty_trace: AgentTrace, failure_type: FailureType) -> RecoveryResult:
        strategy = self.STRATEGY_MAP[failure_type]

        match strategy:

            case RecoveryStrategy.RETRY_REPHRASED:

                return self._retry_rephrased(faulty_trace)

            case RecoveryStrategy.FALLBACK_TOOL:

                return self._fallback_tool(faulty_trace)

            case RecoveryStrategy.PARTIAL_ANSWER:

                return self._partial_answer(faulty_trace)

            case _:

                raise ValueError(f"Unknown recovery strategy: {strategy}")

    # ── strategies ────────────────────────────────────────────────────────────

    def _retry_rephrased(self, trace: AgentTrace) -> RecoveryResult:
        """Re-run with a rephrased query; build a clean corrected trace."""
        rephrased = f"Please explain clearly: {trace.query}"
        corrected_steps = [
            AgentStep(
                step_index   = 0,
                thought      = f"Previous attempt failed. Retrying with rephrased query: '{rephrased}'",
                action       = "retrieval_tool",
                action_input = {"query": rephrased},
                observation  = (
                    "RAG combines a retrieval system with a language model. "
                    "It fetches relevant documents and conditions generation on them."
                ),
            ),
            AgentStep(
                step_index = 1,
                thought    = (
                    "The retrieval returned useful context. "
                    "I can now synthesise a grounded answer."
                ),
                is_final   = True,
            ),
        ]
        recovered_trace = AgentTrace(
            query          = trace.query,
            ground_truth   = trace.ground_truth,
            final_answer   = (
                "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving "
                "relevant documents from a knowledge base via FAISS or BM25, then "
                "conditioning the model's output on those documents. "
                "This reduces hallucinations and keeps responses grounded in "
                "verifiable sources."
            ),
            steps          = corrected_steps,
            retrieved_docs = trace.retrieved_docs,
            expected_tools = trace.expected_tools,
        )
        return RecoveryResult(
            strategy_used = RecoveryStrategy.RETRY_REPHRASED,
            recovered     = True,
            trace         = recovered_trace,
            confidence    = 0.90,
            recovery_note = f"Rephrased query to: '{rephrased}'",
        )

    def _fallback_tool(self, trace: AgentTrace) -> RecoveryResult:
        """Replace ghost tool call with a valid fallback tool."""
        corrected_steps = [
            AgentStep(
                step_index   = 0,
                thought      = (
                    "The requested tool is not available. "
                    "Falling back to 'retrieval_tool'."
                ),
                action       = "retrieval_tool",        # ← valid tool
                action_input = {"query": trace.query},
                observation  = (
                    "RAG grounds LLM outputs in external knowledge by retrieving "
                    "documents and conditioning generation on them."
                ),
            ),
            AgentStep(
                step_index = 1,
                thought    = "Fallback retrieval succeeded. Composing final answer.",
                is_final   = True,
            ),
        ]
        recovered_trace = AgentTrace(
            query          = trace.query,
            ground_truth   = trace.ground_truth,
            final_answer   = (
                "RAG (Retrieval-Augmented Generation) pairs a retrieval system "
                "with a language model. Documents are fetched from a knowledge base "
                "and used to condition the LLM's output, reducing hallucinations "
                "and improving factual accuracy."
            ),
            steps          = corrected_steps,
            retrieved_docs = trace.retrieved_docs,
            expected_tools = trace.expected_tools,
        )
        return RecoveryResult(
            strategy_used = RecoveryStrategy.FALLBACK_TOOL,
            recovered     = True,
            trace         = recovered_trace,
            confidence    = 0.85,
            recovery_note = "Replaced ghost tool with 'retrieval_tool'",
        )

    def _partial_answer(self, trace: AgentTrace) -> RecoveryResult:
        """Return the best partial answer available with a low confidence flag."""
        corrected_steps = [
            AgentStep(
                step_index   = 0,
                thought      = (
                    "Agent terminated early. Attempting to construct a partial "
                    "answer from available context."
                ),
                action       = "retrieval_tool",
                action_input = {"query": trace.query},
                observation  = "Retrieved partial context about RAG.",
            ),
            AgentStep(
                step_index = 1,
                thought    = "Partial context retrieved. Returning best-effort answer.",
                is_final   = True,
            ),
        ]
        recovered_trace = AgentTrace(
            query          = trace.query,
            ground_truth   = trace.ground_truth,
            final_answer   = (
                "[Partial — low confidence] RAG stands for Retrieval-Augmented "
                "Generation. It combines retrieval and generation to ground LLM "
                "outputs in external documents."
            ),
            steps          = corrected_steps,
            retrieved_docs = trace.retrieved_docs,
            expected_tools = trace.expected_tools,
        )
        return RecoveryResult(
            strategy_used = RecoveryStrategy.PARTIAL_ANSWER,
            recovered     = True,
            trace         = recovered_trace,
            confidence    = 0.55,
            recovery_note = "Returned partial answer — agent had terminated early",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DEMO
# ─────────────────────────────────────────────────────────────────────────────

QUERY = "What is Retrieval-Augmented Generation and why is it useful?"

GROUND_TRUTH = (
    "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
    "language model. It fetches relevant documents from a knowledge base and "
    "conditions the LLM's output on them, reducing hallucinations and keeping "
    "responses grounded in verifiable sources."
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
    print(f"  {label:<20} │ "
          f"faith={result.faithfulness:.2f}  "
          f"rel={result.answer_relevance:.2f}  "
          f"traj={result.trajectory_score:.2f}  "
          f"→ {status}")


def main() -> None:
    print("=" * 60)
    print(" 06 · Agent Failure Modes & Recovery")
    print("=" * 60)

    api_key  = os.getenv("GROQ_API_KEY", "")
    client   = Groq(api_key=api_key) if api_key else Groq(api_key="dummy")
    use_judge = bool(api_key)

    evaluator  = AgentEvaluator(groq_client=client, use_judge=use_judge)
    guardrails = GuardrailSuite()
    recovery   = RecoveryAgent(guardrails)
    faulty     = FaultyAgent(QUERY, GROUND_TRUTH, RETRIEVED_DOCS)

    failure_configs = [
        FailureConfig(FailureType.INFINITE_LOOP,       loop_repeats=4),
        FailureConfig(FailureType.TOOL_HALLUCINATION),
        FailureConfig(FailureType.EMPTY_THOUGHT),
        FailureConfig(FailureType.OBSERVATION_IGNORED),
        FailureConfig(FailureType.EARLY_TERMINATION),
    ]

    summary_rows: list[tuple[str, EvalResult, EvalResult]] = []

    for cfg in failure_configs:
        name = cfg.failure_type.name.replace("_", " ").title()
        _divider(name)

        # ── build faulty trace ──────────────────────────────────────────────
        bad_trace   = faulty.build_trace(cfg)

        # ── run guardrails ──────────────────────────────────────────────────
        gr_report   = guardrails.run(bad_trace)
        gr_report.print()

        # ── eval faulty trace ───────────────────────────────────────────────
        evaluator.tracker.records.clear()
        bad_result  = evaluator.evaluate_trace(bad_trace)

        # ── recover ─────────────────────────────────────────────────────────
        rec         = recovery.recover(bad_trace, cfg.failure_type)
        strategy    = rec.strategy_used.name.replace("_", " ").title()
        print(f"\n  Recovery  : {strategy}")
        print(f"  Note      : {rec.recovery_note}")
        print(f"  Confidence: {rec.confidence:.0%}")

        # ── eval recovered trace ─────────────────────────────────────────────
        evaluator.tracker.records.clear()
        good_result = evaluator.evaluate_trace(rec.trace)

        _print_eval("Before recovery", bad_result)
        _print_eval("After  recovery", good_result)

        summary_rows.append((name, bad_result, good_result))

    # ── summary table ────────────────────────────────────────────────────────
    _divider()
    print(f"\n{'─'*60}")
    print(f"  {'Failure':<25} │ {'Before':^20} │ {'After':^10}")
    print(f"  {'─'*25}─┼─{'─'*20}─┼─{'─'*10}")

    for name, bad, good in summary_rows:
        b_faith = f"faith={bad.faithfulness:.2f}"
        g_faith = f"faith={good.faithfulness:.2f}"
        b_pass  = "PASS" if bad.passed  else "FAIL"
        g_pass  = "PASS" if good.passed else "FAIL"
        print(f"  {name:<25} │ {b_faith}  {b_pass:>4}   │ {g_faith}  {g_pass:>4}")

    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()