"""
05_agent_evaluation.py
======================
Agent Evaluation Framework
Stack: Python · Groq · FAISS · BM25 · sentence-transformers

Covers:
  - Token / cost tracking   (per-call accounting, Groq pricing table)
  - Trajectory evaluation   (step-level correctness)
  - Tool-use evaluation     (precision / recall on tool calls)
  - Response quality        (faithfulness, answer relevance, context precision)
  - Retrieval evaluation    (hit-rate, MRR, NDCG)
  - LLM-as-judge            (Groq-powered rubric scoring)
  - Regression harness      (baseline comparison, pass/fail thresholds)
"""

import json
import math
import time
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODEL  = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"

MODEL_COSTS: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "gemma2-9b-it":            {"input": 0.20, "output": 0.20},
    "mixtral-8x7b-32768":      {"input": 0.24, "output": 0.24},
}

THRESHOLDS = {
    "faithfulness":       0.70,
    "answer_relevance":   0.75,
    "context_precision":  0.60,
    "tool_precision":     0.80,
    "tool_recall":        0.80,
    "trajectory_score":   0.70,
    "retrieval_hit_rate": 0.70,
    "retrieval_mrr":      0.50,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    step_index:   int
    thought:      str
    action:       Optional[str] = None
    action_input: Optional[Any] = None
    observation:  Optional[str] = None
    is_final:     bool          = False


@dataclass
class AgentTrace:
    query:          str
    ground_truth:   str
    final_answer:   str
    steps:          list[AgentStep]
    retrieved_docs: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    query: str

    # Response quality
    faithfulness:         float = 0.0
    faithfulness_details: list  = field(default_factory=list)
    unsupported_claims:   list  = field(default_factory=list)
    answer_relevance:     float = 0.0
    context_precision:    float = 0.0

    # Tool use
    tool_precision: float = 0.0
    tool_recall:    float = 0.0
    tool_f1:        float = 0.0

    # Trajectory
    trajectory_score: float = 0.0

    # Retrieval
    retrieval_hit_rate: float = 0.0
    retrieval_mrr:      float = 0.0
    retrieval_ndcg:     float = 0.0

    # LLM judge
    judge_score:    float = 0.0
    judge_feedback: str   = ""

    # Cost / tokens
    total_tokens: int   = 0
    cost_usd:     float = 0.0

    # Meta
    passed:    bool  = False
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EMBEDDING HELPER
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingHelper:

    def __init__(self, model_name: str = EMBED_MODEL):
        print(f"[EmbeddingHelper] Loading '{model_name}' …")
        self.model = SentenceTransformer(model_name)

    def cosine(self, a: str, b: str) -> float:
        ea, eb = self.model.encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(ea, eb))

    def cosine_batch(self, queries: list[str], docs: list[str]) -> np.ndarray:
        eq = self.model.encode(queries, convert_to_tensor=True)
        ed = self.model.encode(docs,    convert_to_tensor=True)
        return util.cos_sim(eq, ed).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TOKEN / COST TRACKER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    model:         str
    input_tokens:  int
    output_tokens: int
    latency_s:     float
    label:         str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        costs = MODEL_COSTS.get(self.model, {"input": 0.0, "output": 0.0})
        return (self.input_tokens  / 1_000_000 * costs["input"] +
                self.output_tokens / 1_000_000 * costs["output"])


class TokenCostTracker:

    def __init__(self):
        self.records: list[CallRecord] = []

    def record_from_response(self, response: Any, latency_s: float,
                              label: str = "") -> CallRecord:
        usage = response.usage
        rec   = CallRecord(
            model         = response.model,
            input_tokens  = usage.prompt_tokens,
            output_tokens = usage.completion_tokens,
            latency_s     = latency_s,
            label         = label,
        )
        self.records.append(rec)
        return rec

    def wrap_client(self, client: Any, label: str = "") -> Any:
        original = client.chat.completions.create

        def _tracked(*args, **kwargs):
            t0   = time.perf_counter()
            resp = original(*args, **kwargs)
            self.record_from_response(resp, time.perf_counter() - t0, label)
            return resp

        client.chat.completions.create = _tracked
        return client

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 6)

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {}
        by_label: dict[str, list[CallRecord]] = defaultdict(list)
        for r in self.records:
            by_label[r.label or "unlabelled"].append(r)

        label_stats = {}
        for lbl, recs in by_label.items():
            label_stats[lbl] = {
                "calls":         len(recs),
                "input_tokens":  sum(r.input_tokens  for r in recs),
                "output_tokens": sum(r.output_tokens for r in recs),
                "cost_usd":      round(sum(r.cost_usd for r in recs), 6),
                "avg_latency_s": round(sum(r.latency_s for r in recs) / len(recs), 3),
            }
        return {
            "total_calls":    len(self.records),
            "total_tokens":   self.total_tokens(),
            "total_cost_usd": self.total_cost_usd(),
            "by_label":       label_stats,
        }

    def print_summary(self) -> None:
        s = self.summary()
        if not s:
            print("[TokenCostTracker] No calls recorded.")
            return
        print(f"\n{'─'*54}")
        print(f"  Token / Cost Summary")
        print(f"{'─'*54}")
        print(f"  Total calls   : {s['total_calls']}")
        print(f"  Total tokens  : {s['total_tokens']:,}")
        print(f"  Total cost    : ${s['total_cost_usd']:.6f}")
        print(f"{'─'*54}")
        for lbl, st in s["by_label"].items():
            print(f"  [{lbl}]")
            print(f"    calls={st['calls']}  in={st['input_tokens']:,}  "
                  f"out={st['output_tokens']:,}  "
                  f"cost=${st['cost_usd']:.6f}  "
                  f"avg_lat={st['avg_latency_s']}s")
        print(f"{'─'*54}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RESPONSE-QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

class ResponseQualityEvaluator:

    def __init__(self, embedder: EmbeddingHelper):
        self.embedder = embedder

    @staticmethod
    def _sentences(text: str) -> list[str]:
        import re
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def faithfulness(self, answer: str, context_docs: list[str]) -> dict:
        """
        Per-claim faithfulness check.
        Each sentence in the answer is checked against every retrieved doc
        individually; the best-matching doc wins per claim.
        Returns score, per-claim details, and a flat unsupported_claims list.
        """
        if not answer or not context_docs:
            return {"score": 0.0, "details": [], "unsupported_claims": []}

        claims    = self._sentences(answer)
        supported = 0
        details   = []

        for claim in claims:
            best_doc = None
            best_sim = 0.0
            for doc in context_docs:
                sim = self.embedder.cosine(claim, doc)
                if sim > best_sim:
                    best_sim = sim
                    best_doc = doc
            is_supported = best_sim >= 0.45
            if is_supported:
                supported += 1
            details.append({
                "claim":      claim,
                "supported":  is_supported,
                "similarity": round(best_sim, 3),
                "evidence":   best_doc,
            })

        score = supported / len(claims) if claims else 0.0
        return {
            "score":              round(score, 3),
            "details":            details,
            "unsupported_claims": [d["claim"] for d in details if not d["supported"]],
        }

    def answer_relevance(self, answer: str, query: str) -> float:
        return self.embedder.cosine(answer, query)

    def context_precision(self, query: str, context_docs: list[str],
                          relevance_threshold: float = 0.40) -> float:
        if not context_docs:
            return 0.0
        relevant = sum(
            1 for doc in context_docs
            if self.embedder.cosine(query, doc) >= relevance_threshold
        )
        return relevant / len(context_docs)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TOOL-USE EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class ToolUseEvaluator:

    def evaluate(self, trace: AgentTrace) -> dict[str, float]:
        predicted = [s.action for s in trace.steps if s.action and not s.is_final]
        expected  = trace.expected_tools

        if not expected and not predicted:
            return {"tool_precision": 1.0, "tool_recall": 1.0,
                    "tool_f1": 1.0, "tool_order_acc": 1.0}

        pred_set = set(predicted)
        exp_set  = set(expected)
        inter    = pred_set & exp_set

        precision = len(inter) / len(pred_set) if pred_set else 0.0
        recall    = len(inter) / len(exp_set)  if exp_set  else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)

        correct_order = sum(p == e for p, e in zip(predicted, expected))
        order_acc     = correct_order / max(len(predicted), len(expected), 1)

        return {
            "tool_precision":  precision,
            "tool_recall":     recall,
            "tool_f1":         f1,
            "tool_order_acc":  order_acc,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAJECTORY EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryEvaluator:

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    def evaluate(self, trace: AgentTrace) -> dict[str, float]:
        steps = trace.steps

        # 1. thought coverage
        empty_thoughts = sum(1 for s in steps if not s.thought.strip())
        thought_score  = 1.0 - empty_thoughts / max(len(steps), 1)

        # 2. observation coverage
        action_steps = [s for s in steps if s.action and not s.is_final]
        missing_obs  = sum(1 for s in action_steps if not s.observation)
        obs_score    = 1.0 - missing_obs / max(len(action_steps), 1)

        # 3. loop detection
        actions        = [s.action for s in action_steps]
        unique_actions = len(set(actions))
        loop_score     = min(1.0, unique_actions / max(len(actions), 1))

        # 4. final-answer check
        finals      = [s for s in steps if s.is_final]
        final_score = 1.0 if len(finals) == 1 else 0.0

        # 5. length penalty
        length_score = (1.0 if len(steps) <= self.max_steps
                        else max(0.0, 1.0 - (len(steps) - self.max_steps) / self.max_steps))

        weights          = [0.25, 0.25, 0.20, 0.20, 0.10]
        scores           = [thought_score, obs_score, loop_score, final_score, length_score]
        trajectory_score = sum(w * s for w, s in zip(weights, scores))

        return {
            "trajectory_score":     trajectory_score,
            "thought_coverage":     thought_score,
            "observation_coverage": obs_score,
            "loop_avoidance":       loop_score,
            "final_answer_valid":   final_score,
            "length_score":         length_score,
            "n_steps":              len(steps),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RETRIEVAL EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEvaluator:

    def __init__(self, embedder: EmbeddingHelper):
        self.embedder = embedder

    @staticmethod
    def hit_rate(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
        return 1.0 if any(i in relevant_ids for i in retrieved_ids) else 0.0

    @staticmethod
    def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
        gains = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids[:k]]
        dcg   = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(gains, reverse=True)
        idcg  = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        return dcg / idcg if idcg else 0.0

    def evaluate_dataset(
        self,
        queries:      list[str],
        corpus:       list[str],
        relevant_ids: list[set[int]],
        k:            int = 5,
    ) -> dict[str, float]:
        sims                    = self.embedder.cosine_batch(queries, corpus)
        hit_rates, mrrs, ndcgs = [], [], []

        for row, rel in zip(sims, relevant_ids):
            ranked = list(np.argsort(-row))
            hit_rates.append(self.hit_rate(ranked[:k], rel))
            mrrs.append(self.mrr(ranked[:k], rel))
            ndcgs.append(self.ndcg(ranked[:k], rel, k))

        return {
            f"retrieval_hit_rate@{k}": float(np.mean(hit_rates)),
            f"retrieval_mrr@{k}":      float(np.mean(mrrs)),
            f"retrieval_ndcg@{k}":     float(np.mean(ndcgs)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LLM-AS-JUDGE
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = textwrap.dedent("""\
    You are an expert evaluator of AI agent responses.
    Evaluate the agent's final answer against the reference answer using this rubric:

    Score 1 – Completely wrong or hallucinated
    Score 2 – Partially correct but major gaps
    Score 3 – Mostly correct with minor issues
    Score 4 – Correct and well-reasoned
    Score 5 – Perfect: accurate, concise, fully grounded in context

    Respond ONLY in this JSON format (no extra text):
    {
      "score": <int 1-5>,
      "feedback": "<one sentence>"
    }
""")

JUDGE_USER_TEMPLATE = textwrap.dedent("""\
    Query:            {query}
    Reference Answer: {ground_truth}
    Agent Answer:     {final_answer}
    Retrieved Context (first 600 chars): {context}
""")


class LLMJudge:

    def __init__(self, client: Groq, model: str = GROQ_MODEL):
        self.client = client
        self.model  = model

    def evaluate(self, trace: AgentTrace) -> dict[str, Any]:
        context_snippet = " | ".join(trace.retrieved_docs)[:600]
        user_msg = JUDGE_USER_TEMPLATE.format(
            query        = trace.query,
            ground_truth = trace.ground_truth,
            final_answer = trace.final_answer,
            context      = context_snippet,
        )
        try:
            resp = self.client.chat.completions.create(
                model    = self.model,
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.0,
                max_tokens  = 150,
            )
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            return {
                "judge_score":    data.get("score", 0) / 5.0,
                "judge_feedback": data.get("feedback", ""),
            }
        except Exception as exc:
            return {"judge_score": 0.0, "judge_feedback": f"[Judge error] {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class AgentEvaluator:

    def __init__(
        self,
        groq_client: Groq,
        embed_model: str              = EMBED_MODEL,
        thresholds:  dict[str, float] = THRESHOLDS,
        use_judge:   bool             = True,
        max_steps:   int              = 10,
    ):
        self.thresholds = thresholds
        self.use_judge  = use_judge

        self.tracker   = TokenCostTracker()
        groq_client    = self.tracker.wrap_client(groq_client, label="judge")

        self.embedder  = EmbeddingHelper(embed_model)
        self.rq_eval   = ResponseQualityEvaluator(self.embedder)
        self.tool_eval = ToolUseEvaluator()
        self.traj_eval = TrajectoryEvaluator(max_steps=max_steps)
        self.judge     = LLMJudge(groq_client) if use_judge else None

    def evaluate_trace(self, trace: AgentTrace) -> EvalResult:
        t0     = time.perf_counter()
        result = EvalResult(query=trace.query)

        # faithfulness — returns dict, unpack directly
        faith                       = self.rq_eval.faithfulness(trace.final_answer, trace.retrieved_docs)
        result.faithfulness         = faith["score"]
        result.faithfulness_details = faith["details"]
        result.unsupported_claims   = faith["unsupported_claims"]

        # answer relevance and context precision — called directly
        result.answer_relevance  = self.rq_eval.answer_relevance(trace.final_answer, trace.query)
        result.context_precision = self.rq_eval.context_precision(trace.query, trace.retrieved_docs)

        # tool use
        tu = self.tool_eval.evaluate(trace)
        result.tool_precision = tu["tool_precision"]
        result.tool_recall    = tu["tool_recall"]
        result.tool_f1        = tu["tool_f1"]

        # trajectory
        tr = self.traj_eval.evaluate(trace)
        result.trajectory_score = tr["trajectory_score"]

        # LLM judge
        if self.judge:
            jd = self.judge.evaluate(trace)
            result.judge_score    = jd["judge_score"]
            result.judge_feedback = jd["judge_feedback"]

        result.latency_s    = time.perf_counter() - t0
        result.total_tokens = self.tracker.total_tokens()
        result.cost_usd     = self.tracker.total_cost_usd()
        result.passed       = self._pass_check(result)
        return result

    def evaluate_dataset(self, traces: list[AgentTrace]) -> dict[str, Any]:
        results = [self.evaluate_trace(t) for t in traces]
        return self._aggregate(results)

    def _pass_check(self, r: EvalResult) -> bool:
        checks = {
            "faithfulness":      r.faithfulness,
            "answer_relevance":  r.answer_relevance,
            "context_precision": r.context_precision,
            "tool_precision":    r.tool_precision,
            "tool_recall":       r.tool_recall,
            "trajectory_score":  r.trajectory_score,
        }
        return all(v >= self.thresholds.get(k, 0.0) for k, v in checks.items())

    @staticmethod
    def _aggregate(results: list[EvalResult]) -> dict[str, Any]:
        if not results:
            return {}

        numeric_keys = [
            "faithfulness", "answer_relevance", "context_precision",
            "tool_precision", "tool_recall", "tool_f1",
            "trajectory_score", "judge_score",
            "latency_s", "total_tokens", "cost_usd",
        ]
        agg: dict[str, Any] = {}
        for k in numeric_keys:
            vals = [getattr(r, k) for r in results]
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }
        agg["pass_rate"] = sum(r.passed for r in results) / len(results)
        agg["n_traces"]  = len(results)
        return agg


# ─────────────────────────────────────────────────────────────────────────────
# 10.  REGRESSION HARNESS
# ─────────────────────────────────────────────────────────────────────────────

class RegressionHarness:

    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance

    def save_baseline(self, agg: dict[str, Any], path: str) -> None:
        with open(path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[RegressionHarness] Baseline saved → {path}")

    def load_baseline(self, path: str) -> dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    def compare(self, baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        regressions:  list[str]       = []
        improvements: list[str]       = []
        details:      dict[str, dict] = {}

        numeric_keys = [
            "faithfulness", "answer_relevance", "context_precision",
            "tool_precision", "tool_recall", "tool_f1",
            "trajectory_score", "judge_score",
        ]
        for k in numeric_keys:
            if k not in baseline or k not in current:
                continue
            b_mean = baseline[k]["mean"]
            c_mean = current[k]["mean"]
            delta  = c_mean - b_mean
            details[k] = {"baseline": b_mean, "current": c_mean, "delta": delta}
            if delta < -self.tolerance:
                regressions.append(k)
            elif delta > self.tolerance:
                improvements.append(k)

        report = {
            "passed":       len(regressions) == 0,
            "regressions":  regressions,
            "improvements": improvements,
            "details":      details,
        }
        self._print_report(report)
        return report

    @staticmethod
    def _print_report(report: dict) -> None:
        status = "✅ PASSED" if report["passed"] else " FAILED"
        print(f"\n{'='*60}")
        print(f" Regression Report  {status}")
        print(f"{'='*60}")
        for metric, vals in report["details"].items():
            arrow = "↑" if vals["delta"] > 0 else ("↓" if vals["delta"] < 0 else "→")
            print(f"  {metric:<25} {vals['baseline']:.3f} → {vals['current']:.3f}  "
                  f"{arrow} {vals['delta']:+.3f}")
        if report["regressions"]:
            print(f"\n  Regressions:  {', '.join(report['regressions'])}")
        if report["improvements"]:
            print(f"  Improvements: {', '.join(report['improvements'])}")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_trace() -> AgentTrace:
    return AgentTrace(
        query        = "What is Retrieval-Augmented Generation and why is it useful?",
        ground_truth = (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
            "language model. It fetches relevant documents from a knowledge base and "
            "conditions the LLM's output on them, reducing hallucinations and keeping "
            "responses grounded in verifiable sources."
        ),
        final_answer = (
            "RAG stands for Retrieval-Augmented Generation. It enhances LLMs by first "
            "retrieving relevant documents from a knowledge base (via FAISS or BM25) "
            "and then generating answers conditioned on those documents. This reduces "
            "hallucinations and allows the model to cite sources."
        ),
        steps=[
            AgentStep(
                step_index   = 0,
                thought      = "I need to search the knowledge base for information about RAG.",
                action       = "retrieval_tool",
                action_input = {"query": "Retrieval-Augmented Generation"},
                observation  = "Found 3 relevant documents about RAG architecture.",
            ),
            AgentStep(
                step_index = 1,
                thought    = "I have enough context to answer. I will synthesise the information.",
                is_final   = True,
            ),
        ],
        retrieved_docs = [
            "RAG combines retrieval and generation to ground LLM outputs in external knowledge.",
            "Dense retrieval with FAISS enables fast approximate nearest-neighbour search.",
            "BM25 is a sparse retrieval method effective for keyword-heavy queries.",
        ],
        expected_tools = ["retrieval_tool"],
    )


def main():
    import os

    print("=" * 60)
    print(" 05 · Agent Evaluation Framework")
    print("=" * 60)

    api_key   = os.getenv("GROQ_API_KEY", "")
    client    = Groq(api_key=api_key) if api_key else None
    use_judge = bool(client)

    evaluator = AgentEvaluator(
        groq_client = client or Groq(api_key="dummy"),
        use_judge   = use_judge,
    )

    trace  = _build_mock_trace()
    print(f"\n[Query] {trace.query}\n")
    result = evaluator.evaluate_trace(trace)

    print("── Response Quality ─────────────────────────────────────")
    print(f"  Faithfulness      : {result.faithfulness:.3f}")
    if result.unsupported_claims:
        for c in result.unsupported_claims:
            print(f"    ✗ unsupported: {c}")
    print(f"  Answer Relevance  : {result.answer_relevance:.3f}")
    print(f"  Context Precision : {result.context_precision:.3f}")

    print("── Tool Use ─────────────────────────────────────────────")
    print(f"  Precision / Recall / F1 : "
          f"{result.tool_precision:.3f} / {result.tool_recall:.3f} / {result.tool_f1:.3f}")

    print("── Trajectory ───────────────────────────────────────────")
    print(f"  Score : {result.trajectory_score:.3f}")

    if use_judge:
        print("── LLM Judge ────────────────────────────────────────────")
        print(f"  Score    : {result.judge_score:.3f}  (normalised 0-1)")
        print(f"  Feedback : {result.judge_feedback}")

    print(f"\n  {'PASSED' if result.passed else ' FAILED (one or more thresholds missed)'}")
    print(f"  Latency      : {result.latency_s:.2f}s")
    print(f"  Total tokens : {result.total_tokens:,}")
    print(f"  Cost         : ${result.cost_usd:.6f}\n")

    evaluator.tracker.print_summary()

    # dataset aggregation
    agg = evaluator.evaluate_dataset([trace, trace])
    print("── Dataset Aggregation ──────────────────────────────────")
    for metric in ["faithfulness", "answer_relevance", "trajectory_score", "cost_usd"]:
        m = agg[metric]
        print(f"  {metric:<25} mean={m['mean']:.4f}  std={m['std']:.4f}")
    print(f"  pass_rate : {agg['pass_rate']:.0%}  ({agg['n_traces']} traces)\n")

    # regression demo
    harness  = RegressionHarness(tolerance=0.02)
    degraded = json.loads(json.dumps(agg))
    degraded["faithfulness"]["mean"] -= 0.05
    harness.compare(agg, degraded)


if __name__ == "__main__":
    main()