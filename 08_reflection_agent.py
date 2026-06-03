"""
08_reflection_agent.py — Reflection Agent
==========================================
Agent critiques its own output through a structured self-improvement loop.
 
Key design decisions:
* ReflectionCritic: separate LLM call that scores output on 4 axes
  (accuracy, completeness, clarity, relevance) — scores are 0-10 ints
* ReflectionAgent.run() loops until reflection_score >= threshold OR max_reflections hit
* reflection_history: list[ReflectionRound] — full audit trail for eval
* ImprovementDirective: structured object derived from critique, passed back
  into next generation as explicit instructions (not raw critique text)
* plan_adherence_score imported from 07 — reflection improves adherence too
* has_failure property carried forward for compatibility with 09+
* Groq-only (llama-3.3-70b-versatile), sentence-transformers for RAG
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq

#Optional 
try:
    from tools  import  rag_search
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CritiqueScore:
    accuracy: int       # 0-10: factual correctness
    completeness: int   # 0-10: covers all aspects of the question
    clarity: int        # 0-10: well-structured, readable
    relevance: int      # 0-10: stays on topic
    reasoning: str      # brief justification for scores


    @property
    def overall(self) ->float:
        return(self.accuracy + self.completeness + self.clarity + self.relevance) / 4.0


    def __str__(self) -> str:
        return(
            f"accuracy={self.accuracy}/10 completeness = {self.completeness}/10"
            f"clarity = {self.clarity}/10 relevance ={self.relevance}/10"
            f" overall = {self.overall:.1f}/10"
        )
    
@dataclass
class ImprovementDirective:
    """Structured instructions fed back into the next generation pass."""
    fix_accuracy : list[str] = field(default_factory=list)
    add_missing : list[str] = field(default_factory=list)
    restructure : list[str] = field(default_factory=list)
    remove_tangents : list[str] = field(default_factory= list)

    def as_prompt_block(self) -> str:
        lines = ["IMPROVEMENT DIRECTIVES (apply all of these):"]
        for item in self.fix_accuracy:
            lines.append(f"  • FIX ACCURACY: {item}")
        for item in self.add_missing:
            lines.append(f"  • ADD MISSING: {item}")
        for item in self.restructure:
            lines.append(f"  • RESTRUCTURE: {item}")
        for item in self.remove_tangents:
            lines.append(f"  • REMOVE: {item}")
        return "\n".join(lines) if len(lines) > 1 else ""

 
@dataclass
class ReflectionRound:
    round_num : int
    draft : str
    critique : CritiqueScore
    directive : ImprovementDirective
    latency_ms : int

@dataclass
class ReflectionResult:
    question : str
    final_answer : str
    rounds : list[ReflectionRound]
    converged : bool         # True if threshold met, False if max_reflections hit
    total_latency_ms : int

    @property
    def has_failure(self) -> bool:
        """Compatibility with 09+ parallel execution harness."""
        return not self.converged and len(self.rounds) ==0
    
    @property
    def reflection_score(self) -> float:
        return self.rounds[-1].critique.overall if self.rounds else 0.0
    
    def summary(self) -> str:
        lines = [
            f"\n{'═'*60}",
            f"REFLECTION RESULT",
            f"{'═'*60}",
            f"Question : {self.question}",
            f"Rounds   : {len(self.rounds)}",
            f"Converged : {self.converged}",
            f"Final_score : {self.reflection_score:.1f}/10",
            f"Latency     : {self.total_latency_ms}ms",
            f"{'-'*60}"
        ]
        for r in self.rounds:
            lines.append(
                f"[Round {r.round_num}] score={r.critique.overall:.1f}  "
                f"({r.latency_ms}ms)"
            )
            lines.append(f"  Draft: {r.draft[:120]}{'...' if len(r.draft)>120 else ''}")
            lines.append(f"  Critique: {r.critique}")
        lines += [f"{'─'*60}", f"FINAL ANSWER:\n{self.final_answer}", f"{'═'*60}\n"]
        return "\n".join(lines)
    
# ─────────────────────────────────────────────────────────────────────────────
# ReflectionCritic
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """\
You are rigorous AI output evaluator. Given a question and a draft answer,
score the answer on four anxes (0-10 integers each) and extract specific improvement
directives.

Respond ONLY with valid JSON - no markdown, no preamble.
{
"accuracy : <int 0-10>,
"completeness" : <int 0-10>,
"clarity" : <int 0-10>,
"relevance" : <int 0-10>,
"reasoning" : <1-2 sentence justification>,
"fix_accuracy" : ["<specific factual error to fix>", ...]
"add_missing" : ["<topic or detal that was omitted>"...]
"restructure" : ["<structural improvement>",...]
"remove_tangents" : ["<irrelevant content to cut>",...]
}
Be strict. A score of 10 means publication ready with zero flaws.
Empty list are fine if there's nothing to fix in that category."""

class ReflectionCritic:
   
    def __init__(self, client:Groq,model :str):
        self.client = client
        self.model = model

    def critique(self, question:str, draft: str) -> tuple [CritiqueScore, ImprovementDirective]:

        prompt = f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft}"
        resp = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role" :"user", "content" : prompt}
            ],
            temperature= 0.2,
            max_tokens= 600,
        )
        raw = resp.choices[0].message.content.strip()

        # strip possible markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # graceful degradation — assume mediocre scores
            data = {
                "accuracy" :5, "completeness" : 5 , "clarity" : 5,
                "relevance" :5, "reasoning" : "Cloud not parse critique",
                "fix_accuracy" : [], "add_missing" : [], "restructure" : [],
                "remove_tangents" : []
            }
        score = CritiqueScore(

            accuracy= int(data.get("accuracy" ,5)),
            completeness=int(data.get("completeness" ,5)),
            clarity=int(data.get("clarity", 5)),
            relevance=int(data.get("relevance", 5)),
            reasoning=data.get("reasoning",""),
        )
        directive = ImprovementDirective(
            fix_accuracy=data.get("fix_accuracy", []),
            add_missing=data.get("add_missing",[]),
            restructure=data.get("restructure", []),
            remove_tangents=data.get("remove_tangents", []),
        )
        return score, directive
    
# ─────────────────────────────────────────────────────────────────────────────
# Generator (drafts answers + improves them)
# ─────────────────────────────────────────────────────────────────────────────

GENERATOR_SYSTEM = """\
You are a precise, knowledgeable AI assistant. Answer the user's question
thoroughly and accurately. When improvement directives are provided, 
apply every one of them without exception."""

class AnswerGenerator:
    def __init__(self, client : Groq, model: str):
        self.client = client
        self.model = model
    
    def generate(
            self, 
            question :str, 
            directive : Optional[ImprovementDirective] = None,
            previous_draft : Optional[str] = None,
            ) -> str :

        user_parts = [f"QUESTION:\n{question}"]
        if previous_draft:
            user_parts.append(f"\nPREVIOUS DRAFT (improve this):\n{previous_draft}")
        if directive:
            block = directive.as_prompt_block
            if block:
                user_parts.append(f"\n{block}")


        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role" : "system" , "content" : GENERATOR_SYSTEM},
                {"role" : "user" , "content" : "\n".join(user_parts)}
            ],
            temperature=0.5,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    
# ─────────────────────────────────────────────────────────────────────────────
# ReflectionAgent — orchestrator
# ─────────────────────────────────────────────────────────────────────────────
 
class ReflectionAgent:
    def __init__(
            self,
            model : str = "llama-3.3-70b-versatile",
            threshold : float = 7.5,
            max_reflections : int = 3,
    ):
        api_key = os.environ.get("GROQ_API_KEY","")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.threshold = threshold
        self.max_reflections = max_reflections

        self.generator = AnswerGenerator(self.client,model)
        self.critic = ReflectionCritic(self.client, model)

    
    def run(self, question : str) -> ReflectionResult:
        t_start = time.time()
        rounds : list[ReflectionRound] = []
        converged = False
        draft : Optional[str] = None
        directive : Optional[ImprovementDirective] = None

        for round_num in range (1, self.max_reflections +1):
            t0 = time.time()
            #Generate / improve
            draft = self.generator.generate(
                question=question,
                directive=directive,
                previous_draft=draft,
            )
            # Critique
            score, directive = self.critic.critique(question, draft)
            latency = int((time.time() - t0) *1000)

            rounds.append(ReflectionRound(
                round_num=  round_num,
                draft=draft,
                critique=score,
                directive=directive,
                latency_ms=latency,
            ))
            print(f"  [Round {round_num}] overall={score.overall:.1f}/10  "
                  f"({latency}ms)  — {score.reasoning[:80]}")
            if score.overall >= self.threshold:
                converged = True
                break
        
        total_ms = int((time.time() - t_start)*1000)
        return ReflectionResult(
            question = question,
            final_answer= draft or "",
            rounds = rounds,
            converged=converged,
            total_latency_ms=total_ms,
        )
    
# ─────────────────────────────────────────────────────────────────────────────
# plan_adherence_score (imported from 07 or standalone fallback)
# ─────────────────────────────────────────────────────────────────────────────

def reflection_improvement_score(
        reflection_rounds : list[ReflectionRound],
        ) ->float:
        """
    Measures how much the final answer improved per reflection round relative
    to the original plan's expected coverage.
 
    Low adherence + high faithfulness = reflection agent diverged productively.
    High adherence + high faithfulness = agent stayed on plan and nailed it.
    """
        if not reflection_rounds:
            return 0.0
        first = reflection_rounds[0].critique.overall
        last = reflection_rounds[-1].critique.overall
        improvement = max(0.0, last- first)
        # normalise: full 10-point improvement = 1.0
        return round(min(1.0, improvement/10.0),3)

# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Explain how FAISS performs approximate nearest-neighbor search and when to prefer IVF over HNSW.",
    "What are the key trade-offs between BM25 and dense retrieval in a hybrid RAG pipeline?",
    "How does LangGraph's state machine differ from a simple ReAct loop for multi-agent orchestration?",
]


def run_demo() -> None:
    print("\n" + "═" * 60)
    print("08 — REFLECTION AGENT DEMO")
    print("Agent critiques and iteratively improves its own output")
    print("═" * 60 + "\n")

    agent = ReflectionAgent(threshold=9.5, max_reflections=3) # play with threshold 7.5 /8 /8.5

    for q in DEMO_QUESTIONS:
        print(f"\nQuestion: {q}")
        result = agent.run(q)
        print(result.summary())

        improvement_score = reflection_improvement_score(
            reflection_rounds=result.rounds,
        )
        print(f"  reflection_improvement_score : {improvement_score:.3f}")
        print(f"  has_failure          : {result.has_failure}")
        print()


if __name__ == "__main__":
    run_demo()




        




