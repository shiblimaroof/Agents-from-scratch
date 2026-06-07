"""
12_ragas_langsmith.py — RAGAS Evaluation + LangSmith Feedback Logging
======================================================================
Evaluates agents from 09, 10, and 11 using RAGAS metrics, then pushes
scores to LangSmith as structured run feedback.

Pipeline per sample:
  1. Run target agent → capture answer + retrieved contexts
  2. Compute RAGAS metrics (faithfulness, answer_relevancy, context_precision)
  3. Push scores to LangSmith via create_feedback()

Key design decisions:
* RAGAS 0.4.x API — EvaluationDataset + SingleTurnSample + evaluate()
* LLM for RAGAS scoring: ChatGroq (llama-3.3-70b-versatile) — no OpenAI needed
* Embeddings for answer_relevancy: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
* Contexts captured at agent boundary — not re-retrieved after the fact
* LangSmith run_id generated per sample — scores pushed as create_feedback()
* Aggregate scores per agent printed + logged — mean across eval samples
* Hardcoded EVAL_DATASET — portable, readable, no external file dependency
* Ground truth matches toy corpus content — realistic but reproducible
* Each agent gets 3 eval samples — enough to show variance, fast to run
* Unique thread_id per agent-10 sample — no memory bleed between eval runs
"""

from __future__ import annotations

import os
import sys
import uuid
import time
import importlib.util
from dataclasses import dataclass, field
from typing import Optional, Annotated

from dotenv import load_dotenv
from langsmith import Client as LangsmithClient

load_dotenv()

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
try:
    from langchain.community.chat_models.vertexai import ChatVertexAI
except ImportError:
    ChatVertexAI = None

# ── RAGAS LLM + embeddings (Groq — no OpenAI needed) ─────────────────────────

_groq_llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = os.environ.get("GROQ_API_KEY", ''),
    temperature= 0.0,
)

_embedddings = HuggingFaceBgeEmbeddings(model_name = "all-MiniLM-L6-v2")

RAGAS_LLM = LangchainLLMWrapper(_groq_llm)
RAGAS_EMBEDDINGS = LangchainEmbeddingsWrapper(_embedddings)

METRICS = [
    Faithfulness(llm=RAGAS_LLM),
    AnswerRelevancy(llm=RAGAS_LLM, embeddings=RAGAS_EMBEDDINGS),
    ContextPrecision(llm=RAGAS_LLM),
]

METRIC_NAMES = ["faithfulness" , "answer_relevancy" , "context_precision"]

# ── LangSmith client ──────────────────────────────────────────────────────────

langsmith_client = LangsmithClient(
    api_key = os.environ.get("LANGCHAIN_API_KEY",'')
)
LANGSMITH_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "agents-from-scratch")

# ─────────────────────────────────────────────────────────────────────────────
# Eval dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    agent: str                                           # "09" | "10" | "11"
    contexts: list[str] = field(default_factory=list)   # filled at runtime
    answer: str = ""                                     # filled at runtime
    run_id: Optional[str] = None                         # LangSmith run ID

EVAL_DATASET : list[EvalSample] = [
    # ── 09: multi-agent Researcher + Writer ──────────────────────────────────
    EvalSample(
        agent = "09",
        question = "What is BM25 and how does it score documents.?",
        ground_truth = (
            "BM25 is a sparse retrieval method based on TF-IDF term weighting "
            "with document length normalisation. It scores documents using term "
            "frequency, inverse document frequency, and a length penalty "
            "controlled by hyperparameters k1 and b."
        ),
    ),
    EvalSample(
        agent = "09",
        question = "What is FAISS used for.?",
        ground_truth = (
            "FAISS is a library for efficient approximate nearest neighbour "
            "search over dense vectors. It supports IVF and HNSW indexing "
            "strategies and scales to billions of vectors."
        ),
    ),
    EvalSample(
        agent = "09",
        question = "How does hybrid search combine BM25 and dense retrieval?",
        ground_truth = (
            "Hybrid search combines BM25 sparse retrieval with dense vector "
            "search, then fuses results using Reciprocal Rank Fusion (RRF). "
            "A cross-encoder reranker scores the merged candidate set to "
            "produce the final ranking."
        ),
    ),
    # ── 10: persistent memory ─────────────────────────────────────────────────
    EvalSample(
        agent = "10",
        question = "What is dense retrieval and how does it differ from BM25.?",
        ground_truth = (
            "Dense retrieval uses bi-encoder models to embed queries and "
            "documents into shared vector spaces. Unlike BM25, it captures "
            "semantic similarity rather than relying on exact keyword matching."
        ),
        
    ),
    EvalSample(
        agent = "10",
        question = "What indexing strategies does FAISS support.?",
        ground_truth = (
            "FAISS supports IVF and HNSW indexing strategies. IVF partitions "
            "the vector space into clusters; HNSW builds a hierarchical graph "
            "for logarithmic-time search."
        ),
    ),
    EvalSample(
        agent="10",
        question = "What is RAG why does it reduce hallucination.?",
        ground_truth = (
            "RAG combines retrieval and generation. Retrieved context grounds "
            "the LLM output, reducing hallucination by anchoring the model's "
            "response to retrieved evidence rather than parametric memory."
        ),
    ),
    # ── 11: structured tool calling ───────────────────────────────────────────
    EvalSample(
        agent="11",
        question = "What is BM25?",
        ground_truth=(
            "BM25 is a sparse retrieval method based on TF-IDF term weighting "
            "with document length normalisation controlled by hyperparameters "
            "k1 and b."
        ),
    ),
    EvalSample(
        agent = "11",
        question = "What is FAISS and what indexing methods does it use.?",
        ground_truth=(
            "FAISS is a library for efficient approximate nearest neighbour "
            "search over dense vectors, supporting IVF and HNSW indexing."
        ),
    ),
    EvalSample(
        agent = "11",
        question = "How does hybrid search work.?",
        ground_truth=(
            "Hybrid search combines BM25 sparse retrieval with dense vector "
            "search and uses Reciprocal Rank Fusion to merge results, followed "
            "by cross-encoder reranking."
        ),
    ),

]

# ─────────────────────────────────────────────────────────────────────────────
# Agent loader
# importlib loads numbered filenames (invalid Python identifiers).
# sys.modules registration before exec_module fixes Annotated resolution
# under importlib — without it, TypedDict fields fail in Python 3.11.
# ─────────────────────────────────────────────────────────────────────────────
def _load_agent(filename : str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    name = filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ─────────────────────────────────────────────────────────────────────────────
# Agent runners — return (answer, contexts, run_id)
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_09(question : str) -> tuple[str, list[str], str]:
    """Run 09_multi_agent.py and capture answer + research context."""
    mod = _load_agent("09_multi_agent.py")
    run_id = str(uuid.uuid4())
    result = mod.run(question)
    answer = result.get("answer","")
    research = result.get("research","")
    contexts = [research] if research else ["No context retrieved."]
    return answer, contexts, run_id

def run_agent_10(question:str) -> tuple[str, list[str], str]:
    """Run 10_agent_memory_long.py with a fresh thread per eval sample."""
    mod = _load_agent("10_agent_memory_long.py")
    run_id = str(uuid.uuid4())
    thread_id = f"eval_{run_id[:8]}"
    checkpointer = mod.setup_checkpointer()
    app = mod.build_graph(checkpointer)
    result = mod.run(app, question, thread_id)
    answer = result.get("answer","")
    research = result.get("research","")
    contexts = [research] if research else ["No context retrieved"]
    return answer, contexts, run_id

def run_agent_11(question:str)-> tuple[str, list[str], str]:
    """Run 11_tool_calling.py and capture answer + tool-retrieved contexts."""
    from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
    mod = _load_agent("11_tool_calling.py")
    run_id = str(uuid.uuid4())
    initial_messages = [
        SystemMessage(content=mod.SYSTEM_PROMPT),
        HumanMessage(content=question),        
    ]
    result = mod.app.invoke({"messages": initial_messages})
    answer = result["messages"][-1].content
    contexts = [
        msg.content for msg in result["messages"]
        if isinstance(msg, ToolMessage) and msg.content
    ] or ["No content retrieved"]
    return answer, contexts,run_id

AGENT_RUNNERS = {
    "09": run_agent_09,
    "10": run_agent_10,
    "11": run_agent_11,
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ragas(samples : list[EvalSample]) ->list[dict]:
    """
    RAGAS 0.2.x API:
      - SingleTurnSample(user_input, response, retrieved_contexts, reference)
      - EvaluationDataset(samples=[...])
      - evaluate(dataset, metrics=[...])
    """
    ragas_samples = [
        SingleTurnSample(
            user_input = s.question,
            response = s.answer,
            retrieved_contexts = s.contexts,
            reference = s.ground_truth

        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples = ragas_samples)

    print(f"\n  [RAGAS] evaluating {len(samples)} samples...")

    results = evaluate(dataset=dataset, metrics=METRICS)
    df = results.to_pandas()

    scores = df[METRIC_NAMES].fillna(0.0).astype(float).to_dict(orient="records")
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# LangSmith feedback logging
# ─────────────────────────────────────────────────────────────────────────────
 
def push_to_langsmith(sample : EvalSample, scores : dict) ->None:
    """Push RAGAS scores to LangSmith as run feedback."""
    try:
        run_id = sample.run_id or str(uuid.uuid4())
        langsmith_client.create_run(
            id = run_id,
            name = f"eval_agent_{sample.agent}",
            run_type = "chain",
            project_name=LANGSMITH_PROJECT,
            inputs = {"question" : sample.question},
            output = {"answer":sample.answer},
        )
        time.sleep(0.03)
        for metric_name, score in scores.items():
            langsmith_client.create_feedback(
                run_id=run_id,
                key = metric_name,
                score= score,
                comment = f"RAGAS {metric_name} - agent {sample.agent}",
            )
        print(f"  [LangSmith] pushed {len(scores)} scores → run {run_id[:8]}...")
    except Exception as e:
        print(f"  [LangSmith] failed: {e}")

 
# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ─────────────────────────────────────────────────────────────────────────────
 
def run_eval()-> None:

    print("\n" + "═" * 60)
    print("12 — RAGAS EVAL + LANGSMITH FEEDBACK")
    print("Agents: 09 (multi-agent) | 10 (memory) | 11 (tool calling)")
    print("═" * 60)
    
    # Step 1: Run all agents
    print("\n[Step 1] Running agents...")
    for sample in EVAL_DATASET:
        print(f"\n Agent {sample.agent} | Q: {sample.question[:55]}...")
        try:
            runner = AGENT_RUNNERS[sample.agent]
            answer, contexts, run_id = runner(sample.question)
            sample.answer = answer
            sample.contexts = contexts
            sample.run_id = run_id
            print(f" {answer[:80]}")
        except Exception as e:
            print(f" FAILED : {e}")
            sample.answer = f"Agent failed : {e}"
            sample.contexts = ["No context retrieved"]
            sample.run_id = str(uuid.uuid4())

    # Step 2: RAGAS metrics per agent
    print("\n[Step 2] Computing RAGAS metrics...")
    for agent_id in ["09", "10","11"]:
        agent_samples = [s for s in EVAL_DATASET if s.agent == agent_id]
        valid = [s for s in agent_samples if not s.answer.startswith("Agent failed")]

        if not valid:
            print(f"  Agent {agent_id}: all samples failed — skipping")
            continue
        try:
            scores = compute_ragas(valid)
            for sample,score in zip(valid, scores):
                sample._scores = score
        except Exception as e:
            print(f"  Agent {agent_id} RAGAS failed: {e}")
        
    # Step 3: Push to LangSmith
    print("\n[Step 3] Pushing to LangSmith...")
    for sample in EVAL_DATASET:
        scores = getattr(sample, "_scores", {})
        if scores:
            push_to_langsmith(sample,scores)

    # Step 4: Print results table
    print("\n" + "═" * 60)
    print("RESULTS")
    print("═" * 60)  

    for agent_id in ["09", "10", "11"]:
        agent_samples = [s for s in EVAL_DATASET if s.agent ==agent_id]
        scores_list = [getattr(s, "_scores", {}) for s in agent_samples if hasattr(s,"_scores")]

        print(f"\nAgent {agent_id}:")
        print(f"  {'Question':<45} {'Faith':>6} {'Relev':>6} {'Prec':>6}")
        print(f"  {'─'*45} {'─'*6} {'─'*6} {'─'*6}")

        for sample in agent_samples:
            sc =getattr(sample, "_scores",{})
            if sc:
                print(
                        f"  {sample.question[:45]:<45} "
                        f"{sc.get('faithfulness', 0):.3f} "
                        f"{sc.get('answer_relevancy', 0):.3f} "
                        f"{sc.get('context_precision', 0):.3f}"
                    )
        if scores_list:
            means = {
                    m: sum(s.get(m, 0) for s in scores_list) / len(scores_list)
                    for m in METRIC_NAMES
                }
            print(
                    f"  {'MEAN':<45} "
                    f"{means['faithfulness']:.3f} "
                    f"{means['answer_relevancy']:.3f} "
                    f"{means['context_precision']:.3f}"
                )
 
    print(f"\n{'═'*60}")
    print("LangSmith: https://smith.langchain.com")
    print(f"Project:   {LANGSMITH_PROJECT}")
    print(f"{'═'*60}\n")
 
 
if __name__ == "__main__":
    run_eval()

