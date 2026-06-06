"""
02_tools.py
===========
Replace the mock rag_search with your real RAG pipeline.

In file 01, rag_search was a dictionary lookup — fake.
Here we import your actual pipeline from generation.py
and wrap it as an agent tool.

This is the key insight about agents:
  A tool is just a function that takes a string and returns a string.
  Your entire RAG pipeline — hybrid search, reranking, generation —
  becomes one tool the agent can call.

Design decision — why wrap RAG as a tool instead of calling it directly:
  The agent decides WHEN to call RAG.
  Sometimes the answer doesn't need retrieval at all (math, logic).
  The agent routes intelligently — RAG only fires when needed.
  This saves tokens, latency, and Groq API calls.
"""

import os
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Load your real RAG pipeline ──────────────────────────────────────────────
# Same importlib trick as 10_debugging_workflow.py
import importlib.util

def load_generation():
    gen_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "rag_pipeline",
        "generation.py"
    ))

    if not os.path.exists(gen_path):
        print(f"[warn] generation.py not found at {gen_path}")
        print("[warn] Using mock RAG instead")
        return None
    spec = importlib.util.spec_from_file_location("generation", gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen

print("[info] Loading RAG pipeline...")
gen = load_generation()

# ── Tool definitions ─────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def rag_search(query: str) -> str:
    """
    Search the RAG knowledge base.
    Uses real pipeline if available, mock if not.
    """
    if gen is not None:
        try:
            result = gen.rag(query)
            return result["answer"]
        except Exception as e:
            return f"RAG error: {e}"
    else:
        # fallback mock
        knowledge = {
            "attention":    "Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
            "transformer":  "Transformer uses self-attention to process tokens in parallel.",
            "distilbert":   "DistilBERT is a smaller, faster BERT with 66M parameters.",
            "lora":         "LoRA injects low-rank matrices into attention layers for efficient fine-tuning.",
            "bm25":         "BM25 is a sparse retrieval method based on TF-IDF term weighting with document length normalisation.",
            "faiss":        "FAISS performs approximate nearest neighbour search via IVF or HNSW indexing of dense vectors.",
            "dense":        "Dense retrieval uses bi-encoder models to embed queries and documents into shared vector spaces.",
            "hybrid":       "Hybrid search combines BM25 sparse retrieval with dense vector search, re-ranked by a cross-encoder.",
            "langgraph":    "LangGraph models agent workflows as typed state machines with conditional routing between nodes.",
            "rag":          "RAG combines retrieval and generation — retrieved context grounds the LLM output and reduces hallucination.",
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return value
        return "No relevant information found."

def get_current_date(_: str) -> str:
    """Return today's date — useful for time-sensitive queries."""
    from datetime import date
    return str(date.today())

TOOLS = {
    "calculator"       : calculator,
    "rag_search"       : rag_search,
    "get_current_date" : get_current_date,
}

TOOL_DESCRIPTIONS = """
You have access to these tools:

1. calculator(expression)
   - Use for any math calculation
   - Input: valid Python math expression e.g. "2 ** 10" or "850 * 0.15"

2. rag_search(query)
   - Use to look up information from the knowledge base
   - Input: a search query string
   - Returns: answer from the knowledge base

3. get_current_date(input)
   - Returns today's date
   - Input: anything (ignored)

To use a tool respond EXACTLY in this format:
THOUGHT: your reasoning
ACTION: tool_name
INPUT: tool input

When you have the final answer:
THOUGHT: I now have enough information
ANSWER: your final answer
"""

# ── ReAct loop (same as file 01) ─────────────────────────────────────────────

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def run_agent(query: str, max_steps: int = 5) -> str:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": TOOL_DESCRIPTIONS},
        {"role": "user", "content": query}
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()
        print(content)

        if "ANSWER:" in content:
            answer = content.split("ANSWER:")[-1].strip()
            print(f"\n{'='*60}")
            print(f"Final Answer: {answer}")
            print(f"{'='*60}")
            return answer

        if "ACTION:" in content and "INPUT:" in content:
            action = content.split("ACTION:")[-1].split("\n")[0].strip()
            input_ = content.split("INPUT:")[-1].split("\n")[0].strip()
            print(f"\n[tool call] {action}({input_})")
            observation = TOOLS.get(action, lambda x: f"Unknown tool: {action}")(input_)
            print(f"[observation] {observation}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
        else:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Please use THOUGHT/ACTION/INPUT format."})

    return "Agent reached max steps without an answer."

# ── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "What is 25% of 1200?",
        "How does scaled dot-product attention work?",
        "What is today's date?",
        "What is 2 to the power of 8, and how does LoRA work?",
    ]
    for q in queries:
        run_agent(q)
        print()