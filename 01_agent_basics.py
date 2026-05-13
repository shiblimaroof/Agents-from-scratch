"""
01_agent_basics.py
==================
What is an agent and how does the ReAct pattern work?

RAG is linear:
  query → retrieve → generate → done

An agent is a loop:
  query → think → act → observe → think → act → observe → answer

ReAct = Reasoning + Acting
  The model reasons about what to do next,
  takes an action (calls a tool),
  observes the result,
  reasons again.

This file builds a minimal ReAct agent from scratch.
No LangChain. No LangGraph. Just Groq + a loop.

The agent has two tools:
  - calculator : evaluates math expressions
  - rag        : searches your knowledge base (mocked here)

We'll connect your real RAG pipeline in file 02.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ── Tools ────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def rag_search(query: str) -> str:
    """Mock RAG search — replaced with real pipeline in file 02."""
    knowledge = {
        "attention": "Scaled dot-product attention computes Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
        "transformer": "The transformer uses self-attention to process tokens in parallel.",
        "distilbert": "DistilBERT is a smaller, faster version of BERT with 66M parameters.",
        "lora": "LoRA fine-tunes large models by injecting low-rank matrices into attention layers.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "No relevant information found in knowledge base."

TOOLS = {
    "calculator": calculator,
    "rag_search": rag_search,
}

# ── Tool definitions for the LLM ─────────────────────────────────────────────

TOOL_DESCRIPTIONS = """
You have access to these tools:

1. calculator(expression)
   - Use for any math calculation
   - Input: a valid Python math expression like "2 + 2" or "100 * 0.15"
   - Returns: the result as a string

2. rag_search(query)
   - Use to look up information from the knowledge base
   - Input: a search query string
   - Returns: relevant information or "No relevant information found"

To use a tool, respond EXACTLY in this format:
THOUGHT: your reasoning about what to do next
ACTION: tool_name
INPUT: tool input

When you have the final answer:
THOUGHT: I now have enough information
ANSWER: your final answer here
"""

# ── ReAct loop ───────────────────────────────────────────────────────────────

def run_agent(query: str, max_steps: int = 5) -> str:
    """
    ReAct agent loop.

    max_steps: safety limit — agents can loop forever without it.
    This is one of the most common agent failure modes.
    File 06 covers this in detail.
    """
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

        # Parse ANSWER
        if "ANSWER:" in content:
            answer = content.split("ANSWER:")[-1].strip()
            print(f"\n{'='*60}")
            print(f"Final Answer: {answer}")
            print(f"{'='*60}")
            return answer

        # Parse ACTION + INPUT
        if "ACTION:" in content and "INPUT:" in content:
            action = content.split("ACTION:")[-1].split("\n")[0].strip()
            input_ = content.split("INPUT:")[-1].split("\n")[0].strip()

            print(f"\n[tool call] {action}({input_})")

            if action in TOOLS:
                observation = TOOLS[action](input_)
            else:
                observation = f"Unknown tool: {action}"

            print(f"[observation] {observation}")

            # Add to message history so model sees what happened
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        else:
            # Model didn't follow format — happens sometimes
            print("[warn] Model didn't follow ReAct format. Retrying.")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Please use the THOUGHT/ACTION/INPUT format."})

    return "Agent reached max steps without an answer."

# ── Test queries ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "What is 15% of 850?",
        "How does attention work in transformers?",
        "What is 2 to the power of 10, and what is DistilBERT?",
    ]

    for q in queries:
        run_agent(q)
        print()

