# Agents From Scratch

AI Agent built from first principles.
No LangChain. No LangGraph. Just Python, Groq, and a loop.

---

## What this is

Most agent tutorials hide the complexity inside frameworks.
This repo builds the agent layer by layer — every design decision explained inline.

The core idea:

```
query → THOUGHT → ACTION → OBSERVATION → THOUGHT → ... → ANSWER
```

That loop, plus tools, plus memory, is all an agent is.

---

## Files

| File | What it does |
|------|-------------|
| `01_agent_basics.py` | ReAct loop from scratch — think, act, observe |
| `02_tools.py` | Real RAG pipeline wrapped as a tool |
| `03_memory.py` | Conversation memory across turns |
| `04_multi_tool_agent.py` | Sliding window context + multi-tool routing |

---

## Key Concepts

**ReAct pattern — Reasoning + Acting**
The model doesn't answer immediately. It thinks about what tool to use, calls it, observes the result, then thinks again. This loop continues until it's confident enough to answer.

**Why tools matter**
A tool is just a function that takes a string and returns a string. Your entire RAG pipeline — hybrid search, reranking, generation — becomes one tool the agent can call. The agent decides when to use it.

**Conversation memory**
`self.messages` persists across turns. The entire conversation history is sent with every API call. That's why "add 200 to that" works — the model sees what "that" referred to two turns ago.

**Sliding window**
Memory grows forever without a limit. After N turns, the context window overflows. Sliding window keeps the system message + last N messages — old turns get dropped, recent context stays.

**Why max_steps exists**
Without a step limit, agents can loop forever — calling tools repeatedly without converging on an answer. `max_steps=5` is a safety gate. One of the most common agent failure modes in production.

---

## Real output

```
User: What is 15% of 850?
Agent: 127.5

User: Add 200 to that result.
Agent: 327.5          ← knew "that" = 127.5 from memory

User: Now multiply the total by 3.
Agent: 982.5          ← knew "total" = 327.5 from memory
```

---

## Stack

Python · Groq API · FAISS · BM25 · Sentence Transformers

RAG pipeline: [rag-from-scratch](https://github.com/shiblimaroof/rag-from-scratch)

---

## Setup

```bash
git clone https://github.com/shiblimaroof/Agents-from-scratch.git
cd Agents-from-scratch
pip install groq sentence-transformers faiss-cpu rank-bm25 python-dotenv
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Run any file:

```bash
python 01_agent_basics.py
python 04_multi_tool_agent.py
```

---

## Author

Shibli |
[@BuildwithShibli](https://twitter.com/BuildwithShibli) · [github.com/shiblimaroof](https://github.com/shiblimaroof)
