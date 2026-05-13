import os
from groq import Groq
from dotenv import load_dotenv
import importlib.util

load_dotenv()
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
MODEL = "llama-3.3-70b-versatile"

# ── Load your real RAG pipeline ───────────────────────────────────────────────────────────────────────────────

def load_generation():
    gen_path = os.path.join(
        os.path.dirname(os.getcwd()),
        "rag_pipeline",
        "generation.py"
    )
    if not os.path.exists(gen_path):
        print(f"[warn] generation.py not found at {gen_path}")
        print("[warn] Using mock RAG instead")
        return None
    spec = importlib.util.spec_from_file_location("generation",gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen

print('[info] Loading RAG pipelimne')
gen = load_generation()

# ── Tools ────────────────────────────────────────────────────────────────────────────────────────

def calculator(expression :str) -> str:
    """Evaluate a math expression safely."""
    try:
        result = eval(expression, {"__builtins__":{}})
        return str(result)
    except Exception as e:
        return f"Error : {e}"    

def rag_search(query:str) -> str:
    """
    Search the RAG knowledge base.
    Uses real pipeline if available, mock if not.
    """
    if gen is not None:
        try:
            result = gen.rag(query)
            return result["answer"]
        except Exception as e:
            return f"RAG error :{e}"
    else:
        #fallback mock
        knowledge = {
            "attention": "Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
            "transformer": "Transformer uses self-attention to process tokens in parallel.",
            "distilbert": "DistilBERT is a smaller, faster BERT with 66M parameters.",
            "lora": "LoRA injects low-rank matrices into attention layers for efficient fine-tuning.",
        }    
        for key,value in knowledge.items():
            if key in query.lower():
                return value
        return "No relevant information found."
    
def get_current_date(_:str) ->str:
    """Return today's date — useful for time-sensitive queries."""
    from datetime import date
    return str(date.today())

TOOLS = {
    "calculator" : calculator,
    "rag_search" : rag_search,
    "get_current_date" : get_current_date
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

# ── Agent with memory ────────────────────────────────────────────────────────

class AgentWithMemory:
    def __init__(self):
        self.messages = [
            {"role" : "system", "content" : TOOL_DESCRIPTIONS}
        ]

    def get_content_window(self, max_messages:int =10) -> list:
        # always keep system message (index 0)
        # keep last max_messages from the rest
        system = self.messages[:1]
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages[1:]
        return system + recent

    def chat(self, user_input :str, max_steps :int =5) -> str:
        self.messages.append({"role":"user", "content" : user_input})

        for step in range(max_steps):

            response = client.chat.completions.create(
                model = MODEL,
                messages= self.get_content_window(),
                temperature= 0.0,
                max_tokens = 512,
            )
            content = response.choices[0].message.content.strip()
            print(content)

            if "ANSWER" in content:
                answer = content.split("ANSWER:")[-1].strip()
                self.messages.append({"role" : "assistant", "content" : content})
                return answer
            
            if "ACTION:" in content and "INPUT:" in content:
                action = content.split('ACTION:')[-1].split('\n')[0].strip()
                input_ = content.split('INPUT:')[-1].split('\n')[0].strip() 
                print(f"\n[tool_call] {action} ({input_})")
                observation= TOOLS.get(action, lambda x: f"Unknown tool : {action}")(input_)
                print(f"[obseervation]{observation}")
                self.messages.append({"role" : "assistant", "content" : content})
                self.messages.append({"role" : "user", "content" : f"OBSERVATION : {observation}"})
            else:
                print("[warn] Model didn't follow ReAct format. Retrying.")
                self.messages.append({"role" : "assistant", "content" : content})
                self.messages.append({"role" : "user", "content": "Please use THOUGHT/ACTION/INPUT format."})

        return "Agent reached max steps without an answer"
    
    def reset(self):
        self.messages = [
            {"role" : "system", "content" : TOOL_DESCRIPTIONS}
        ]
        print("[INFO] Memory cleared ")


if __name__ == "__main__":
    agent = AgentWithMemory()

    questions = [
        "What is today's date?",
        "How does attention work in transformers?",
        "What is 15% of 850?",
        "Add 200 to that and tell me what LoRA is.",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        answer = agent.chat(q)
        print(f"Agent: {answer}")
                                       
