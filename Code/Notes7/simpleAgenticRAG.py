import re
import fitz
import pickle
import faiss
import json
import numpy as np
from openai import OpenAI
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where embeddings and index are stored (same as in createIndex script)
SAVE_DIR = os.path.join(script_dir, "book_index")
os.makedirs(SAVE_DIR, exist_ok=True)

client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"  # prototyping, might want to switch to a larger model if
                                        # wanted to put into production

# Load the saved FAISS index, embeddings, and chunk metadata
def load_index():
    index = faiss.read_index(f"{SAVE_DIR}/faiss.index")
    embeddings = np.load(f"{SAVE_DIR}/embeddings.npy")
    with open(f"{SAVE_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, embeddings, chunks

if os.path.exists(f"{SAVE_DIR}/faiss.index"):
    index, X, chunks = load_index()
    print("Loaded saved index.")
else:
    print("No saved index found — build embeddings first.")

# Create a search tool for our RAG agent
def embed_texts(texts, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])
    return np.array(embs, dtype=np.float32)

def search_book(query: str, k: int = 6):
    q = embed_texts([query])
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        c = chunks[int(idx)]
        results.append({
            "score": float(score),
            "chapter": c["chapter"],
            "heading": c["heading"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "text": c["text"][:1200],  # keep tool output short
            "chunk_id": int(idx),
        })
    return results

def _extract_text_from_output(resp):
    parts = []
    for o in resp.output:
        # looking for assistant "message" outputs
        if getattr(o, "type", None) == "message":
            for c in getattr(o, "content", []) or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
                elif c.get("type") == "text":
                    parts.append(c.get("text", ""))
    return "\n".join([p for p in parts if p])

#########################################################
# Agent

# System prompt
SYSTEM = """You are a causal inference assistant grounded in a textbook.
You have access to a tool `search_book(query,k)` that returns relevant excerpts
with chapter + page ranges.

Rules:
1) Use `search_book` before recommending a method or giving step-by-step guidance.
2) When you propose code, tie each major step to a cited excerpt (chapter + pages).
3) Output format:

RECOMMENDATION
ASSUMPTIONS / CHECKS
PYTHON TEMPLATE
CITED SUPPORT (bullet list with Ch, pages, short description)

Be honest about what the book does/does not cover in the retrieved excerpts.
"""

# Tool definition for agent
tools = [{
    "type": "function",
    "name": "search_book",
    "description": "Search the uploaded textbook chapters for relevant passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 6}
        },
        "required": ["query"]
    }
}]


# Agent loop
def run_agent(user_question: str, model="gpt-4.1-mini", max_iters=8):
    if "index" not in globals() or "chunks" not in globals():
        raise RuntimeError("FAISS index not loaded. Build embeddings/index first (faiss.index missing).")

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_question},
        ],
        tools=tools,
        tool_choice="auto",
    )

    for _ in range(max_iters):
        if getattr(resp, "output_text", ""):
            return resp.output_text

        calls = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
        if not calls:
            return ""

        fc = next((c for c in calls if c.name == "search_book"), None)
        if fc is None:
            return ""

        args = json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments
        result = search_book(**args)

        call_id = getattr(fc, "call_id", None)
        if not call_id:
            raise RuntimeError("function_call object missing call_id (expected something like 'call_...').")

        resp = client.responses.create(
            model=model,
            previous_response_id=resp.id,
            input=[{
                "type": "function_call_output",
                "call_id": call_id,              # MUST be call_...
                "output": json.dumps(result),
            }],
        )

    raise RuntimeError(f"Agent did not finish within {max_iters} iterations.")



print(run_agent(
    "I have outcome Y, treatment D, and high-dimensional controls X under unconfoundedness. "
    "Recommend an approach and give a Python template with cross-fitting, and cite the book."
))
