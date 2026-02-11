# To execute, run this script with streamlit from within terminal:
# streamlit run "C:\Users\Chansen1\Documents\GitHub\MBA-MachineLearning\Spring 2026 Course Materials\Notes 7\agenticRAGdemo.py"
# adjust the path as needed for your setup

import os, json, pickle
import numpy as np
import streamlit as st
import faiss
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL_DEFAULT = "gpt-4.1-mini"

# ----------------------------
# THIS PROMPT REALLY USES THE BOOK! (see SYSTEM_RAG below)
#
#SYSTEM_RAG = """You are a causal inference assistant grounded in a textbook.
#You have access to a tool `search_book(query,k)` that returns relevant excerpts
#with chapter + page ranges.
#
#Rules:
#1) Use `search_book` before recommending a method or giving step-by-step guidance.
#2) When you propose code, tie each major step to a cited excerpt (chapter + pages).
#3) Output format:
#
#RECOMMENDATION
#ASSUMPTIONS / CHECKS
#PYTHON TEMPLATE
#CITED SUPPORT (bullet list with Ch, pages, short description)
#
#Be honest about what the book does/does not cover in the retrieved excerpts.
#"""

SYSTEM_RAG = """You are a helpful assistant that can answer questions on any topic.

You have an optional tool:
- search_book(query, k): returns relevant excerpts from a causal inference / causal ML textbook with chapter + page ranges.

Decision policy (follow strictly):
- If the question is likely answerable from the causal inference textbook (e.g., causal identification, estimands, unconfoundedness, propensity scores, IPW/AIPW, DML/cross-fitting, IV, RD, DiD, panel/event studies, policy learning, sensitivity/robustness, heterogeneous effects, ML nuisance estimation), then call search_book first.
- If the question is clearly unrelated (cooking, travel, sports, generic programming, etc.), do not call the tool.
- If you are uncertain whether it relates to causal inference, do one quick search_book call with a short query anyway, and use it only if results are relevant.

How to answer:
1) Start with a brief one-line “plan” only when helpful (e.g., “I’ll check the book for how it frames cross-fitting, then give a template.”). If you did not use the tool, do not mention it.
2) If you used search_book and results are relevant: ground the answer in those excerpts and cite “Ch X, pp. a–b” for key steps/claims.
3) If you used search_book but results are not relevant: say the excerpts didn’t directly address it, then answer from general knowledge.
4) Never refuse to answer just because the book doesn’t cover it.

Output format (only when the question is causal-methods related):
RECOMMENDATION
ASSUMPTIONS / CHECKS
PYTHON TEMPLATE
CITED SUPPORT (bullets with Ch, pages)

Otherwise, respond in the most natural format for the question.
Tone: clear, practical, not overly formal.
"""

SYSTEM_BASELINE = """You are a helpful assistant.
Answer normally without using any external documents or tools.
If you are unsure, say so.
"""

# ----------------------------
# Load index (cached)
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(script_dir, "book_index")

@st.cache_resource
def load_index():
    index = faiss.read_index(os.path.join(SAVE_DIR, "faiss.index"))
    _ = np.load(os.path.join(SAVE_DIR, "embeddings.npy"))  
    with open(os.path.join(SAVE_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

index, chunks = load_index()

# ----------------------------
# Embed + FAISS search
# ----------------------------
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
        if idx < 0:
            continue
        c = chunks[int(idx)]
        results.append({
            "score": float(score),
            "chapter": c.get("chapter"),
            "heading": c.get("heading"),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "text": (c.get("text") or "")[:1200],
            "chunk_id": int(idx),
        })
    return results

tools = [{
    "type": "function",
    "name": "search_book",
    "description": "Search the indexed textbook chunks for relevant passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 6}
        },
        "required": ["query"]
    }
}]

# ----------------------------
# RAG agent 
# ----------------------------
def run_agent_rag(user_question: str, model: str, k: int = 6, max_iters: int = 8):
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": user_question},
        ],
        tools=tools,
        tool_choice="auto",
    )

    last_retrieval = None

    for _ in range(max_iters):
        if getattr(resp, "output_text", ""):
            return resp.output_text, last_retrieval

        calls = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
        if not calls:
            return "", last_retrieval

        fc = next((c for c in calls if c.name == "search_book"), None)
        if fc is None:
            return "", last_retrieval

        args = json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments
        args["k"] = int(args.get("k", k))
        last_retrieval = search_book(**args)

        call_id = getattr(fc, "call_id", None)
        if not call_id:
            raise RuntimeError("function_call missing call_id")

        resp = client.responses.create(
            model=model,
            previous_response_id=resp.id,
            input=[{
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(last_retrieval),
            }],
        )

    return "ERROR: RAG agent did not finish (max_iters reached).", last_retrieval

def run_default(user_question: str, model: str):
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_BASELINE},
            {"role": "user", "content": user_question},
        ],
    )
    return resp.output_text or ""

# ----------------------------
# UI: dual chat
# ----------------------------
st.set_page_config(page_title="RAG vs Default Chat Demo", layout="wide")
st.title("Demo: RAG vs Default")

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Chat model", CHAT_MODEL_DEFAULT)
    k = st.slider("Top-k chunks", 2, 12, 6, 1)
    show_sources = st.checkbox("Show retrieved chunks (RAG)", value=False)
    if st.button("Clear conversation"):
        st.session_state.pop("history", None)

if "history" not in st.session_state:
    # each item: {"user": str, "rag": str, "base": str, "sources": list|None}
    st.session_state["history"] = []

# render history
left, right = st.columns(2)
with left:
    st.markdown("### RAG (uses indexed chapters)")
    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["rag"])
            if show_sources and turn.get("sources"):
                with st.expander("Retrieved chunks"):
                    for r in turn["sources"]:
                        st.markdown(f"**{r['chapter']} pp. {r['page_start']}-{r['page_end']}** — {r.get('heading') or ''}  \nScore: {r['score']:.3f}")
                        st.code(r["text"])

with right:
    st.markdown("### Default (no retrieval)")
    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["base"])

# live input
prompt = st.chat_input("Type a prompt… ")
if prompt:
    # show immediate user message in both columns by appending to history first
    st.session_state["history"].append({"user": prompt, "rag": "…", "base": "…", "sources": None})

    # run both
    rag_answer, sources = run_agent_rag(prompt, model=model, k=k)
    base_answer = run_default(prompt, model=model)

    st.session_state["history"][-1]["rag"] = rag_answer
    st.session_state["history"][-1]["base"] = base_answer
    st.session_state["history"][-1]["sources"] = sources

    st.rerun()
