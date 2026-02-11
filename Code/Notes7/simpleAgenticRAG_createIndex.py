import re
import fitz
import pickle
import faiss
import numpy as np
from openai import OpenAI
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to store embeddings and index
SAVE_DIR = os.path.join(script_dir, "book_index")
os.makedirs(SAVE_DIR, exist_ok=True)

# Files that we're indexing
chapter8 = os.path.join(script_dir, "CausalML_chap_8.pdf")
chapter9 = os.path.join(script_dir, "CausalML_chap_9.pdf")

####################################################
# Helper functions to extract and chunk pdf text
def extract_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        text = doc.load_page(i).get_text("text")
        pages.append({"page": i + 1, "text": text})
    return pages

HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+(.+?)\s*$")

def chunk_pages(pages, chapter_label: str, max_chars=3500):
    """
    Chunk by detected headings; fall back to size-based splits.
    Stores (chapter, page_start, page_end, heading, text).
    """
    chunks = []
    cur = {"chapter": chapter_label, "page_start": None, "page_end": None,
           "heading": None, "text": ""}

    def flush():
        nonlocal cur
        if cur["text"].strip():
            chunks.append(cur)
        cur = {"chapter": chapter_label, "page_start": None, "page_end": None,
               "heading": None, "text": ""}

    for p in pages:
        page_num = p["page"]
        lines = p["text"].splitlines()

        for line in lines:
            m = HEADING_RE.match(line)
            if m and len(line) < 120:  # crude guard against false positives
                # new section => flush current chunk
                flush()
                cur["heading"] = f"{m.group(1)} {m.group(3)}"
                cur["page_start"] = page_num
                cur["page_end"] = page_num
                cur["text"] = line.strip() + "\n"
                continue

            if cur["page_start"] is None:
                cur["page_start"] = page_num
            cur["page_end"] = page_num

            # append
            cur["text"] += line + "\n"

            # size cap
            if len(cur["text"]) >= max_chars:
                flush()

    flush()
    return chunks

####################################################
# chunk pdfs
chap8_pages = extract_pages(chapter8)
chap9_pages = extract_pages(chapter9)

chap8_chunks = chunk_pages(chap8_pages, chapter_label="Ch8")
chap9_chunks = chunk_pages(chap9_pages, chapter_label="Ch9")

chunks = chap8_chunks + chap9_chunks
print("Number of chunks:", len(chunks))
print("Chunk fields:", chunks[0].keys() if chunks else "No chunks created")

####################################################
# Helper functions for embedding and indexing

def embed_texts(texts, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])
    return np.array(embs, dtype=np.float32)

def save_index(index, embeddings, chunks):
    faiss.write_index(index, f"{SAVE_DIR}/faiss.index")
    np.save(f"{SAVE_DIR}/embeddings.npy", embeddings)
    with open(f"{SAVE_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


######################################################
# Create and save index
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"  # prototyping, might want to switch to a larger model if
                                        # wanted to put into production

chunk_texts = [c["text"] for c in chunks]
X = embed_texts(chunk_texts)

# Normalize for cosine similarity via inner product
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

save_index(index, X, chunks)
print("Index saved.")
