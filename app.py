import os
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import anthropic
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"
MODEL = "claude-haiku-4-5-20251001"

def read_file(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        with open(path) as f:
            text = f.read()
    return " ".join(text.split())

def chunk_text(text, size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks

def load_chunks(docs_dir):
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not (filename.endswith(".md") or filename.endswith(".pdf")):
            continue
        text = read_file(os.path.join(docs_dir, filename))
        for piece in chunk_text(text):
            chunks.append({"text": piece, "source": filename})
    return chunks

print("Loading documents and embedding model...")
chunks = load_chunks(DOCS_DIR)
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
client = anthropic.Anthropic()

def expand_query(question):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": (
            "Break this question into 2 to 4 short, focused search queries that "
            "together cover everything needed to answer it. Return only the queries, "
            "one per line, with no numbering or extra text.\n\nQuestion: " + question
        )}],
    )
    subs = []
    for line in resp.content[0].text.splitlines():
        cleaned = line.lstrip("0123456789.-*) ").strip()
        if cleaned:
            subs.append(cleaned)
    return [question] + subs

def retrieve(question, k=10, per_source=3, pool=16):
    queries = expand_query(question)
    best = {}
    for q in queries:
        q_emb = model.encode(q, normalize_embeddings=True)
        scores = chunk_embeddings @ q_emb
        for i in np.argsort(scores)[::-1][:pool]:
            best[int(i)] = max(best.get(int(i), -1.0), float(scores[i]))
    ranked = sorted(best, key=lambda i: best[i], reverse=True)
    selected, counts = [], {}
    for i in ranked:
        src = chunks[i]["source"]
        if counts.get(src, 0) >= per_source:
            continue
        selected.append(chunks[i])
        counts[src] = counts.get(src, 0) + 1
        if len(selected) >= k:
            break
    return selected

def answer(question):
    if not question or not question.strip():
        return "Please enter a question.", ""
    top_chunks = retrieve(question)
    sources = ", ".join(sorted(set(c["source"] for c in top_chunks)))
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in top_chunks)
    prompt = f"""You are a research assistant. Answer using only the context passages below, and cite the source file in brackets after each fact you use.
The context often gives general principles (like IRS rules) without naming the specific product in the question. When that happens, apply those principles to the specific case and explain the most likely implication, clearly labeling it as reasoning or inference rather than definitive tax advice. Do not refuse just because the documents don't mention a product by name; reason from the principles they do give. Only call a part unsupported if the context truly offers nothing relevant to it.

Context:
{context}

Question: {question}"""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text, f"**Retrieved from:** {sources}"

demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(lines=2, label="Question",
                      placeholder="How does USDY differ from rOUSG, and what are the tax implications of each?"),
    outputs=[gr.Markdown(label="Answer"), gr.Markdown(label="Sources")],
    title="Tokenized Yield Explainer",
    description=("A retrieval-augmented assistant that answers questions about tokenized real-world assets "
                 "(Ondo USDY and rOUSG, Coinbase, Circle) using only a fixed set of documents, and cites its sources. "
                 "It answers from the documents or says it does not know."),
    examples=[
        ["How does USDY differ from rOUSG, and how is each likely taxed, capital gains versus ordinary income at receipt?"],
        ["What was Coinbase's total trading volume in 2025, and how much did it grow?"],
        ["What backs USDC, and who examined the reserves?"],
        ["Are staking rewards taxable, and when are they included in income?"],
    ],
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()