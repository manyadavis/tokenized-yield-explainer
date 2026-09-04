import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import anthropic
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"

def read_file(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:  # .md or .txt
        with open(path) as f:
            text = f.read()
    # collapse all the wrapped line breaks and extra spaces into single spaces
    return " ".join(text.split())

def chunk_text(text, size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap        # step forward, leaving an overlap
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

# PART 1: load, chunk, and embed everything in docs/
chunks = load_chunks(DOCS_DIR)
print(f"Loaded {len(chunks)} chunks from {DOCS_DIR}/")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

client = anthropic.Anthropic()

# PART 2: the retriever
def retrieve(query, k=5):
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = chunk_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]

print("Ask a question. Type quit to exit.")

while True:
    question = input("\nQuestion: ")
    if question.lower() == "quit":
        break
    top_chunks = retrieve(question)
    print("Retrieved from:", ", ".join(c["source"] for c in top_chunks))
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in top_chunks)
    prompt = f"""Answer the question using ONLY the context passages below.
If the answer is not in the context, reply exactly: I don't know based on the documents.
Cite the source file in brackets right after each fact you use.

Context:
{context}

Question: {question}"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    print("\n" + response.content[0].text)