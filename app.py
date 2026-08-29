import os
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"

def load_chunks(docs_dir):
    chunks = []
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(docs_dir, filename)) as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs:
            if p.startswith("Source:"):
                continue
            chunks.append({"text": p, "source": filename})
    return chunks

# PART 1: load, chunk, and embed every document, once at startup
chunks = load_chunks(DOCS_DIR)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

# PART 3 setup: the LLM client
client = anthropic.Anthropic()

# PART 2: the retriever
def retrieve(query, k=5):   # widened from 4 so both mechanism passages come in
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = chunk_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]

print("Ask a question about USDY and rOUSG. Type quit to exit.")

while True:
    # PART 4: the wrapper you type into
    question = input("\nQuestion: ")
    if question.lower() == "quit":
        break

    top_chunks = retrieve(question)
    print("Retrieved from:", ", ".join(c["source"] for c in top_chunks))

    # label each retrieved chunk with its source file so Claude can cite it
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in top_chunks)

    prompt = f"""Answer the question using ONLY the context passages below.
If the answer is not in the context, reply exactly: I don't know based on the documents.
Cite the source file in brackets, like [usdy.md], right after each fact you use.

Context:
{context}

Question: {question}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n" + response.content[0].text)