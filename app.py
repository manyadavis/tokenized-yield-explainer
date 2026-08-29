import anthropic

# PART 1 - the document, read once at startup
with open("usdy.md") as f:
    document = f.read()

# PART 3 - the LLM client, created once
client = anthropic.Anthropic()

print("Ask a question about USDY. Type quit to exit.")

while True:
    # PART 4 - the wrapper: you type the question here
    question = input("\nQuestion: ")
    if question.lower() == "quit":
        break

    prompt = f"""Answer the question using ONLY the document below.
If the answer is not in the document, reply exactly: I don't know based on the document.
After your answer, add a line that starts with "Source:" and quotes the exact sentence from the document you used.

Document:
{document}

Question: {question}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n" + response.content[0].text)