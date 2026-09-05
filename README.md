# Tokenized Yield Explainer
 
A retrieval-augmented generation (RAG) question-answering assistant that answers questions about tokenized real-world assets using only a fixed set of source documents, and cites the source behind every claim.
 
I built this to understand two things at once: how a RAG system actually works under the hood, and how tokenized US Treasury products (Ondo's USDY and rOUSG, alongside Circle and Coinbase) deliver and account for yield. It answers from its documents or tells you it does not know, rather than guessing.
 
## The question it is built to answer
 
> How does USDY (accruing) differ from rOUSG (rebasing), and what are the accounting and tax implications of each?
 
Answering this well means pulling the yield mechanics from Ondo's material and the tax principles from primary IRS guidance, then reasoning across them. Example output, trimmed:
 
- USDY is an accumulating token: yield shows up as a rising token price while the holder's token count stays fixed. [usdy.md]
- rOUSG is a rebasing token: the price stays near one dollar and the holder's token count grows through a daily rebasing factor. [rousg.md]
- Applying IRS principles as an inference: crypto is treated as property, so USDY's appreciation would generally be taxed on disposition, closer to a capital gain [irs-notice-2014-21.pdf], while reward tokens received are income at receipt, so rOUSG's rebased tokens look more like ordinary income when received [irs-rev-rul-2023-14.pdf].
- The assistant labels this as reasoning by analogy, notes the IRS documents do not name these specific tokens, and flags that USDY is not offered to US persons. It is not tax advice.
## Other questions it answers
 
The flagship question is the hardest one, but the same pipeline handles narrower, single-document lookups just as well, for example:
 
- What was Coinbase's total trading volume in 2025, and how much did it grow?
- What backs USDC, and who examined the reserves?
- Are staking rewards taxable, and when are they included in income?
- Who is eligible to hold USDY?
Each answer cites the document it came from, or says it does not know when the corpus does not cover it.
 
## How it works
 
A RAG system has four parts, and this repo builds each one explicitly instead of hiding them behind a framework:
 
1. **Documents.** Source files in `docs/`, both markdown and PDF.
2. **Retriever.** Finds the passages relevant to a question.
3. **LLM.** Claude writes the answer using only those passages.
4. **Wrapper.** A simple command-line loop you type questions into.
The retriever is the interesting part:
 
- Text extraction with `pypdf`, then fixed-size overlapping chunks so messy PDF text splits cleanly.
- Local embeddings with `sentence-transformers` (all-MiniLM-L6-v2) and cosine similarity, so no embedding API is needed.
- Query expansion: Claude rewrites each question into several focused sub-queries, which fixes multi-hop questions where the right document shares no words with the question (for example, "rebasing" versus "staking rewards").
- A per-source diversity cap, so one long document cannot crowd the others out of the results.
Generation is grounded: the prompt requires every fact to come from the retrieved passages and to be cited, allows clearly labeled reasoning across passages, and refuses when the sources do not support an answer.
 
## Corpus
 
- `usdy.md`, `rousg.md`: concise summaries of Ondo's public USDY and OUSG/rOUSG documentation.
- `coinbase-q4-2025-shareholder-letter.pdf`: Coinbase Q4 and full-year 2025 shareholder letter.
- `circle-usdc-reserve.pdf`: Circle USDC reserve examination report.
- `irs-notice-2014-21.pdf`, `irs-rev-rul-2023-14.pdf`: primary IRS guidance on virtual currency and staking rewards.
## Run it
 
```
git clone https://github.com/manyadavis/tokenized-yield-explainer.git
cd tokenized-yield-explainer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
Create a `.env` file with your Anthropic API key:
 
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```
 
Then run:
 
```
python3 app.py
```
 
Ask a question, or type `quit` to exit. The first run downloads the embedding model once.
 
## Tech
 
Python, Claude (Haiku 4.5) via the Anthropic API for generation and query expansion, `sentence-transformers` for local embeddings, `pypdf` for extraction, and `numpy` for similarity. There is no vector database; embeddings are held in memory, which is plenty at this corpus size.
 
## Honest limitations
 
- The Ondo entries are short summaries of Ondo's public docs, not full primary filings.
- The tax output is reasoning by analogy from primary IRS sources, clearly labeled, and is not tax advice.
- Small embedding model and an in-memory store: fine for this corpus, not built for scale.
## Notes
 
Built as a learning project. The guiding choice throughout was to build each RAG component by hand rather than clone a template, so I could explain every part of it.