# RAG Pipeline: Chunking Strategies & Retrieval Quality Evaluation

A minimal RAG pipeline over Paul Graham's essay ["What I Worked
On"](https://www.paulgraham.com/worked.html) (13,884 words), comparing two
chunking strategies across 5 hand-written queries (factual, multi-hop,
negation, comparison, summarization).

## Stack

| Component | Choice | Why |
|---|---|---|
| Corpus | Paul Graham, "What I Worked On" (Feb 2021) | Long-form, real, information-dense narrative — good for multi-hop/comparison/negation queries |
| Chunking A | Fixed-size token window (200 tokens, 50 overlap) | Naive baseline, ignores structure |
| Chunking B | Recursive/structure-aware (LangChain `RecursiveCharacterTextSplitter`, 800 chars, 100 overlap) | Splits on paragraph → sentence → word, in that order |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) | Retrieval-tuned; swapped in after `all-MiniLM-L6-v2` measurably missed gold passages — see ERROR_ANALYSIS.md |
| Vector store | FAISS `IndexFlatIP` (cosine via normalized vectors) | One flat index per strategy — exact search, no approximation needed at this scale |
| Generation + judging | Groq `llama-3.3-70b-versatile` | The API key available for this assignment |

See [DECISIONS.md](DECISIONS.md) for the full reasoning, including RAG
architectures considered and rejected.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY
python src/run.py
```

Outputs land in `outputs/`: `comparison_table.csv`, `comparison_chart.png`,
`chunk_size_histogram.png`, `failure_case.md`, `results.json` (full raw
per-query detail for both strategies).

## Project structure

```
data/corpus.txt        # the essay, cleaned plain text
src/chunking.py         # the two chunking strategies
src/queries.py           # 5 hand-written queries + deterministic gold-relevance checks
src/rag.py               # embedding, FAISS index, retrieval, generation, LLM-judge scoring
src/run.py                # orchestrates everything, writes outputs/
src/chat.py               # interactive terminal Q&A; optional arg: fixed_size or recursive
outputs/                  # generated table, charts, failure case, raw results
```

## Results

| strategy | chunks | avg chunk size (tokens) | Precision@3 | Faithfulness | Relevance |
|---|---|---|---|---|---|
| fixed_size | 111 | 198.7 | 0.53 | 1.00 | 0.52 |
| recursive | 136 | 122.1 | 0.40 | 1.00 | 0.52 |

(Faithfulness/Relevance are LLM-judge 1-5 scores, normalized to 0-1, averaged
across the 5 queries. Precision@3 uses deterministic keyword/phrase
gold-checks, not an LLM judge — see DECISIONS.md.)

Full breakdown and the one clean failure case (a comparison query where both
strategies retrieve chunks that lexically mention both companies but miss
the actual motivation passages, which sit hundreds of paragraphs apart) are
in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) and `outputs/failure_case.md`.

## Recommendation

Deploy the **fixed_size** strategy for this kind of narrative/autobiographical
corpus, at least as the starting baseline. It beat recursive chunking on
Precision@3 (0.53 vs 0.40) in this evaluation, which ran counter to the usual
expectation that structure-aware chunking wins; Relevance came out tied
(0.52 each), so the real edge here is retrieval precision, not answer
quality. The likely reason is that this essay's paragraphs are short, so a
200-token fixed window spans 2-3 paragraphs and captures more surrounding
context per retrieval slot than recursive chunking's smaller, tightly
paragraph-bounded chunks. That advantage is corpus-specific and would likely
reverse on a document with longer, denser paragraphs, such as a legal
contract, so this should not be read as a general rule. To mitigate
hallucination risk, keep the current "answer only from context, say so if
insufficient" instruction — it already worked in this eval, since the model
declined rather than fabricated on every low-precision retrieval, keeping
faithfulness at 1.00 even as relevance cratered. Add a lightweight
cross-encoder reranker on top of the top-10 dense candidates to fix the
comparison/multi-hop failure mode shown in the failure case, before it ever
reaches generation. Finally, route summarization-shaped queries to a larger
k (5-10 instead of 3), since a single 3-chunk window cannot structurally
cover a whole career arc regardless of chunking strategy.

## AI tools used

Built with Claude Code (Sonnet 5) for code, corpus retrieval/cleaning, and
debugging. Generation and LLM-judge scoring inside the pipeline itself use
Groq's `llama-3.3-70b-versatile`. Validation: every metric in the results
table was produced by actually running `src/run.py` against the live Groq
API (not fabricated), and one retrieval bug (embedding model too weak) and
one data bug (hard line-wraps in the scraped corpus breaking chunk/keyword
boundaries) were caught by manually inspecting `results.json` and re-running
targeted rank checks before trusting the numbers — see ERROR_ANALYSIS.md.
