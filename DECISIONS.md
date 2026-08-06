# Design decisions

## RAG architecture: single-pass dense retrieval, not hybrid/reranked/agentic

Considered and rejected for this assignment:

- **Hybrid retrieval (BM25 + dense)** or **cross-encoder reranking** — both
  would likely raise Precision@3, but they introduce a second independent
  variable on top of chunking strategy. The assignment's grading axis is
  "how does chunking strategy affect retrieval quality" — bolting on a
  stronger retriever muddies that comparison rather than sharpening it.
- **Multi-query expansion / HyDE** — same problem: it would paper over
  retrieval weaknesses that are actually caused by chunk boundaries, which
  is exactly what this eval is supposed to expose.
- **Agentic/iterative retrieval, GraphRAG** — explicit scope creep against
  "minimal RAG pipeline." No loop, no graph construction; one embed, one
  search, one generation call, per query per strategy.

So: one retriever (FAISS flat, cosine), one generator, two chunkers, held
everything else constant. That isolates the variable being tested.

## Chunking strategies

1. **Fixed-size** (`chunking.fixed_size`): tiktoken `cl100k_base` token
   count, 200-token window, 50-token overlap, sliding regardless of
   sentence/paragraph boundaries. The naive baseline nearly everyone starts
   with in production.
2. **Recursive/structure-aware** (`chunking.recursive`): LangChain's
   `RecursiveCharacterTextSplitter`, separators `["\n\n", "\n", ". ", " ",
   ""]` in priority order, 800 chars / 100 overlap. Falls back through
   paragraph → sentence → word only when a higher-priority boundary doesn't
   fit the size budget.

Two strategies, not three, because the assignment asks for "at least 2" and
a third (e.g. semantic/embedding-breakpoint chunking) wouldn't add a
meaningfully different axis of comparison for a single 14k-word essay —
it would mostly reproduce what `recursive` already does on this text.

## Embedding model: bge-small, not MiniLM

Started with `all-MiniLM-L6-v2` (the default "good enough" choice). Testing
against the hand-written queries showed it missing the correct passage
entirely at k=10 for several queries (see ERROR_ANALYSIS.md for the actual
rank numbers). Switched to `BAAI/bge-small-en-v1.5`, a retrieval-tuned
model that expects an instruction prefix on the query side only
(`"Represent this sentence for searching relevant passages: "`) — not
applied to passages, since BGE's asymmetric design intentionally
distinguishes queries from documents. This alone moved several gold
passages from outside top-10 into top-3.

Both are local/free (`sentence-transformers`) — no reason to pay for an
embedding API on a 14k-word corpus.

## Precision@3: deterministic keyword-check, not LLM-judged

`queries.py` defines a `gold` function per query: a boolean check on
whether a *retrieved chunk's text* contains the specific phrase(s) needed to
answer it (e.g. for the negation query, whether the chunk contains "firmly
refused"). This is intentionally not an LLM judge, for two reasons:

- It keeps the *retrieval* metric independent of the *generation* model
  being evaluated — otherwise a bad judge could mask a bad retriever or
  vice versa.
- It's auditable. Anyone can open `queries.py`, read the exact string being
  checked for, and verify it against the source essay by hand. An
  LLM-judged precision score would be a black box on top of a black box.

Faithfulness and Relevance *are* LLM-judged (Groq, 1-5 rubric), because
"is this answer supported by this context" and "does this answer the
question" don't reduce to substring checks the way "is the right fact in
this chunk" does.

## Corpus choice

Paul Graham's "What I Worked On" — fetched live from paulgraham.com rather
than transcribed from memory, specifically to avoid introducing
hallucinated "ground truth" into the very corpus being used to test
hallucination. Chosen over a legal contract or generic Wikipedia article
because its autobiographical, chronological structure naturally supports
multi-hop ("what happened between X and Y") and comparison ("how did A
differ from B") queries without contrivance.
