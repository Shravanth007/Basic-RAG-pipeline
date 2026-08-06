# Limitations

- **Single corpus, single domain.** All results are specific to one
  13,884-word autobiographical essay. The finding that fixed_size beats
  recursive chunking here (see ERROR_ANALYSIS.md) is very likely reversed
  on denser, longer-paragraph text (legal contracts, technical docs) — this
  was not tested and shouldn't be generalized.

- **5 queries is a small, noisy sample.** Precision@3 is quantized to
  {0, 0.33, 0.67, 1.0} per query, and every reported metric is a mean over
  just 5 data points. A single query's outcome can swing the aggregate
  score by 0.1-0.2. This is enough to illustrate qualitative differences
  between strategies, not enough for a statistically confident ranking.

- **Same LLM plays generator and judge.** Groq's `llama-3.3-70b-versatile`
  both answers the questions and scores its own (and the other strategy's)
  answers for faithfulness/relevance. Self-grading bias is a known issue in
  LLM-as-judge setups — a model may rate its own answer style more
  favorably than an independent judge would. Precision@3 avoids this by
  being keyword-based instead, but Faithfulness and Relevance do not.

- **Fixed k=3 for every query type.** As shown in ERROR_ANALYSIS.md, k=3
  structurally cannot serve summarization-shaped queries well regardless of
  chunking strategy. No query-adaptive k was implemented.

- **Gold-check keyword functions are a heuristic, not ground truth.**
  Written by hand against the source text, but they check for specific
  phrases/keywords, not deep semantic relevance. A chunk paraphrasing the
  right content without using the exact tracked phrase would be scored as
  irrelevant (false negative on precision). This trades recall of the
  metric itself for auditability (see DECISIONS.md) — a deliberate
  trade-off, not an oversight.

- **No reranking, hybrid search, or query decomposition** — deliberately
  out of scope per DECISIONS.md, but it means the pipeline as built will
  keep failing comparison/multi-hop queries the way `outputs/failure_case.md`
  shows, until one of those is added.

- **No source citation shown to the end user.** The generated answers don't
  currently surface which chunk(s) they came from in the response itself
  (only in the eval harness's logging) — a production system should show
  this so users can verify claims themselves rather than trusting
  faithfulness scores they never see.

- **Local embedding model quality ceiling.** `bge-small-en-v1.5` is the
  small variant, chosen for speed on a laptop with no GPU. A larger BGE or
  a commercial embedding API would likely retrieve better on harder queries
  (comparison/multi-hop) — not tested here, since this pipeline intentionally
  stayed within free/local tooling.

- **`temperature=0` on Groq is not fully deterministic.** Re-running the
  identical pipeline (same chunks, same prompts) produced different
  per-query Relevance judge scores on a second run — see ERROR_ANALYSIS.md
  for the specific before/after numbers. The reported metrics are a single
  run, not an average over repeats; a more rigorous eval would run each
  judge call 3-5 times and report the mean/variance.

- **Free-tier Groq rate limits.** Not hit during this run, but a larger
  query set or corpus would need retry/backoff handling that isn't
  implemented — `src/rag.py` makes a bare API call with no retry logic.
