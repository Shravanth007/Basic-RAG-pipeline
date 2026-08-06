# Error analysis

## Bug 1: embedding model too weak for exact-fact queries

First pass used `all-MiniLM-L6-v2`. For the factual query ("What language
did Paul Graham write his new Lisp, Bel, in?"), the gold sentence ("I wrote
this new Lisp, called Bel, in itself in Arc.") didn't appear in the top 10
retrieved chunks for the `fixed_size` strategy at all, and ranked 9th
(score 0.35) for `recursive` — both effectively invisible at k=3. The
retrieved chunks were topically adjacent (other Lisp/McCarthy passages) but
not the answer.

Root cause: MiniLM is a general sentence-similarity model, not tuned for
asymmetric query-to-passage retrieval. Swapped to `BAAI/bge-small-en-v1.5`
with its recommended query-side instruction prefix — the same gold sentence
moved to rank 2 (recursive) immediately, verified with a standalone rank
check before rerunning the full pipeline. Not a chunking-strategy artifact;
would have affected any strategy equally, which is why it was worth fixing
before trusting the comparison.

## Bug 2: hard line-wraps in the scraped corpus

The source HTML hard-wraps paragraphs at ~70 characters using single `<br>`
tags rendered as literal `\n` (not `\n\n`, which the site uses only between
paragraphs). First-pass corpus cleaning collapsed `<br><br>` into paragraph
breaks correctly but left single mid-paragraph `\n` characters in place.
Concretely: "he firmly refused to work on it" was stored as
`"he firmly\nrefused to work on it"`.

Effects: (1) the `recursive` splitter's separator list includes `"\n"` at
higher priority than `". "`, so it was silently splitting on every
mid-sentence line-wrap as if it were a real structural boundary — degrading
that strategy's whole reason for existing; (2) keyword gold-checks for
Precision@3 that assumed contiguous phrases (e.g. `"firmly refused"`) failed
to match even when the relevant sentence was fully present in a chunk,
because of the embedded newline.

Fix: normalize `data/corpus.txt` by collapsing single newlines to spaces
while preserving blank-line paragraph breaks, before any chunking happens.
Re-ran the full pipeline after the fix; Precision@3 changed materially
(fixed_size: 0.33 → 0.53; recursive: 0.27 → 0.40), confirming this wasn't
a cosmetic issue.

## Real finding: fixed_size beat recursive on Precision@3

Counter to the usual expectation that structure-aware chunking wins,
`fixed_size` scored higher on Precision@3 (0.53 vs 0.40); Relevance came out
tied (0.52 each). Likely explanation for the precision gap: this essay's
paragraphs are short and conversational (mean recursive chunk ≈ 552 chars,
median ≈ 565, well under its 800-char budget — see the histogram), so `recursive`
chunks often stop at a single short paragraph, while `fixed_size`'s
200-token window (≈900 chars) spans 2-3 paragraphs regardless of where they
end. On this specific text, wider fixed windows happened to capture more of
the surrounding narrative context per retrieval slot. This is a
corpus-specific result, not a general claim — flagged explicitly in
README.md's recommendation and LIMITATIONS.md.

## Note: LLM-judge scores drift slightly between identical runs

Re-running the full pipeline after an unrelated code fix (tightening the
judge-score regex in `src/rag.py`'s `_score_1_to_5`) changed the per-query
Relevance scores for several
queries even though retrieval, chunking, and prompts were byte-identical —
e.g. `fixed_size`'s multi-hop Relevance moved from 4/5 to 2/5, and
`recursive`'s negation Relevance moved from 1/5 to 5/5. Groq's API is called
with `temperature=0`, which should be close to deterministic but evidently
isn't perfectly so on their inference stack. Net effect on the aggregate
Relevance score was small this time (0.60/0.36 → 0.52/0.52), but it's a real
caveat for anyone trying to reproduce these exact numbers — see
LIMITATIONS.md.

## Failure case: comparison query (see `outputs/failure_case.md`)

Query: "How did Paul Graham's motivation for starting Viaweb compare to
his motivation for starting Y Combinator?" — Precision@3 = 0.00 for both
strategies.

The two answer-bearing passages ("I only started Viaweb because I needed
the money" and the Y Combinator founding narrative around "Screw the VCs
who were taking so long...") sit roughly 900 paragraphs apart in the essay.
A single dense query embedding pulled chunks that happen to *mention* both
"Viaweb" and "Y Combinator" by name (e.g. a passage about pricing strategy
used in both companies) rather than the chunks that actually explain the
motivation behind either. The generator correctly declined to answer rather
than fabricate a comparison — faithfulness stayed high (5/5) but relevance
cratered (1/5), because "I don't have enough information" is a faithful
non-answer to a question the system was fully capable of answering, just
not with 3 chunks from one query vector.

This is a structural limitation of single-pass dense retrieval on
comparison/multi-hop queries, not something either chunking strategy can
fix on its own — see README.md's recommendation for mitigations (reranking,
or decomposing comparison queries into sub-queries per entity before
retrieval).

## Secondary pattern: summarization query gets P=1.00 but R=1/5

For both strategies, all 3 retrieved chunks for the summarization query
contained at least one career-stage keyword (Precision@3 = 1.00), yet the
generator still answered "not enough information" (Relevance = 1/5). This
isn't a retrieval failure — it's k=3 being structurally too small to cover
a 13-stage, 20-year career arc, regardless of which 3 chunks are chosen.
Confirms the k-tuning recommendation in README.md is about query type, not
chunking strategy.
