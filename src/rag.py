"""Embedding + retrieval + generation + LLM-judge scoring.

Embeddings: sentence-transformers BAAI/bge-small-en-v1.5, run locally (no
API key needed, no per-call cost -- there's no reason to pay for an
embedding API for a 14k-word corpus). BGE is a retrieval-tuned model, not
a generic sentence-similarity one like MiniLM -- swapped in after MiniLM
measurably missed the gold passage for several hand-written queries even
at k=10 (see ERROR_ANALYSIS.md). BGE expects an instruction prefix on the
query side only, not on passages -- that asymmetry is intentional.

Generation + judging: Groq (llama-3.3-70b-versatile), since that's the
API key available for this assignment. Same model plays two roles
(answer generator and judge) -- see LIMITATIONS.md for why that's a
caveat, not a design flaw to hide.
"""
import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_GROQ = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def embed(texts, is_query=False):
    if is_query:
        texts = [_QUERY_PREFIX + t for t in texts]
    vecs = _EMBEDDER.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_index(chunks):
    vecs = embed(chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def retrieve(query, index, chunks, k=3):
    qvec = embed([query], is_query=True)
    scores, ids = index.search(qvec, k)
    return [
        {"chunk": chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1
    ]


def _chat(prompt, max_tokens=400):
    resp = _GROQ.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def generate_answer(question, retrieved_chunks):
    context = "\n\n---\n\n".join(r["chunk"] for r in retrieved_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain enough information, say so explicitly "
        "instead of guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return _chat(prompt)


def _score_1_to_5(prompt):
    reply = _chat(prompt, max_tokens=10).strip()
    m = re.match(r"[1-5]\b", reply) or re.search(r"\b[1-5]\b", reply)
    return int(m.group()) if m else 3  # ponytail: unparseable judge reply defaults to neutral


def judge_faithfulness(answer, retrieved_chunks):
    context = "\n\n---\n\n".join(r["chunk"] for r in retrieved_chunks)
    prompt = (
        "You are grading whether an ANSWER is fully supported by the given CONTEXT, "
        "ignoring whether it's factually true in the real world -- only whether the "
        "context supports it. Rate 1 (unsupported / hallucinated) to 5 (fully supported). "
        "Reply with ONLY the digit.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\nScore (1-5):"
    )
    return _score_1_to_5(prompt)


def judge_relevance(question, answer):
    prompt = (
        "Rate how well the ANSWER addresses the QUESTION on a scale of 1 (off-topic / "
        "doesn't answer it) to 5 (directly and completely answers it). "
        "Reply with ONLY the digit.\n\n"
        f"Question: {question}\n\nAnswer:\n{answer}\n\nScore (1-5):"
    )
    return _score_1_to_5(prompt)
