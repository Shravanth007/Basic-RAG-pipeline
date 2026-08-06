"""End-to-end pipeline: chunk -> embed/index -> retrieve -> generate -> evaluate.

Run: python src/run.py
Writes everything under outputs/.
"""
import json
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import chunking
import rag
from queries import QUERIES

CORPUS_PATH = "data/corpus.txt"
OUT_DIR = "outputs"


def run_strategy(name, chunk_fn, text):
    print(f"[{name}] chunking...")
    chunks = chunk_fn(text)
    sizes_chars = [len(c) for c in chunks]
    sizes_tokens = [chunking.token_count(c) for c in chunks]

    print(f"[{name}] embedding {len(chunks)} chunks...")
    index = rag.build_index(chunks)

    query_results = []
    for q in QUERIES:
        print(f"[{name}] query: {q['id']}")
        retrieved = rag.retrieve(q["question"], index, chunks, k=3)
        precision = sum(1 for r in retrieved if q["gold"](r["chunk"])) / len(retrieved)
        answer = rag.generate_answer(q["question"], retrieved)
        faithfulness = rag.judge_faithfulness(answer, retrieved)
        relevance = rag.judge_relevance(q["question"], answer)
        query_results.append({
            "query_id": q["id"],
            "question": q["question"],
            "retrieved": [{"score": r["score"], "chunk": r["chunk"]} for r in retrieved],
            "answer": answer,
            "precision_at_3": precision,
            "faithfulness_1_5": faithfulness,
            "relevance_1_5": relevance,
        })

    return {
        "strategy": name,
        "chunk_count": len(chunks),
        "avg_chunk_chars": stats.mean(sizes_chars),
        "avg_chunk_tokens": stats.mean(sizes_tokens),
        "chunk_sizes_chars": sizes_chars,
        "avg_precision_at_3": stats.mean(r["precision_at_3"] for r in query_results),
        "avg_faithfulness": stats.mean(r["faithfulness_1_5"] for r in query_results) / 5,
        "avg_relevance": stats.mean(r["relevance_1_5"] for r in query_results) / 5,
        "query_results": query_results,
    }


def write_comparison_table(results):
    rows = [{
        "strategy": r["strategy"],
        "chunk_count": r["chunk_count"],
        "avg_chunk_size_chars": round(r["avg_chunk_chars"], 1),
        "avg_chunk_size_tokens": round(r["avg_chunk_tokens"], 1),
        "precision_at_3": round(r["avg_precision_at_3"], 2),
        "faithfulness": round(r["avg_faithfulness"], 2),
        "relevance": round(r["avg_relevance"], 2),
    } for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/comparison_table.csv", index=False)
    print(df.to_string(index=False))
    return df


def write_comparison_chart(df):
    metrics = ["precision_at_3", "faithfulness", "relevance"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(df))
    width = 0.25
    for i, m in enumerate(metrics):
        ax.bar([xi + i * width for xi in x], df[m], width, label=m)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(df["strategy"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality by chunking strategy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/comparison_chart.png", dpi=150)
    plt.close(fig)


def write_histogram(results):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        ax.hist(r["chunk_sizes_chars"], bins=20, alpha=0.6, label=r["strategy"])
    ax.set_xlabel("chunk size (characters)")
    ax.set_ylabel("count")
    ax.set_title("Chunk size distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chunk_size_histogram.png", dpi=150)
    plt.close(fig)


def write_failure_case(results):
    worst = None
    worst_score = 999
    for r in results:
        for qr in r["query_results"]:
            score = qr["precision_at_3"] + qr["faithfulness_1_5"] / 5 + qr["relevance_1_5"] / 5
            if score < worst_score:
                worst_score = score
                worst = (r["strategy"], qr)

    strategy, qr = worst
    lines = [
        "# Failure case",
        "",
        f"**Strategy:** {strategy}",
        f"**Query ({qr['query_id']}):** {qr['question']}",
        f"**Precision@3:** {qr['precision_at_3']:.2f}  "
        f"**Faithfulness:** {qr['faithfulness_1_5']}/5  "
        f"**Relevance:** {qr['relevance_1_5']}/5",
        "",
        "## Retrieved chunks",
    ]
    for i, r in enumerate(qr["retrieved"]):
        lines.append(f"\n**Chunk {i+1}** (score={r['score']:.3f}):\n\n> {r['chunk'][:500]}")
    lines += ["", "## Generated answer", "", qr["answer"]]
    with open(f"{OUT_DIR}/failure_case.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWorst case: {strategy} / {qr['query_id']} (combined score {worst_score:.2f})")


def main():
    text = open(CORPUS_PATH, encoding="utf-8").read()
    results = [run_strategy(name, fn, text) for name, fn in chunking.STRATEGIES.items()]

    with open(f"{OUT_DIR}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df = write_comparison_table(results)
    write_comparison_chart(df)
    write_histogram(results)
    write_failure_case(results)
    print("\nDone. See outputs/")


if __name__ == "__main__":
    main()
