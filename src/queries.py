"""5 hand-written queries against data/corpus.txt (Paul Graham, "What I Worked On").

Each query has a `gold` function: a deterministic, auditable check on whether
a *retrieved chunk* actually contains the passage needed to answer the
question. This is used for Precision@3 instead of an LLM judge, so the
retrieval metric doesn't depend on the same model being evaluated.
"""

def _has_all(text, *needles):
    t = text.lower()
    return all(n.lower() in t for n in needles)


def _has_any(text, *needles):
    t = text.lower()
    return any(n.lower() in t for n in needles)


QUERIES = [
    {
        "id": "factual",
        "question": "What language did Paul Graham write his new Lisp, Bel, in, and what made that unusual?",
        "gold": lambda c: _has_all(c, "bel") and _has_any(c, "in itself", "in arc"),
    },
    {
        "id": "multihop",
        "question": "Paul Graham created two programming languages, Arc and Bel. In what years did he start and finish each one?",
        "gold": lambda c: (
            _has_all(c, "arc") and _has_any(c, "2000", "spring of 2000", "dan and i switched")
        ) or (
            _has_all(c, "bel") and _has_any(c, "2019", "march 26, 2015", "finished")
        ),
    },
    {
        "id": "negation",
        "question": "Did Robert Morris agree to work with Paul Graham on the Aspra startup idea in 2000?",
        "gold": lambda c: _has_any(c, "firmly refused", "he firmly refused to work on it"),
    },
    {
        "id": "comparison",
        "question": "How did Paul Graham's motivation for starting Viaweb compare to his motivation for starting Y Combinator?",
        "gold": lambda c: _has_any(
            c, "needed the money", "why don't i become rich", "get rich"
        ) or _has_any(
            c, "screw the vcs", "fixed about venture capital", "march 11", "converged"
        ),
    },
    {
        "id": "summarization",
        "question": "Summarize Paul Graham's career path from college through founding Y Combinator.",
        "gold": lambda c: _has_any(
            c, "cornell", "risd", "interleaf", "viaweb", "yahoo", "y combinator", "hacker news"
        ),
    },
]
