"""Interactive terminal chat with the RAG pipeline.

Run: python src/chat.py [fixed_size|recursive]   (default: recursive)
"""
import sys
import chunking
import rag

strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed_size"  # matches README's recommended strategy
if strategy not in chunking.STRATEGIES:
    sys.exit(f"unknown strategy {strategy!r}, choose from {list(chunking.STRATEGIES)}")

text = open("data/corpus.txt", encoding="utf-8").read()
chunks = chunking.STRATEGIES[strategy](text)
index = rag.build_index(chunks)
print(f"[{strategy}] {len(chunks)} chunks indexed. Ask about Paul Graham's essay. 'quit' to exit.")

while True:
    try:
        q = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q or q.lower() in ("quit", "exit"):
        break
    retrieved = rag.retrieve(q, index, chunks, k=3)
    print(rag.generate_answer(q, retrieved))
