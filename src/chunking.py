"""Two chunking strategies for the RAG pipeline.

fixed_size:   naive sliding window over raw token counts. Ignores sentence
              and paragraph boundaries entirely -- the baseline everyone
              starts with.
recursive:    structure-aware. Splits on paragraph breaks first, then
              sentences, then words, so a chunk boundary lands on a natural
              boundary whenever the size budget allows it.
"""
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

ENC = tiktoken.get_encoding("cl100k_base")


def fixed_size(text, chunk_tokens=200, overlap_tokens=50):
    tokens = ENC.encode(text)
    stride = chunk_tokens - overlap_tokens
    chunks = []
    for start in range(0, len(tokens), stride):
        window = tokens[start:start + chunk_tokens]
        if not window:
            break
        chunks.append(ENC.decode(window).strip())
        if start + chunk_tokens >= len(tokens):
            break
    return [c for c in chunks if c]


def recursive(text, chunk_chars=800, overlap_chars=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


STRATEGIES = {
    "fixed_size": fixed_size,
    "recursive": recursive,
}


def token_count(text):
    return len(ENC.encode(text))
