"""Hand-built text chunker (no LangChain — that's the point of the project).

Sizes are measured in words (whitespace-delimited), not model tokens: word
counts need no tokenizer dependency, are identical across environments, and
track token counts closely enough for chunk-size sweeps (English prose runs
roughly 1.3 model tokens per word).

Chunking is a sliding window: chunk_size words per chunk, stepping
chunk_size - chunk_overlap words each time, so consecutive chunks share
chunk_overlap words and the final chunk always reaches the end of the text.
Chunk text is re-joined with single spaces, i.e. whitespace-normalized
relative to the source.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic chunk ID — also the primary key of the chunks table and
    the label the generator is asked to cite."""
    return f"{doc_id}::{chunk_index}"


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping word-window pieces.

    Empty or whitespace-only text yields no pieces. A text of chunk_size words
    or fewer yields exactly one.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap} "
            f"for chunk_size {chunk_size}"
        )

    words = text.split()
    if not words:
        return []

    stride = chunk_size - chunk_overlap
    pieces: list[str] = []
    start = 0
    while True:
        pieces.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            return pieces
        start += stride


def chunk_document(doc_id: str, text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk one document's text into Chunks with deterministic IDs.

    The caller decides what "text" is — ingestion prepends the title so it is
    searchable, e.g. f"{title}\\n\\n{body}".
    """
    return [
        Chunk(
            chunk_id=make_chunk_id(doc_id, index),
            doc_id=doc_id,
            chunk_index=index,
            text=piece,
        )
        for index, piece in enumerate(chunk_text(text, chunk_size, chunk_overlap))
    ]
