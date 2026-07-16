"""Vector similarity retrieval: embed the query, cosine top-k in pgvector.

Chunks always come back with similarity scores — the Phase 2 no-answer
threshold (guardrail 2) and DEBUG diagnostics both depend on them.
"""

import logging
import time
from dataclasses import dataclass

import psycopg
from pgvector import Vector

from rag.config import Settings, get_settings
from rag.db import get_ingestion_meta
from rag.embedding import Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # cosine similarity: 1.0 = same direction, 0.0 = orthogonal


def is_answerable(chunks: list[RetrievedChunk], threshold: float) -> bool:
    """Guardrail 2 decision: does retrieval clear the no-answer bar?

    True iff at least one chunk came back and the best one's similarity is at
    or above the threshold. Pure and side-effect-free so it unit-tests without
    a database; the query flow (main.py) uses it to refuse before generating.
    """
    return bool(chunks) and chunks[0].score >= threshold


def verify_corpus_compatible(conn: psycopg.Connection, embedder: Embedder) -> None:
    """Refuse a query-time embedder that mismatches the ingested corpus —
    similarities between vectors from different models are garbage, silently."""
    ingested = get_ingestion_meta(conn)
    if ingested is None:
        raise RuntimeError(
            "no ingested corpus found — run: python ingest.py --dataset scifact"
        )
    if ingested != (embedder.model_name, embedder.dimension):
        raise RuntimeError(
            f"corpus was ingested with {ingested[0]!r} (dim {ingested[1]}) but the "
            f"active embedder is {embedder.model_name!r} (dim {embedder.dimension}); "
            "switch EMBEDDING_PROFILE back or re-ingest into a fresh database"
        )


def retrieve(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return the k most similar chunks to the query, best first."""
    settings = settings or get_settings()
    k = k or settings.top_k
    started = time.monotonic()

    query_vector = Vector(embedder.embed_query(query))
    with conn.cursor() as cur:
        # <=> is pgvector's cosine *distance*; similarity = 1 - distance.
        cur.execute(
            "SELECT id, doc_id, text, 1 - (embedding <=> %s) AS score"
            " FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (query_vector, query_vector, k),
        )
        chunks = [
            RetrievedChunk(chunk_id=row[0], doc_id=row[1], text=row[2], score=row[3])
            for row in cur.fetchall()
        ]

    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info("retrieved %d chunks in %.0f ms (k=%d)", len(chunks), elapsed_ms, k)
    for chunk in chunks:
        logger.debug("retrieved %s score=%.4f", chunk.chunk_id, chunk.score)
    return chunks
