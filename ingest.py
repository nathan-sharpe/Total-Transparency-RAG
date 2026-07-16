"""Batch ingestion CLI: load a dataset, chunk, embed, insert into Postgres.

    python ingest.py --dataset scifact [--limit N]

Idempotent and resumable: a document whose doc_id already has chunks in the
database is skipped, and each document's chunks are inserted in one
transaction, so the existence check is a trustworthy resume marker. The
database is the progress record — rerunning after an interruption just picks
up where it left off.
"""

import argparse
import logging
import os
import time

from pgvector import Vector
from psycopg.types.json import Json

from rag import db
from rag.chunking import chunk_document
from rag.config import Settings, get_settings
from rag.datasets import DATASETS, Document, load_dataset
from rag.embedding import Embedder, get_embedder

logger = logging.getLogger("ingest")

PROGRESS_EVERY_DOCS = 100


def document_already_ingested(conn, doc_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM chunks WHERE doc_id = %s LIMIT 1", (doc_id,))
        return cur.fetchone() is not None


def record_ingestion_meta(conn, embedder: Embedder) -> None:
    """Record which embedder built this corpus (first run only — ensure_schema
    has already refused to proceed if an existing record mismatches)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingestion_meta (id, embedding_model, embedding_dimension)"
            " VALUES (1, %s, %s) ON CONFLICT (id) DO NOTHING",
            (embedder.model_name, embedder.dimension),
        )


def ingest_document(conn, embedder: Embedder, doc: Document, settings: Settings) -> int:
    """Chunk, embed, and insert one document atomically. Returns chunks written."""
    # The title is prepended so it is searchable; it also rides along in
    # metadata for citation display later.
    text = f"{doc.title}\n\n{doc.text}" if doc.title else doc.text
    chunks = chunk_document(doc.doc_id, text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        logger.debug("skip %s: document has no text", doc.doc_id)
        return 0
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, doc_id, chunk_index, text, embedding, metadata)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.chunk_index,
                    chunk.text,
                    Vector(vector),
                    Json({"title": doc.title}),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
    return len(chunks)


def run(dataset: str, limit: int | None) -> None:
    settings = get_settings()
    embedder = get_embedder(settings)
    with db.connect(settings) as conn:
        # Autocommit + explicit per-document conn.transaction() blocks: the
        # idempotency SELECTs don't hold a transaction open between documents.
        conn.autocommit = True
        db.ensure_schema(conn, settings)
        record_ingestion_meta(conn, embedder)

        started = time.monotonic()
        seen = ingested = skipped = chunk_count = 0
        for doc in load_dataset(dataset, settings):
            if limit is not None and seen >= limit:
                break
            seen += 1
            if document_already_ingested(conn, doc.doc_id):
                skipped += 1
                logger.debug("skip %s: already ingested", doc.doc_id)
            else:
                chunk_count += ingest_document(conn, embedder, doc, settings)
                ingested += 1
            if seen % PROGRESS_EVERY_DOCS == 0:
                rate = seen / (time.monotonic() - started)
                logger.info(
                    "progress: %d docs (%d ingested, %d skipped), %d chunks, %.1f docs/s",
                    seen,
                    ingested,
                    skipped,
                    chunk_count,
                    rate,
                )
        elapsed = time.monotonic() - started
        logger.info(
            "done: %d docs in %.1fs — %d ingested (%d chunks written), %d skipped",
            seen,
            elapsed,
            ingested,
            chunk_count,
            skipped,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-ingest a corpus into the chunk store.")
    parser.add_argument("--dataset", default="scifact", choices=sorted(DATASETS))
    parser.add_argument("--limit", type=int, default=None, help="process at most N documents")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # httpx logs one INFO line per request — thousands during ingestion.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    run(args.dataset, args.limit)


if __name__ == "__main__":
    main()
