"""Database connection and schema management for the chunk store.

Run directly to bootstrap the schema against the compose-managed database:

    python -m rag.db
"""

import logging

import psycopg
from pgvector.psycopg import register_vector

from rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


def connect(settings: Settings | None = None) -> psycopg.Connection:
    """Open a connection with the pgvector type registered."""
    settings = settings or get_settings()
    conn = psycopg.connect(settings.database_dsn)
    # The vector extension must exist before register_vector can look up the type.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def get_ingestion_meta(conn: psycopg.Connection) -> tuple[str, int] | None:
    """Return (embedding_model, dimension) the corpus was ingested with, or None."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('ingestion_meta')")
        if cur.fetchone()[0] is None:
            return None
        cur.execute(
            "SELECT embedding_model, embedding_dimension FROM ingestion_meta WHERE id = 1"
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else None


def get_chunks_dimension(conn: psycopg.Connection) -> int | None:
    """Return the vector dimension of the existing chunks table, or None.

    For pgvector columns the type modifier (atttypmod) is the declared dimension.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
    return row[0] if row and row[0] > 0 else None


def ensure_schema(conn: psycopg.Connection, settings: Settings | None = None) -> None:
    """Create the chunk store tables if missing.

    The vector column dimension comes from the active embedding profile, so a
    profile change is a config change — but it refuses to run against a
    database built for a *different* embedder: a mismatched query-time embedder
    would silently return garbage similarities.
    """
    settings = settings or get_settings()
    model = settings.resolved_embedding_model
    dim = settings.resolved_embedding_dimension

    ingested = get_ingestion_meta(conn)
    if ingested is not None and ingested != (model, dim):
        raise RuntimeError(
            f"Database was ingested with embedder {ingested[0]!r} (dim {ingested[1]}), "
            f"but the active profile resolves to {model!r} (dim {dim}). "
            "Re-ingest into a fresh database or switch EMBEDDING_PROFILE back."
        )
    existing_dim = get_chunks_dimension(conn)
    if existing_dim is not None and existing_dim != dim:
        raise RuntimeError(
            f"Existing chunks table has vector({existing_dim}) but the active "
            f"profile needs vector({dim}). Drop the table or switch EMBEDDING_PROFILE back."
        )

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_meta (
                id smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                embedding_model text NOT NULL,
                embedding_dimension integer NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        # dim is an int from validated settings, so the f-string is safe here;
        # a vector column dimension cannot be a bind parameter.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id text PRIMARY KEY,
                doc_id text NOT NULL,
                chunk_index integer NOT NULL,
                text text NOT NULL,
                embedding vector({dim}) NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb
            )
            """
        )
        # Ingestion idempotency checks look up chunks by their source document.
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id)")
    conn.commit()
    logger.info(
        "schema ready: profile=%s model=%s dim=%d",
        settings.embedding_profile,
        model,
        dim,
    )


if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    with connect() as bootstrap_conn:
        ensure_schema(bootstrap_conn)
