"""Phase 0 smoke test: proves the whole substrate works end to end.

Connects to Postgres, creates the schema, inserts dummy vectors, and retrieves
the nearest one by cosine similarity. Requires the compose database:

    docker compose up -d
"""

import uuid

import psycopg
import pytest
from pgvector import Vector

from rag import db
from rag.config import get_settings


@pytest.fixture(scope="module")
def conn():
    try:
        connection = db.connect()
    except psycopg.OperationalError as exc:
        pytest.fail(
            f"Could not connect to Postgres: {exc}\n"
            "Is the database running? Try: docker compose up -d "
            "(and check the values in .env)"
        )
    yield connection
    connection.close()


def test_vector_round_trip(conn):
    settings = get_settings()
    dim = settings.resolved_embedding_dimension
    db.ensure_schema(conn, settings)

    doc_id = f"smoke-test-{uuid.uuid4()}"
    # Two orthogonal unit vectors; the query points almost exactly at chunk 0.
    target = [1.0] + [0.0] * (dim - 1)
    decoy = [0.0, 1.0] + [0.0] * (dim - 2)
    query = [0.9, 0.1] + [0.0] * (dim - 2)

    try:
        with conn.cursor() as cur:
            for i, vec in enumerate([target, decoy]):
                cur.execute(
                    "INSERT INTO chunks (id, doc_id, chunk_index, text, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (f"{doc_id}::{i}", doc_id, i, f"smoke chunk {i}", Vector(vec)),
                )
            # <=> is pgvector's cosine *distance*; similarity = 1 - distance.
            cur.execute(
                "SELECT id, 1 - (embedding <=> %s) AS similarity FROM chunks"
                " WHERE doc_id = %s ORDER BY embedding <=> %s LIMIT 1",
                (Vector(query), doc_id, Vector(query)),
            )
            best_id, similarity = cur.fetchone()

        assert best_id == f"{doc_id}::0"
        assert similarity > 0.85
    finally:
        # The inserts were never committed, so rollback leaves the DB untouched.
        conn.rollback()
