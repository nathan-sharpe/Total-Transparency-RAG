"""FastAPI entry point — a thin layer over the rag/ modules.

    uvicorn main:app          (or: python main.py)

Everything the endpoints do is importable and callable without HTTP; eval
scripts use the rag/ functions directly, never this server.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from rag import db
from rag.config import get_settings
from rag.embedding import Embedder, get_embedder
from rag.generation import NO_ANSWER_RESPONSE, generate_answer, ground_citations
from rag.notify import notify_error
from rag.retrieval import is_answerable, retrieve, verify_corpus_compatible

# LOG_DIR (unset locally; the compose api service points it at a volume-mounted
# directory) adds a file handler so logs persist beyond `docker compose down`.
_log_handlers: list[logging.Handler] = [logging.StreamHandler()]
if os.getenv("LOG_DIR"):
    _log_dir = Path(os.environ["LOG_DIR"])
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_handlers.append(logging.FileHandler(_log_dir / "api.log", encoding="utf-8"))
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=_log_handlers,
)
logger = logging.getLogger("api")


def check_corpus_ready(conn: psycopg.Connection, embedder: Embedder) -> bool:
    """True once an ingested corpus exists; raises if it mismatches the embedder.

    The distinction matters at startup: a *mismatched* corpus is a config error
    and must fail fast, but an *empty* database is the normal state of a fresh
    clone whose container starts before first ingestion — the API serves 503
    on /query until a corpus appears instead of crash-looping.
    """
    if db.get_ingestion_meta(conn) is None:
        return False
    verify_corpus_compatible(conn, embedder)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.embedder = get_embedder(settings)
    with db.connect(settings) as conn:
        app.state.corpus_ready = check_corpus_ready(conn, app.state.embedder)
    if app.state.corpus_ready:
        logger.info(
            "startup checks passed: corpus matches embedder %s", app.state.embedder.model_name
        )
    else:
        logger.warning("no ingested corpus yet — /query returns 503 until ingestion runs")
    yield


app = FastAPI(title="SciFact RAG", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_within_limits(cls, value: str) -> str:
        # Guardrail 1 (input validation): reject blank queries and enforce the
        # configured length cap. Checked here rather than in Field(...) so the
        # limit is read from settings at request time, not import time.
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        limit = get_settings().max_query_chars
        if len(value) > limit:
            raise ValueError(f"query exceeds the {limit}-character limit")
        return value


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    # Guardrail 3b: citations the model emitted that don't exist in the
    # retrieved set. Their markers are stripped from `answer`; they're
    # surfaced here so a non-empty list flags the response as suspect.
    ungrounded_citations: list[str] = []
    sources: list[SourceChunk]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    with db.connect(settings) as conn:
        if not app.state.corpus_ready:
            # Startup found an empty database; ingestion may have run since,
            # so re-check before refusing (no API restart needed after ingest).
            if not check_corpus_ready(conn, app.state.embedder):
                raise HTTPException(
                    status_code=503,
                    detail="no corpus ingested yet — run: python ingest.py --dataset scifact",
                )
            app.state.corpus_ready = True
        chunks = retrieve(conn, app.state.embedder, request.query, settings=settings)
    sources = [
        SourceChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, score=c.score, text=c.text)
        for c in chunks
    ]

    # Guardrail 2 (no-answer path): if retrieval didn't clear the threshold,
    # refuse honestly and skip the generator entirely — no LLM call, no risk of
    # a confidently wrong answer grounded in irrelevant chunks.
    if not is_answerable(chunks, settings.no_answer_threshold):
        top = chunks[0].score if chunks else 0.0
        logger.info(
            "no-answer: top score %.4f below threshold %.2f", top, settings.no_answer_threshold
        )
        return QueryResponse(answer=NO_ANSWER_RESPONSE, citations=[], sources=sources)

    result = generate_answer(request.query, chunks, settings=settings)
    # Guardrail 3b (citation grounding): only chunk IDs actually retrieved may
    # be cited; hallucinated IDs are stripped from the answer and flagged.
    grounded = ground_citations(result.answer, [c.chunk_id for c in chunks])
    return QueryResponse(
        answer=grounded.answer,
        citations=grounded.grounded_citations,
        ungrounded_citations=grounded.ungrounded_citations,
        sources=sources,
    )


@app.get("/health")
def health() -> dict:
    """OK only if the app can reach the database (what platforms poll)."""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
    return {"status": "ok"}


# Fire-and-forget notification tasks need a strong reference until they finish,
# or the event loop may garbage-collect them mid-flight.
_notify_tasks: set[asyncio.Task] = set()


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    # Backstop guardrail: no stack trace ever reaches the client. The error_id
    # links the client's response to the full detail in the logs.
    error_id = uuid.uuid4().hex[:12]
    logger.exception("unhandled error %s on %s %s", error_id, request.method, request.url.path)
    settings = get_settings()
    if settings.error_webhook_url:
        # Fire-and-forget: a slow or dead webhook must not delay the error
        # response. The message carries the exception *class* only — the
        # message text could contain connection details; full detail stays
        # in the logs under the error_id.
        task = asyncio.create_task(
            notify_error(
                f"[rag-api] unhandled {type(exc).__name__} — error_id={error_id} "
                f"on {request.method} {request.url.path}",
                settings=settings,
            )
        )
        _notify_tasks.add(task)
        task.add_done_callback(_notify_tasks.discard)
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error", "error_id": error_id},
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port)
