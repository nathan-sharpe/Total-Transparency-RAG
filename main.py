"""FastAPI entry point — a thin layer over the rag/ modules.

    uvicorn main:app          (or: python main.py)

Everything the endpoints do is importable and callable without HTTP; eval
scripts use the rag/ functions directly, never this server.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from rag import db
from rag.config import get_settings
from rag.embedding import get_embedder
from rag.generation import NO_ANSWER_RESPONSE, generate_answer, ground_citations
from rag.retrieval import is_answerable, retrieve, verify_corpus_compatible

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.embedder = get_embedder(settings)
    # Fail fast at startup, not on the first query, if the DB is unreachable
    # or was ingested with a different embedder.
    with db.connect(settings) as conn:
        verify_corpus_compatible(conn, app.state.embedder)
    logger.info("startup checks passed: corpus matches embedder %s", app.state.embedder.model_name)
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


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    # Backstop guardrail: no stack trace ever reaches the client. The error_id
    # links the client's response to the full detail in the logs.
    # (Phase 4 adds a webhook notification here.)
    error_id = uuid.uuid4().hex[:12]
    logger.exception("unhandled error %s on %s %s", error_id, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error", "error_id": error_id},
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port)
