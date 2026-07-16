"""Grounded answer generation via Ollama chat.

The prompt format is EVAL-SENSITIVE: once Phase 2/3 baselines exist, any
change here invalidates prior EVALS.md numbers and must be run as a measured
experiment, not edited casually. Citations are requested from this very first
version for exactly that reason — retrofitting the format later would reset
the baselines. Citation *verification* is a Phase 3 guardrail; here we only
ask for and extract them.

Resource guardrails (guardrail 4) at this call site: request timeout, cap on
chunks fed in, output token limit — all from settings.
"""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from rag.config import Settings, get_settings
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a careful assistant answering questions about scientific literature.
Answer using ONLY the context chunks provided. Follow these rules:
- After each claim, cite the chunk ID that supports it in square brackets, e.g. [4983::0].
- Cite only chunk IDs that appear in the context.
- If the context does not contain the information needed to answer, reply exactly: \
I can't answer that from the documents I have.
- Be concise: a short paragraph at most."""

# Chunk IDs are {doc_id}::{chunk_index}; matched anywhere in the answer so
# bracketed lists like [4983::0, 4983::1] are caught too.
CITATION_PATTERN = re.compile(r"[\w.-]+::\d+")


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    cited_chunk_ids: list[str]
    model: str


def build_user_prompt(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in chunks)
    return f"Context chunks:\n\n{context}\n\nQuestion: {query}"


def extract_citations(answer: str) -> list[str]:
    """Chunk IDs cited in the answer, deduplicated, in order of first mention."""
    return list(dict.fromkeys(CITATION_PATTERN.findall(answer)))


def generate_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings | None = None,
    transport: httpx.BaseTransport | None = None,
) -> GeneratedAnswer:
    """Ask the generator model for a cited answer grounded in the chunks."""
    settings = settings or get_settings()
    if len(chunks) > settings.max_context_chunks:
        logger.debug(
            "capping context: %d chunks retrieved, feeding %d",
            len(chunks),
            settings.max_context_chunks,
        )
        chunks = chunks[: settings.max_context_chunks]

    payload = {
        "model": settings.generator_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, chunks)},
        ],
        "stream": False,
        "options": {
            "temperature": settings.generator_temperature,
            "num_predict": settings.max_output_tokens,
        },
    }
    started = time.monotonic()
    with httpx.Client(
        base_url=settings.ollama_url,
        timeout=settings.generation_timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post("/api/chat", json=payload)
        response.raise_for_status()
        answer = response.json()["message"]["content"].strip()

    cited = extract_citations(answer)
    logger.info(
        "generated answer in %.1fs: model=%s chars=%d citations=%d",
        time.monotonic() - started,
        settings.generator_model,
        len(answer),
        len(cited),
    )
    return GeneratedAnswer(answer=answer, cited_chunk_ids=cited, model=settings.generator_model)
