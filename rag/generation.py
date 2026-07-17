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

# The one canonical refusal string. The generator is instructed to emit it when
# the context is insufficient, and guardrail 2 (rag/retrieval.py + main.py)
# returns this same text when it refuses *before* generation — one wording for
# "not in my corpus" regardless of which layer produced it.
NO_ANSWER_RESPONSE = "I can't answer that from the documents I have."

SYSTEM_PROMPT = f"""\
You are a careful assistant answering questions about scientific literature.
Answer using ONLY the context chunks provided. Follow these rules:
- After each claim, cite the chunk ID that supports it in square brackets, e.g. [4983::0].
- Cite only chunk IDs that appear in the context.
- If the context does not contain the information needed to answer, reply exactly: \
{NO_ANSWER_RESPONSE}
- Be concise: a short paragraph at most."""

# Chunk IDs are {doc_id}::{chunk_index}; matched anywhere in the answer so
# bracketed lists like [4983::0, 4983::1] are caught too.
CITATION_PATTERN = re.compile(r"[\w.-]+::\d+")

# A bracketed group like [4983::0] or [4983::0, 123::1] — the unit that gets
# rewritten when grounding strips an invalid citation.
_CITATION_GROUP = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    cited_chunk_ids: list[str]
    model: str


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str  # answer text with any ungrounded citation markers stripped
    grounded_citations: list[str]
    ungrounded_citations: list[str]


def _strip_citations(answer: str, drop: set[str]) -> str:
    """Remove the given chunk IDs from bracketed citation groups in the answer.

    A bracket that only contained dropped IDs disappears entirely; brackets
    holding anything that isn't a chunk ID (e.g. "[sic]") are left untouched.
    """

    def rewrite(match: re.Match) -> str:
        tokens = [t.strip() for t in match.group(1).split(",")]
        if not all(CITATION_PATTERN.fullmatch(t) for t in tokens):
            return match.group(0)  # not a citation group
        kept = [t for t in tokens if t not in drop]
        return f"[{', '.join(kept)}]" if kept else ""

    stripped = _CITATION_GROUP.sub(rewrite, answer)
    # Tidy what a removed bracket leaves behind: doubled spaces, a space
    # stranded before punctuation.
    stripped = re.sub(r" {2,}", " ", stripped)
    stripped = re.sub(r" ([.,;:!?])", r"\1", stripped)
    return stripped.strip()


def ground_citations(
    answer: str, retrieved_chunk_ids: Sequence[str]
) -> GroundedAnswer:
    """Guardrail 3b: verify cited chunk IDs against the retrieved set.

    The prompt *asks* the model to cite only provided chunks; this enforces it
    in code. Citations pointing outside the retrieved set (hallucinated IDs)
    are stripped from the answer text and reported separately so the caller
    can flag them. (Full per-claim faithfulness checking at serve time is the
    expensive strong version — documented, not built.)
    """
    retrieved = set(retrieved_chunk_ids)
    cited = extract_citations(answer)
    grounded = [c for c in cited if c in retrieved]
    ungrounded = [c for c in cited if c not in retrieved]
    if ungrounded:
        logger.warning(
            "ungrounded citations stripped from answer: %s", ", ".join(ungrounded)
        )
        answer = _strip_citations(answer, set(ungrounded))
    return GroundedAnswer(
        answer=answer, grounded_citations=grounded, ungrounded_citations=ungrounded
    )


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
