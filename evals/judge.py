"""LLM-as-judge for generation quality (Tier 2), hand-rolled.

Given the query, the context chunks the generator saw, and its answer, a
*different* local model scores the answer on a rubric. Two dimensions:

- **faithfulness** (1-5): is every claim in the answer supported by the
  context chunks? (5 = fully supported, 1 = contradicts or invents freely)
- **relevance** (1-5): does the answer actually address the question?

RAGAS's third headline metric, context *precision*, is deliberately not
judged: Tier 1 already measures retrieval quality against real golden-set
labels, which beats asking an 8B model to guess relevance.

The rubric prompt is EVAL-SENSITIVE: changing it invalidates prior Tier-2
numbers in EVALS.md and must be run as a measured experiment.

Guardrail 3a lives here: judge output is constrained to a JSON schema at the
Ollama API level AND schema-validated on receipt — retry once (with the
validation error fed back), then fail loudly with JudgeError. Garbage never
enters metrics. Guardrail 4 as at every LLM call site: timeout, capped
context, output token limit.
"""

import logging
import time
from collections.abc import Sequence

import httpx
from pydantic import BaseModel, Field, ValidationError

from rag.config import Settings, get_settings
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class JudgeError(RuntimeError):
    """Judge output failed schema validation twice — the verdict is unusable."""


class JudgeVerdict(BaseModel):
    """The schema judge output must satisfy. `reasoning` is declared first so
    the constrained decoder makes the model reason before it scores."""

    reasoning: str = Field(max_length=2000)
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)


JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of answers produced by a retrieval-augmented \
system over scientific literature. You are given a question, the context \
chunks the system retrieved, and the answer it produced. Score the answer:

- faithfulness (1-5): Is every claim in the answer supported by the context \
chunks? 5 = every claim is directly supported; 3 = mostly supported with \
minor unsupported additions; 1 = contradicts the context or invents facts. \
Judge ONLY against the provided chunks, not your own knowledge.
- relevance (1-5): Does the answer address the question asked? 5 = directly \
and completely answers it; 3 = partially addresses it; 1 = off-topic or \
evasive.

First write one or two sentences of reasoning, then the scores. Respond with \
JSON only, matching: {"reasoning": string, "faithfulness": 1-5, "relevance": 1-5}."""


def build_judge_prompt(
    query: str, chunks: Sequence[RetrievedChunk], answer: str
) -> str:
    context = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in chunks)
    return (
        f"Question: {query}\n\nContext chunks:\n\n{context}\n\n"
        f"Answer to evaluate:\n{answer}"
    )


def judge_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    answer: str,
    settings: Settings | None = None,
    transport: httpx.BaseTransport | None = None,
) -> JudgeVerdict:
    """Score one answer. Raises JudgeError after a failed retry (guardrail 3a)."""
    settings = settings or get_settings()
    if len(chunks) > settings.max_context_chunks:
        chunks = chunks[: settings.max_context_chunks]

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(query, chunks, answer)},
    ]
    started = time.monotonic()
    with httpx.Client(
        base_url=settings.ollama_url,
        timeout=settings.judge_timeout_seconds,
        transport=transport,
    ) as client:
        last_error: Exception | None = None
        for attempt in (1, 2):
            payload = {
                "model": settings.judge_model,
                "messages": messages,
                "stream": False,
                # Constrained decoding: Ollama enforces this JSON schema on the
                # output. Validation below still runs — the schema constrains
                # shape, not bounds like 1 <= score <= 5.
                "format": JudgeVerdict.model_json_schema(),
                "options": {
                    "temperature": settings.judge_temperature,
                    "num_predict": settings.judge_max_output_tokens,
                },
            }
            response = client.post("/api/chat", json=payload)
            response.raise_for_status()
            raw = response.json()["message"]["content"]
            try:
                verdict = JudgeVerdict.model_validate_json(raw)
            except ValidationError as error:
                last_error = error
                logger.warning(
                    "judge output failed validation (attempt %d): %s", attempt, error
                )
                # Feed the failure back so the retry isn't a deterministic
                # repeat of the same invalid output (temperature is 0).
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "That response failed validation: "
                            f"{error}. Respond again with ONLY valid JSON "
                            "matching the required schema."
                        ),
                    },
                ]
                continue
            logger.debug(
                "judged in %.1fs: faithfulness=%d relevance=%d",
                time.monotonic() - started,
                verdict.faithfulness,
                verdict.relevance,
            )
            return verdict

    raise JudgeError(f"judge output failed schema validation twice: {last_error}")
