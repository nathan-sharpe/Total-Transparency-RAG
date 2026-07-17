"""Judge tests against a mock Ollama transport: prompt assembly, schema
validation, and the retry-once-then-fail-loudly behavior (guardrail 3a)."""

import json

import httpx
import pytest

from evals.judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeError,
    JudgeVerdict,
    build_judge_prompt,
    judge_answer,
)
from rag.config import Settings
from rag.retrieval import RetrievedChunk


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, postgres_user="t", postgres_password="t", **overrides)


def make_chunk(index: int, doc_id: str = "4983") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}::{index}",
        doc_id=doc_id,
        text=f"chunk text {index}",
        score=0.9 - index * 0.1,
    )


def transport_returning(*contents: str) -> tuple[httpx.MockTransport, list[dict]]:
    """Mock Ollama that returns each content string in turn, recording payloads."""
    payloads: list[dict] = []
    replies = list(contents)

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": replies.pop(0)}})

    return httpx.MockTransport(handler), payloads


VALID = json.dumps({"reasoning": "Supported by chunk 0.", "faithfulness": 5, "relevance": 4})


def test_judge_prompt_is_locked():
    # Eval-sensitive: the rubric dimensions and scale are baked into the frozen
    # prompt. Changing them invalidates Tier-2 EVALS.md baselines.
    assert "faithfulness (1-5)" in JUDGE_SYSTEM_PROMPT
    assert "relevance (1-5)" in JUDGE_SYSTEM_PROMPT
    assert "Judge ONLY against the provided chunks" in JUDGE_SYSTEM_PROMPT


def test_judge_prompt_carries_query_chunks_and_answer():
    prompt = build_judge_prompt("why?", [make_chunk(0)], "Because [4983::0].")
    assert "Question: why?" in prompt
    assert "[4983::0]\nchunk text 0" in prompt
    assert prompt.endswith("Answer to evaluate:\nBecause [4983::0].")


def test_judge_valid_first_try():
    transport, payloads = transport_returning(VALID)
    verdict = judge_answer(
        "q", [make_chunk(0)], "a", settings=make_settings(), transport=transport
    )
    assert verdict == JudgeVerdict(
        reasoning="Supported by chunk 0.", faithfulness=5, relevance=4
    )
    payload = payloads[0]
    assert payload["model"] == make_settings().judge_model
    assert payload["format"] == JudgeVerdict.model_json_schema()  # constrained decoding
    assert payload["messages"][0]["content"] == JUDGE_SYSTEM_PROMPT


def test_judge_applies_resource_guardrails():
    transport, payloads = transport_returning(VALID)
    settings = make_settings(max_context_chunks=2, judge_max_output_tokens=77)
    judge_answer(
        "q",
        [make_chunk(0), make_chunk(1), make_chunk(2)],
        "a",
        settings=settings,
        transport=transport,
    )
    payload = payloads[0]
    assert "4983::1" in payload["messages"][1]["content"]
    assert "4983::2" not in payload["messages"][1]["content"]  # capped at 2 chunks
    assert payload["options"]["num_predict"] == 77


def test_judge_retries_once_with_error_fed_back():
    # First reply is out of bounds (score 7); the retry must include the
    # failure context and its valid reply must be accepted.
    invalid = json.dumps({"reasoning": "r", "faithfulness": 7, "relevance": 4})
    transport, payloads = transport_returning(invalid, VALID)
    verdict = judge_answer(
        "q", [make_chunk(0)], "a", settings=make_settings(), transport=transport
    )
    assert verdict.faithfulness == 5
    retry_messages = payloads[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": invalid}
    assert "failed validation" in retry_messages[-1]["content"]


def test_judge_fails_loudly_after_two_invalid():
    transport, _ = transport_returning("not json", '{"reasoning": "r"}')
    with pytest.raises(JudgeError, match="failed schema validation twice"):
        judge_answer(
            "q", [make_chunk(0)], "a", settings=make_settings(), transport=transport
        )
