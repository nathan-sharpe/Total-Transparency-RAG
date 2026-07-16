"""Generation tests against a mock Ollama transport: prompt assembly,
resource guardrails (chunk cap, token limit), and citation extraction."""

import json

import httpx

from rag.config import Settings
from rag.generation import (
    SYSTEM_PROMPT,
    GeneratedAnswer,
    build_user_prompt,
    extract_citations,
    generate_answer,
)
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


def test_extract_citations_dedupes_in_first_mention_order():
    answer = "Claim one [4983::0]. Claim two [123::4, 4983::0]. Also [doc-1.2::3]."
    assert extract_citations(answer) == ["4983::0", "123::4", "doc-1.2::3"]


def test_extract_citations_none_present():
    assert extract_citations("No citations in this answer.") == []


def test_user_prompt_labels_chunks_by_id():
    prompt = build_user_prompt("why?", [make_chunk(0), make_chunk(1)])
    assert "[4983::0]\nchunk text 0" in prompt
    assert "[4983::1]\nchunk text 1" in prompt
    assert prompt.endswith("Question: why?")


def test_generate_answer_applies_resource_guardrails():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "Answer [4983::0]. "}})

    settings = make_settings(max_context_chunks=2, max_output_tokens=99)
    result = generate_answer(
        "q",
        [make_chunk(0), make_chunk(1), make_chunk(2)],
        settings=settings,
        transport=httpx.MockTransport(handler),
    )

    payload = payloads[0]
    assert payload["model"] == settings.generator_model
    assert payload["messages"][0]["content"] == SYSTEM_PROMPT
    assert "4983::1" in payload["messages"][1]["content"]
    assert "4983::2" not in payload["messages"][1]["content"]  # capped at 2 chunks
    assert payload["options"]["num_predict"] == 99  # output token limit
    assert result == GeneratedAnswer(
        answer="Answer [4983::0].",  # stripped
        cited_chunk_ids=["4983::0"],
        model=settings.generator_model,
    )
