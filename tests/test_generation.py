"""Generation tests against a mock Ollama transport: prompt assembly,
resource guardrails (chunk cap, token limit), and citation extraction."""

import json

import httpx

from rag.config import Settings
from rag.generation import (
    NO_ANSWER_RESPONSE,
    SYSTEM_PROMPT,
    GeneratedAnswer,
    build_user_prompt,
    extract_citations,
    generate_answer,
    ground_citations,
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


def test_system_prompt_embeds_exact_refusal_string():
    # Eval-sensitive: the generator's refusal and guardrail 2's pre-generation
    # refusal must be the same wording, and this exact text is baked into the
    # frozen prompt. Changing it invalidates EVALS.md baselines.
    assert NO_ANSWER_RESPONSE == "I can't answer that from the documents I have."
    assert f"reply exactly: {NO_ANSWER_RESPONSE}" in SYSTEM_PROMPT


def test_user_prompt_labels_chunks_by_id():
    prompt = build_user_prompt("why?", [make_chunk(0), make_chunk(1)])
    assert "[4983::0]\nchunk text 0" in prompt
    assert "[4983::1]\nchunk text 1" in prompt
    assert prompt.endswith("Question: why?")


def test_ground_citations_all_valid_leaves_answer_untouched():
    answer = "Claim [4983::0]. More [4983::1]."
    grounded = ground_citations(answer, ["4983::0", "4983::1"])
    assert grounded.answer == answer
    assert grounded.grounded_citations == ["4983::0", "4983::1"]
    assert grounded.ungrounded_citations == []


def test_ground_citations_strips_hallucinated_bracket():
    grounded = ground_citations("True [4983::0]. Invented [999::9].", ["4983::0"])
    assert grounded.answer == "True [4983::0]. Invented."
    assert grounded.grounded_citations == ["4983::0"]
    assert grounded.ungrounded_citations == ["999::9"]


def test_ground_citations_filters_within_multi_id_bracket():
    grounded = ground_citations("Claim [4983::0, 999::9].", ["4983::0"])
    assert grounded.answer == "Claim [4983::0]."
    assert grounded.ungrounded_citations == ["999::9"]


def test_ground_citations_leaves_non_citation_brackets_alone():
    grounded = ground_citations("Quote [sic] and claim [4983::0].", ["4983::0"])
    assert grounded.answer == "Quote [sic] and claim [4983::0]."
    assert grounded.ungrounded_citations == []


def test_ground_citations_no_citations():
    grounded = ground_citations("No citations here.", ["4983::0"])
    assert grounded.answer == "No citations here."
    assert grounded.grounded_citations == []
    assert grounded.ungrounded_citations == []


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
