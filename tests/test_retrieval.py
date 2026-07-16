"""Guardrail 2 decision (is_answerable) — pure, no database needed."""

from rag.retrieval import RetrievedChunk, is_answerable


def chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="d::0", doc_id="d", text="t", score=score)


def test_answerable_when_top_score_meets_threshold():
    assert is_answerable([chunk(0.80), chunk(0.40)], threshold=0.6) is True


def test_answerable_at_exact_threshold():
    assert is_answerable([chunk(0.60)], threshold=0.6) is True


def test_not_answerable_below_threshold():
    assert is_answerable([chunk(0.55), chunk(0.90)], threshold=0.6) is False


def test_not_answerable_when_nothing_retrieved():
    assert is_answerable([], threshold=0.6) is False
