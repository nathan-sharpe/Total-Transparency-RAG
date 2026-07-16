"""Guardrail 1 tests: the API request model rejects bad input before any
retrieval or generation happens."""

import pytest
from pydantic import ValidationError

import main
from rag.config import Settings


@pytest.fixture(autouse=True)
def small_query_limit(monkeypatch):
    # The validator reads the limit via main.get_settings at request time,
    # so patching it keeps the test independent of the developer's .env.
    settings = Settings(
        _env_file=None, postgres_user="t", postgres_password="t", max_query_chars=50
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)


def test_valid_query_is_accepted_and_stripped():
    assert main.QueryRequest(query="  what causes X?  ").query == "what causes X?"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_queries_are_rejected(blank):
    with pytest.raises(ValidationError, match="empty"):
        main.QueryRequest(query=blank)


def test_overlong_query_is_rejected():
    with pytest.raises(ValidationError, match="50"):
        main.QueryRequest(query="x" * 51)


def test_whitespace_padding_does_not_evade_the_length_cap():
    # 50 real characters plus padding is fine (strip first)…
    assert len(main.QueryRequest(query="  " + "x" * 50 + "  ").query) == 50
    # …but 51 stripped characters is not.
    with pytest.raises(ValidationError):
        main.QueryRequest(query="  " + "x" * 51 + "  ")


def test_non_string_query_is_rejected():
    with pytest.raises(ValidationError):
        main.QueryRequest(query=123)
