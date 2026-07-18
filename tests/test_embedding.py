"""Embedder tests against a mock Ollama transport — no server needed.

The SentenceTransformersEmbedder is deliberately untested here: constructing
it downloads a model, which belongs in CI (Phase 5), not unit tests.
"""

import json

import httpx
import pytest

from rag.config import Settings
from rag.embedding import (
    NOMIC_DOCUMENT_PREFIX,
    NOMIC_QUERY_PREFIX,
    OllamaEmbedder,
    get_embedder,
)


def make_settings(**overrides) -> Settings:
    # _env_file=None keeps the test hermetic against the developer's .env, but
    # not against real environment variables — CI exports
    # EMBEDDING_PROFILE=sentence-transformers, which would silently swap the
    # model these Ollama-specific tests assert on. Pin the profile they test.
    overrides.setdefault("embedding_profile", "ollama")
    return Settings(_env_file=None, postgres_user="t", postgres_password="t", **overrides)


def mock_ollama(seen_payloads: list, dimension: int) -> httpx.MockTransport:
    """Fake /api/embed: records each request payload, returns fixed vectors."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_payloads.append(payload)
        vectors = [[0.1] * dimension for _ in payload["input"]]
        return httpx.Response(200, json={"embeddings": vectors})

    return httpx.MockTransport(handler)


def test_nomic_prefixes_are_applied():
    payloads = []
    embedder = OllamaEmbedder(
        make_settings(embedding_dimension=3), transport=mock_ollama(payloads, dimension=3)
    )

    embedder.embed_documents(["alpha", "beta"])
    query_vector = embedder.embed_query("gamma")

    assert payloads[0]["model"] == "nomic-embed-text"
    assert payloads[0]["input"] == [
        NOMIC_DOCUMENT_PREFIX + "alpha",
        NOMIC_DOCUMENT_PREFIX + "beta",
    ]
    assert payloads[1]["input"] == [NOMIC_QUERY_PREFIX + "gamma"]
    assert query_vector == [0.1, 0.1, 0.1]


def test_wrong_dimension_from_server_is_rejected():
    # Profile says 768 (nomic default); the fake server returns 5-dim vectors.
    embedder = OllamaEmbedder(make_settings(), transport=mock_ollama([], dimension=5))
    with pytest.raises(ValueError, match="768"):
        embedder.embed_query("query")


def test_server_error_propagates():
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    embedder = OllamaEmbedder(make_settings(), transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_query("query")


def test_factory_selects_ollama_profile():
    assert isinstance(get_embedder(make_settings(embedding_profile="ollama")), OllamaEmbedder)
