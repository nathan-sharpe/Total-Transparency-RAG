"""Pluggable embedder interface with two implementations.

The interface separates embed_documents from embed_query because good
retrieval embedders are asymmetric: nomic-embed-text is trained with task
prefixes ("search_document: " / "search_query: ") that must be applied at
embedding time, and other models make the same distinction differently.

Profiles (selected by EMBEDDING_PROFILE, defined in rag/config.py):
- "ollama" -> OllamaEmbedder, nomic-embed-text on the host-native Ollama
  server. The primary, local-AI path.
- "sentence-transformers" -> SentenceTransformersEmbedder, all-MiniLM-L6-v2
  on CPU. Used by CI (Phase 5), where there is no GPU and no Ollama.

Whichever profile ingested the corpus is recorded in ingestion_meta; query
paths must refuse an embedder that mismatches it (see rag/db.py).
"""

import logging
from abc import ABC, abstractmethod

import httpx

from rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Task prefixes nomic-embed-text was trained with. Ollama does not apply them
# for you; omitting them measurably hurts retrieval quality.
NOMIC_DOCUMENT_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


class Embedder(ABC):
    """model_name and dimension are set by implementations in __init__; the
    pair is what gets recorded in (and checked against) ingestion_meta."""

    model_name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus chunks for storage."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query for similarity against stored chunks."""


class OllamaEmbedder(Embedder):
    """nomic-embed-text via Ollama's batch /api/embed endpoint.

    The nomic task prefixes are applied here, so callers pass raw text. If the
    model is ever overridden to a non-nomic one via EMBEDDING_MODEL, revisit
    the prefixes — they are model-specific training conventions.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = settings or get_settings()
        self.model_name = settings.resolved_embedding_model
        self.dimension = settings.resolved_embedding_dimension
        self._client = httpx.Client(
            base_url=settings.ollama_url,
            timeout=settings.embed_timeout_seconds,
            transport=transport,
        )
        logger.info("ollama embedder ready: model=%s dim=%d", self.model_name, self.dimension)

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = self._client.post(
            "/api/embed", json={"model": self.model_name, "input": inputs}
        )
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"{self.model_name} returned a {len(vector)}-dim vector but the "
                    f"active profile expects {self.dimension} — check EMBEDDING_PROFILE"
                )
        logger.debug("embedded batch of %d texts", len(inputs))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([NOMIC_DOCUMENT_PREFIX + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([NOMIC_QUERY_PREFIX + text])[0]


class SentenceTransformersEmbedder(Embedder):
    """all-MiniLM-L6-v2 (or profile override) on CPU via sentence-transformers.

    MiniLM has no document/query asymmetry, so both methods encode plainly.
    The import is deferred: it drags in torch, which the Ollama path never
    needs installed.
    """

    def __init__(self, settings: Settings | None = None):
        from sentence_transformers import SentenceTransformer

        settings = settings or get_settings()
        self.model_name = settings.resolved_embedding_model
        self.dimension = settings.resolved_embedding_dimension
        self._model = SentenceTransformer(self.model_name)
        actual = self._model.get_sentence_embedding_dimension()
        if actual != self.dimension:
            raise ValueError(
                f"{self.model_name} embeds at {actual} dims but the active profile "
                f"declares {self.dimension} — fix EMBEDDING_PROFILES in rag/config.py"
            )
        logger.info(
            "sentence-transformers embedder ready: model=%s dim=%d",
            self.model_name,
            self.dimension,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.encode(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text])[0].tolist()


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Build the embedder the active EMBEDDING_PROFILE selects."""
    settings = settings or get_settings()
    if settings.embedding_profile == "ollama":
        return OllamaEmbedder(settings)
    return SentenceTransformersEmbedder(settings)
