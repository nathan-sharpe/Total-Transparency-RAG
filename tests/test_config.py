"""Settings resolution tests. No database required."""

from rag.config import Settings


def make_settings(**overrides) -> Settings:
    # _env_file=None keeps the test independent of any local .env file.
    base = dict(postgres_user="test", postgres_password="test", postgres_db="test")
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_ollama_profile_defaults():
    s = make_settings(embedding_profile="ollama")
    assert s.resolved_embedding_model == "nomic-embed-text"
    assert s.resolved_embedding_dimension == 768


def test_sentence_transformers_profile_defaults():
    s = make_settings(embedding_profile="sentence-transformers")
    assert s.resolved_embedding_model == "all-MiniLM-L6-v2"
    assert s.resolved_embedding_dimension == 384


def test_explicit_embedding_overrides_win():
    s = make_settings(embedding_profile="ollama", embedding_model="custom", embedding_dimension=123)
    assert s.resolved_embedding_model == "custom"
    assert s.resolved_embedding_dimension == 123
