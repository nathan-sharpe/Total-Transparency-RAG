"""Application settings: the single source of truth for configuration.

Every value comes from environment variables (or a local .env file read by
pydantic-settings). No other module reads os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Each profile names an embedding model and the dimension of the vectors it
# produces. The dimension is needed as early as schema creation, because the
# pgvector column is declared vector(N). Profiles are defined here (in the
# config module) so switching is an env-var change, never a code change.
EMBEDDING_PROFILES: dict[str, dict] = {
    "ollama": {"model": "nomic-embed-text", "dimension": 768},
    "sentence-transformers": {"model": "all-MiniLM-L6-v2", "dimension": 384},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    postgres_user: str
    postgres_password: str
    postgres_db: str = "rag"
    db_host: str = "localhost"
    db_port: int = 5432

    # Ollama (containers reach the host-native server at http://host.docker.internal:11434)
    ollama_url: str = "http://localhost:11434"

    # Logging
    log_level: str = "INFO"

    # Datasets: downloads land under data_dir (gitignored)
    data_dir: Path = Path("data")
    scifact_url: str = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    )

    # Chunking, measured in words — see rag/chunking.py for why words, not model tokens
    chunk_size: int = 200
    chunk_overlap: int = 40

    # Embedding: the profile picks sensible defaults; the explicit fields
    # override them when set (used by experiments, never required).
    embedding_profile: Literal["ollama", "sentence-transformers"] = "ollama"
    embedding_model: str | None = None
    embedding_dimension: int | None = None

    # Retrieval
    top_k: int = 5

    # Generation model + resource guardrails (guardrail 4: every LLM call site
    # has a timeout, a cap on chunks fed in, and an output token limit)
    generator_model: str = "llama3.1:8b"
    generator_temperature: float = 0.0
    generation_timeout_seconds: float = 120.0
    embed_timeout_seconds: float = 60.0
    max_context_chunks: int = 5
    max_output_tokens: int = 512

    # Input guardrail (guardrail 1): enforced by the API request model
    max_query_chars: int = 2000

    # API server (uvicorn) bind address, used by `python main.py`
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @property
    def resolved_embedding_model(self) -> str:
        return self.embedding_model or EMBEDDING_PROFILES[self.embedding_profile]["model"]

    @property
    def resolved_embedding_dimension(self) -> int:
        return self.embedding_dimension or EMBEDDING_PROFILES[self.embedding_profile]["dimension"]

    @property
    def database_dsn(self) -> str:
        # libpq keyword/value format, understood by psycopg.connect().
        return (
            f"host={self.db_host} port={self.db_port} dbname={self.postgres_db} "
            f"user={self.postgres_user} password={self.postgres_password}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
