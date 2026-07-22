"""Application settings: the single source of truth for configuration.

Every value comes from environment variables (or a local .env file read by
pydantic-settings). No other module reads os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
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

    # Database. SecretStr so a Settings repr — e.g. echoed into a pytest
    # failure traceback — masks the value; the DSN property unwraps it.
    postgres_user: str
    postgres_password: SecretStr
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

    # Chunking, measured in words — see rag/chunking.py for why words, not model tokens.
    # 300/60 is the Phase 6 chunk-sweep winner on SciFact (see EVALS.md); the original
    # baseline was 200/40.
    chunk_size: int = 300
    chunk_overlap: int = 60

    # Embedding: the profile picks sensible defaults; the explicit fields
    # override them when set (used by experiments, never required).
    # "sentence-transformers" (all-MiniLM-L6-v2) is the adopted default — Phase 6
    # experiments showed it beats the original "ollama"/nomic-embed-text profile on
    # SciFact (see EVALS.md). Generation and judging remain local Ollama regardless.
    embedding_profile: Literal["ollama", "sentence-transformers"] = "sentence-transformers"
    embedding_model: str | None = None
    embedding_dimension: int | None = None

    # Retrieval
    top_k: int = 5

    # Guardrail 2 (no-answer path): if the best retrieved chunk's cosine
    # similarity is below this, the query flow returns an honest refusal
    # WITHOUT calling the generator. Thresholds are PROFILE-SPECIFIC: 0.36 is the
    # Phase 6 sweep value for the adopted sentence-transformers profile (the original
    # nomic/ollama profile tuned to 0.60). See EVALS.md.
    no_answer_threshold: float = 0.36

    # Generation model + resource guardrails (guardrail 4: every LLM call site
    # has a timeout, a cap on chunks fed in, and an output token limit)
    generator_model: str = "llama3.1:8b"
    generator_temperature: float = 0.0
    generation_timeout_seconds: float = 120.0
    embed_timeout_seconds: float = 60.0
    max_context_chunks: int = 5
    max_output_tokens: int = 512

    # Judge (Phase 3, Tier-2 evals): deliberately a different model from
    # generator_model so the generator never scores its own output. Same
    # resource guardrails as every LLM call site (guardrail 4).
    judge_model: str = "qwen2.5:7b"
    judge_temperature: float = 0.0
    judge_timeout_seconds: float = 120.0
    judge_max_output_tokens: int = 512

    # Input guardrail (guardrail 1): enforced by the API request model
    max_query_chars: int = 2000

    # Backstop notification (Phase 4): when set, the API's global exception
    # handler posts a short alert (error id + route, never the stack trace)
    # to this Slack/Discord incoming-webhook URL. Unset = feature off.
    # SecretStr: the URL embeds the webhook token.
    error_webhook_url: SecretStr | None = None
    webhook_timeout_seconds: float = 5.0

    # When set, the API writes its log to LOG_DIR/api.log in addition to
    # stdout. The compose api service points this at a volume-mounted
    # directory so logs persist beyond `docker compose down`.
    log_dir: Path | None = None

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
            f"user={self.postgres_user} password={self.postgres_password.get_secret_value()}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
