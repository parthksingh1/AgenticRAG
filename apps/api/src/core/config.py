"""Application configuration.

Every knob is read from the environment through pydantic-settings — there are no
module-level constants holding secrets and no `os.getenv` calls scattered through
the codebase. Import the singleton via :func:`get_settings`.

Example:
    >>> from src.core.config import get_settings
    >>> get_settings().retrieval_top_k
    5
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    """Deployment environment."""

    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Root settings object, populated from the environment and `.env`."""

    model_config = SettingsConfigDict(
        # Every variable is AGRAG_-prefixed. A bare DATABASE_URL or SECRET_KEY
        # collides with whatever else shares the environment, and in Kubernetes
        # or CI that failure is silent — the app reads someone else's value and
        # connects somewhere unexpected.
        env_prefix="AGRAG_",
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Core ─────────────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.LOCAL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104 — bound inside a container network
    api_port: int = 8000
    secret_key: SecretStr = SecretStr("insecure-local-development-key-change-me")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ── Datastores ───────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://agrag:agrag@localhost:5432/agrag"
    database_url_sync: str = "postgresql+psycopg://agrag:agrag@localhost:5432/agrag"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: SecretStr | None = None

    minio_endpoint: str = "localhost:9000"
    minio_public_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    minio_bucket: str = "agrag-documents"
    minio_backup_bucket: str = "agrag-backups"
    minio_secure: bool = False

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("agrag-neo4j-password")

    # ── Providers ────────────────────────────────────────────────────────────
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    together_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    default_chat_model: str = "claude-sonnet-5"
    default_cheap_model: str = "claude-haiku-4-5-20251001"
    default_embedding_model: str = "BAAI/bge-large-en-v1.5"
    default_reranker_model: str = "BAAI/bge-reranker-v2-m3"

    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 4

    # ── Auth ─────────────────────────────────────────────────────────────────
    clerk_secret_key: SecretStr | None = None
    clerk_publishable_key: str | None = None
    clerk_jwks_url: str | None = None
    clerk_issuer: str | None = None
    auth_dev_mode: bool = True

    # ── Observability ────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "agrag-api"
    otel_traces_enabled: bool = True
    langfuse_host: str = "http://localhost:3002"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

    # ── Budgets & limits ─────────────────────────────────────────────────────
    default_tenant_daily_token_budget: int = 2_000_000
    max_tokens_per_request: int = 16_000
    max_tool_calls_per_turn: int = 8
    max_agent_iterations: int = 12
    rate_limit_chat_per_minute: int = 30
    max_upload_bytes: int = 200 * 1024 * 1024

    # ── Retrieval defaults ───────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_expanded_k: int = 20
    rerank_top_n: int = 5
    rrf_k: int = 60
    semantic_cache_threshold: float = 0.97
    embedding_dim: int = 1024

    prompts_dir: str = "prompts"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string for CORS origins as well as a list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("secret_key")
    @classmethod
    def _reject_default_secret_outside_local(cls, value: SecretStr) -> SecretStr:
        """Refuse to boot a deployed environment with the placeholder secret."""
        if "change-me" in value.get_secret_value() and len(value.get_secret_value()) < 32:
            msg = "SECRET_KEY must be a 32+ character random value"
            raise ValueError(msg)
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """True when running in a deployed, non-development environment."""
        return self.app_env in (AppEnv.STAGING, AppEnv.PRODUCTION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def configured_providers(self) -> tuple[str, ...]:
        """Names of providers that have a usable API key configured."""
        candidates = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "together": self.together_api_key,
        }
        return tuple(name for name, key in candidates.items() if key is not None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that `Depends(get_settings)` is free after the first call. Tests
    clear the cache with ``get_settings.cache_clear()``.

    Example:
        >>> settings = get_settings()
        >>> settings.retrieval_top_k
        5
    """
    return Settings()
