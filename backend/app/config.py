from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- application database -------------------------------------------------
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # --- auth -----------------------------------------------------------------
    # Tokens are minted out of band with `uv run rag-token <user-id>` and pasted
    # into the login screen. HS256 keeps key management to a single secret.
    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "rag"
    session_cookie: str = "rag_session"
    cookie_secure: bool = True

    # --- object storage (originals) -------------------------------------------
    s3_bucket: str = "rag"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None  # set for MinIO / R2 / Backblaze
    # Where the *browser* reaches the same bucket. Presigned links are signed
    # for this host; uploads and deletes still go through the internal one, so
    # they neither leave the box nor wait on a TLS certificate.
    s3_public_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    # MinIO and most self-hosted gateways need bucket-in-path addressing.
    s3_path_style: bool = False
    max_upload_bytes: int = 64 * 1024 * 1024

    # --- Pathway indexer ------------------------------------------------------
    pathway_url: str = "http://localhost:8666"
    pathway_timeout_s: float = 30.0
    retrieve_k: int = 8

    # --- agent ----------------------------------------------------------------
    openrouter_api_key: str
    agent_model: str = "anthropic/claude-sonnet-4-5"
    agent_temperature: float = 0.0
    # Hard ceiling so a confused agent cannot run away with the bill.
    max_tool_calls_per_run: int = 5
    max_model_calls_per_run: int = 8
    # How much chat history is replayed into the agent each turn.
    history_window: int = 20

    # --- http -----------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
