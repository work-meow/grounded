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
    jwt_issuer: str = "rag"
    # The cookie's name and the signing algorithm are constants in
    # app/security.py, not settings.
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
    # Per attempt, and it has to clear the largest upload: 64 MB over the
    # internal network is a second or two, so this is a ceiling on a wedged
    # store rather than a budget for a slow one.
    s3_timeout_s: int = 60

    # --- connected sources ----------------------------------------------------
    # Seals a connector's credentials before they reach the database or the
    # bucket (rag_shared.crypto). The same value as the indexer's SECRETS_KEY,
    # which is what opens them again. Empty disables connected sources rather
    # than storing anything in the clear.
    secrets_key: str = ""
    # Shown to the user so they know whom to share a Drive folder with. Only the
    # address: the key itself belongs to the indexer, the one process that
    # actually reads Drive.
    gdrive_service_account_email: str = ""

    # --- Pathway indexer ------------------------------------------------------
    pathway_url: str = "http://localhost:8666"
    pathway_timeout_s: float = 30.0
    retrieve_k: int = 8

    # --- agent ----------------------------------------------------------------
    openrouter_api_key: str
    # $0.25 per million tokens in, $2 out — cheaper than gemini-2.5-flash, which
    # this replaced, and it follows the citation instructions more reliably.
    agent_model: str = "openai/gpt-5-mini"
    agent_temperature: float = 0.0
    # How much the model is allowed to think before answering, for the models
    # that think. Measured on the deployment against nine questions whose
    # answers were sitting in the retrieved fragments: gpt-5-mini answered 7 of
    # 9 in 9.3 s on its own, and 9 of 9 in 4.7 s at "minimal". More reasoning
    # made it *less* accurate — it talked itself out of evidence it had been
    # handed. Set to "" for a model that does not take the parameter.
    agent_reasoning_effort: str = "minimal"
    # A hung provider must not hold an SSE stream open indefinitely. Long
    # enough for a reasoning model on a long question; short enough that a
    # dead upstream surfaces as an error rather than a page that never ends.
    agent_timeout_s: float = 120.0
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
