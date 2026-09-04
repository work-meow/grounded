from pydantic_settings import BaseSettings, SettingsConfigDict


class IndexerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- object storage (uploads, and the connector manifest) -----------------
    s3_bucket: str = "rag"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    # MinIO and most self-hosted gateways need path-style addressing.
    s3_path_style: bool = False

    # --- connected sources ----------------------------------------------------
    # Opens the credentials in the manifest the API writes. Same value as the
    # API's SECRETS_KEY; without it connected sources are disabled and only
    # uploads are indexed.
    secrets_key: str = ""
    # The service account key, as JSON rather than a path, so it lives in .env
    # with every other secret instead of needing a mounted file. Users share a
    # Drive folder with this account's address.
    gdrive_credentials_json: str = ""
    # How often a connected source is re-read. Ten minutes is a deliberate
    # trade: every poll costs a listing request per source, and an edit that
    # shows up within ten minutes reads as live to a person.
    refresh_interval_s: int = 600
    # How often the manifest's ETag is checked. Only a HEAD against our own
    # MinIO, so it can be frequent — this is the delay between adding a source
    # in the UI and the index restarting to include it.
    manifest_poll_s: int = 30
    # A ceiling on one document, matching the API's upload limit so that a file
    # arriving through a connector cannot do what an upload is not allowed to.
    max_document_bytes: int = 64 * 1024 * 1024

    # --- embeddings via OpenRouter --------------------------------------------
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"

    # --- chunking -------------------------------------------------------------
    # Tokens, not characters (the splitter is given a tiktoken encoding). 500
    # keeps every chunk far below the embedder's 8k limit, so the truncation
    # path is never reached.
    chunk_size: int = 500
    # Overlap carries a sentence across the seam, so an answer that straddles a
    # boundary is still retrievable from either side.
    chunk_overlap: int = 60

    # --- server ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8666
    log_level: str = "INFO"
    # Embeddings are cached here, keyed on chunk content, so a restart does not
    # re-pay for the corpus (see the cache_strategy in pipeline.py — without it
    # this directory holds only Pathway's own persistence and buys nothing).
    cache_dir: str = "./Cache"
