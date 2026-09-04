from pydantic_settings import BaseSettings, SettingsConfigDict


class IndexerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- object storage (the only input) --------------------------------------
    s3_bucket: str = "rag"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    # MinIO and most self-hosted gateways need path-style addressing.
    s3_path_style: bool = False

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
    # Embeddings are cached here, so a restart does not re-pay for every chunk.
    cache_dir: str = "./Cache"
