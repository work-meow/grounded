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

    # --- context on each chunk ------------------------------------------------
    # A chunk from the middle of a policy reading only "28 календарных дней"
    # does not say it is about leave, so it is findable by meaning and by luck.
    # The document's name is already prepended, but only to the first chunk of
    # each part — putting it on all of them would dilute every embedding. This
    # instead asks a cheap model what each chunk is part of and puts that
    # sentence in front of it, which lands in both the vector and the BM25
    # index.
    #
    # Measured on 35 real questions, one variable moved: unsupported claims in
    # answers fell from 25 to 11 and fully grounded answers rose from 10 of 15
    # to 12 of 15, with no change in recall and none in latency. The reason is
    # visible in the fragments: "28 календарных дней" with a line saying it is
    # from the leave section gives the model the frame it was inventing.
    #
    # Recommended, and still off by default: turning it on changes what is
    # indexed, so an existing install re-embeds its whole corpus on the next
    # start. Same class of change as editing chunk_size or the embedding model.
    # Turn it on together with a fresh CACHE_DIR so the old one stays for a
    # rollback.
    contextual_chunks: bool = False
    context_model: str = "google/gemini-2.5-flash-lite"
    # Chunks described in one call. Parts here are usually a page or a short
    # file — one or two chunks — so batching means one call per part instead of
    # one per chunk, and the document's text is paid for once rather than N
    # times.
    context_batch: int = 8
    # How many parts may be described at once. Indexing is not on anybody's
    # critical path, and a restart re-reads everything, so this is politeness
    # to the provider rather than a latency budget.
    context_concurrency: int = 4
    # Beyond this many chunks a part is left alone. A four-thousand-page
    # document would otherwise be several hundred calls before anybody noticed.
    context_max_chunks: int = 40
    # How much of the part the model reads as context. A page of PDF is well
    # under this; a long markdown file is truncated, and its beginning is the
    # part that says what the document is.
    context_document_chars: int = 6000
    context_timeout_s: float = 30.0

    # --- server ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8666
    log_level: str = "INFO"
    # Embeddings are cached here, keyed on chunk content, so a restart does not
    # re-pay for the corpus (see the cache_strategy in pipeline.py — without it
    # this directory holds only Pathway's own persistence and buys nothing).
    cache_dir: str = "./Cache"
