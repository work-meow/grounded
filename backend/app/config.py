from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
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
    # Hard ceilings so a confused agent cannot run away with the bill. The
    # per-tool ones are what make a retry loop safe to encourage: the agent is
    # told to search again with different wording when the first fragments do
    # not answer, and these are what stop that being an invitation to loop.
    max_knowledge_searches_per_run: int = 3
    max_web_searches_per_run: int = 2
    max_tool_calls_per_run: int = 6
    max_model_calls_per_run: int = 8
    max_document_reads_per_run: int = 3
    # A provider that errors used to cost the whole turn: the reader saw an
    # error event and the API a 502. Two retries with backoff cover the blip
    # that a retry fixes; the fallback covers the model being down rather than
    # slow, and answers from the documents on something cheaper instead of not
    # at all. Empty disables the fallback.
    #
    # Retries spend the same budget as max_model_calls_per_run, deliberately:
    # the ceiling is on what one turn may cost, and a turn that spent its
    # calls on retries has still spent them.
    agent_retries: int = 2
    agent_fallback_model: str = "google/gemini-2.5-flash"
    # History is replayed in full up to this many tokens, and summarised above
    # it — twenty long messages otherwise arrive as most of a context window,
    # and the oldest of them is the least likely to matter.
    history_summarise_above_tokens: int = 12000
    # How much chat history is replayed into the agent each turn.
    history_window: int = 20
    # Whose today. A model knows nothing about the current date, and a personal
    # knowledge base is full of questions that only mean anything relative to
    # it, so every turn is told what day it is — in this zone. UTC is a safe
    # default and a wrong one for most people; set it to where you are.
    timezone: str = "UTC"

    # --- relevance ------------------------------------------------------------
    # The index answers every query with its k best fragments whether or not any
    # of them is about the question. A second pass reads the candidates and says
    # which ones bear on it — measured on this deployment at 0.3–0.7 s and about
    # $0.00017 a call, against a baseline where three quarters of what the model
    # read was about something else.
    rerank_enabled: bool = True
    rerank_model: str = "google/gemini-2.5-flash-lite"
    # Retrieved before judging. More candidates cost nothing extra to judge —
    # it is one call either way — and give the judge more to find the answer in.
    rerank_candidates: int = 20
    # Kept after. A ceiling, not a target: the honest answer is often fewer, and
    # sometimes none.
    rerank_keep: int = 6
    rerank_timeout_s: float = 20.0
    # Fragments from any one document on the search page. For browsing only:
    # answering a question about a long document legitimately takes several
    # fragments of it, and a list of places to look wants the opposite.
    max_chunks_per_document: int = 2

    # --- web search (only when the user turns it on for a message) ------------
    # Called directly rather than through LangChain: the web plugin is an
    # OpenRouter extension to the request body, not something a chat model
    # wrapper exposes. Hence a base url here, which nothing else needed.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # A separate, cheaper model does the searching and relays what it found;
    # the agent above it writes the answer.
    web_search_model: str = "google/gemini-2.5-flash-lite"
    # Which search back end OpenRouter should use. Measured on one query:
    # parallel/turbo $0.00125 a call and 2.2 s, exa $0.0072 and 3.3 s with less
    # text per source. Gemini has no native search on OpenRouter at all — that
    # request is refused outright — so an external engine is the only option.
    web_search_engine: str = "parallel"
    # Parallel charges by request, not by result, so asking for more costs
    # nothing and raises the odds that at least one page has real text on it
    # rather than a navigation menu.
    web_search_results: int = 6
    # Its own budget, inside the turn's. A slow search must not eat the whole
    # answer: the agent can still reply from the knowledge base without it.
    web_search_timeout_s: float = 30.0

    # --- http -----------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("timezone")
    @classmethod
    def _zone_exists(cls, value: str) -> str:
        """Fail at startup, not on the first question of the day.

        A typo here would otherwise raise inside the agent, halfway through
        building a prompt, and read as the model being broken.
        """
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
