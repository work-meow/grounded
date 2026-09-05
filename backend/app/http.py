"""The one outbound HTTP client this process uses.

Two things sit on the other end — the indexer's retrieval endpoints and
OpenRouter — and both are called often enough that a pooled connection is the
difference between a request and a fresh TLS handshake on every question. One
client serves both: httpx clients are safe to use concurrently, and what
actually differs between the callers is the per-request timeout, which is passed
per request.

It is created and disposed of by the application lifespan, so nothing here has
to guess when to close it.
"""

import httpx

_client: httpx.AsyncClient | None = None


def set_client(client: httpx.AsyncClient | None) -> None:
    """Wired up by the app lifespan so connections are pooled."""
    global _client
    _client = client


def client() -> httpx.AsyncClient:
    if _client is None:  # pragma: no cover - misconfiguration, not a runtime path
        raise RuntimeError("the outbound HTTP client is not initialised")
    return _client
