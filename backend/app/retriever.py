"""Thin client over the Pathway DocumentStoreServer.

Pathway owns parsing, chunking, embeddings, the vector + BM25 hybrid index and
the retrieval itself. This module's only real job is tenant isolation: every
query is narrowed to one user before it leaves the process.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx

from app.config import Settings
from shared.doc_key import parse_key

_client: httpx.AsyncClient | None = None


def set_client(client: httpx.AsyncClient | None) -> None:
    """Wired up by the app lifespan so connections are pooled."""
    global _client
    _client = client


def _http() -> httpx.AsyncClient:
    if _client is None:  # pragma: no cover - misconfiguration, not a runtime path
        raise RuntimeError("retriever client is not initialised")
    return _client


def _tenant_filter(user_id: UUID, document_id: UUID | None = None) -> str:
    """A JMESPath filter pinned to one user.

    Both ids are ``UUID`` instances, so their string form is hex-and-dashes
    only and cannot break out of the quotes. That typing *is* the injection
    guard — do not loosen it to ``str``.
    """
    clause = f"user_id == '{user_id}'"
    if document_id is not None:
        clause += f" && document_id == '{document_id}'"
    return clause


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    score: float
    document_id: str | None
    filename: str | None
    page: int | None

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> "Chunk":
        meta = hit.get("metadata") or {}
        page = meta.get("page_number")
        return cls(
            text=hit.get("text", ""),
            # Pathway sorts ascending by `dist`; flip it so bigger = better.
            score=-float(hit.get("dist", 0.0)),
            document_id=meta.get("document_id"),
            filename=meta.get("filename"),
            page=int(page) if isinstance(page, (int, float, str)) and str(page).isdigit() else None,
        )


async def retrieve(
    settings: Settings,
    user_id: UUID,
    query: str,
    k: int,
    document_id: UUID | None = None,
) -> list[Chunk]:
    response = await _http().post(
        f"{settings.pathway_url}/v1/retrieve",
        json={
            "query": query,
            "k": k,
            "metadata_filter": _tenant_filter(user_id, document_id),
        },
        timeout=settings.pathway_timeout_s,
    )
    response.raise_for_status()
    return [Chunk.from_hit(hit) for hit in response.json()]


async def indexed_document_ids(settings: Settings, user_id: UUID) -> dict[str, int]:
    """document_id -> number of indexed chunks, for the user's files.

    This is the source of truth for "is it ready yet?". Nothing about indexing
    state is mirrored into Postgres, so the two can never disagree.
    """
    response = await _http().post(
        f"{settings.pathway_url}/v1/inputs",
        json={"metadata_filter": _tenant_filter(user_id)},
        timeout=settings.pathway_timeout_s,
    )
    response.raise_for_status()

    counts: dict[str, int] = {}
    for entry in response.json():
        # /v1/inputs returns one record per indexed chunk's source document;
        # fall back to the raw path when the post-processor metadata is absent.
        document_id = entry.get("document_id")
        if document_id is None and (path := entry.get("path")):
            parsed = parse_key(str(path))
            document_id = parsed["document_id"] if parsed else None
        if document_id:
            counts[document_id] = counts.get(document_id, 0) + 1
    return counts
