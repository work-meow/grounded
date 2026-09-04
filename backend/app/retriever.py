"""Thin client over the Pathway DocumentStoreServer.

Pathway owns parsing, chunking, embeddings, the vector + BM25 hybrid index and
the retrieval itself. This module's only real job is tenant isolation: every
query is narrowed to one user before it leaves the process.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from rag_shared.doc_key import parse_key, user_prefix

from app.config import Settings

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


def _as_page(value: Any) -> int | None:
    """Metadata crosses JSON, so a page number can arrive as 14, "14" or 14.0."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
        return cls(
            text=hit.get("text", ""),
            # Pathway sorts ascending by `dist`; flip it so bigger = better.
            score=-float(hit.get("dist", 0.0)),
            document_id=meta.get("document_id"),
            filename=meta.get("filename"),
            page=_as_page(meta.get("page_number")),
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


async def ready_document_ids(settings: Settings, user_id: UUID) -> set[str]:
    """The user's documents that Pathway has finished indexing.

    This is the source of truth for "is it ready yet?". Nothing about indexing
    state is mirrored into Postgres, so the two can never disagree.

    Two Pathway details shape this request:

    * ``/v1/inputs`` reports *file*-level metadata taken from the connector, so
      it carries only ``path`` — the ``user_id`` the post-processor puts on each
      chunk is not there. Ownership therefore has to come from the key.
    * The endpoint filters the metadata list but zips the statuses against the
      *unfiltered* one, so asking it to filter would misalign every status.

    ponytail: hence we fetch every file and match the prefix here. Ceiling:
    O(all files in the bucket) per call, fine for a personal install. Upgrade
    path: filter server-side once Pathway aligns the two lists.
    """
    response = await _http().post(
        f"{settings.pathway_url}/v1/inputs",
        json={"return_status": True},
        timeout=settings.pathway_timeout_s,
    )
    response.raise_for_status()

    prefix = user_prefix(user_id)
    ready: set[str] = set()
    for entry in response.json():
        path = str(entry.get("path", ""))
        if not path.startswith(prefix) or entry.get("_indexing_status") != "INDEXED":
            continue
        if parsed := parse_key(path):
            ready.add(parsed["document_id"])
    return ready
