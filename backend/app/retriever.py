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


# A pooled connection the indexer has already closed fails on first use, before
# the request is written: httpx reports RemoteProtocolError, and the caller sees
# a question that simply did not work. Nothing is retried by default, so this is
# where an idle stack quietly loses its first request after a pause — observed
# in production as "Server disconnected without sending a response" from
# search_knowledge, and as documents flipping back to "processing" for one poll.
#
# Retrying a POST is normally unsafe. These two are read-only queries that carry
# their arguments in a body because they are too big for a URL, so a second
# attempt cannot duplicate anything. It runs on a connection guaranteed to be
# fresh, which is exactly what the failure asks for.
_STALE_CONNECTION = (httpx.RemoteProtocolError, httpx.ConnectError)


async def _post(settings: Settings, path: str, payload: dict[str, Any]) -> Any:
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            response = await _http().post(
                f"{settings.pathway_url}{path}",
                json=payload,
                timeout=settings.pathway_timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except _STALE_CONNECTION:
            if attempt == attempts:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def _tenant_filter(user_id: UUID, document_id: UUID | None = None) -> str:
    """A JMESPath filter pinned to one user.

    Backticks, not quotes. Pathway rewrites the expression before handing it to
    the engine (see document_store._get_jmespath_filter): a backtick becomes a
    single quote, a single quote is escaped to \\' and a double quote is
    dropped. Written with quotes, `user_id == 'x'` reaches the parser as
    `user_id == \\'x\\'` — which does not parse, and takes the whole indexer
    process down with it rather than returning an error.

    Both ids are ``UUID`` instances, so their string form is hex-and-dashes
    only and carries no character that rewrite could turn into syntax. That
    typing *is* the injection guard — do not loosen it to ``str``.
    """
    clause = f"user_id == `{user_id}`"
    if document_id is not None:
        clause += f" && document_id == `{document_id}`"
    return clause


def _as_int(value: Any) -> int | None:
    """Metadata crosses JSON, so a number can arrive as 14, "14" or 14.0."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """One document as the index knows it.

    Deliberately not the same thing as a :class:`app.models.Document` row: that
    table records files this API wrote to the bucket, and a connected source
    produces documents it never touched.
    """

    document_id: str
    source_id: str
    filename: str
    #: Where a person opens the original. None for uploads, which get a
    #: presigned link to the bucket instead.
    web_url: str | None
    size_bytes: int | None
    #: When the far end last changed it. Stands in for "created" in the file
    #: list, which is the only date that means anything for a file we mirror.
    modified_at: int
    ready: bool


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
            page=_as_int(meta.get("page_number")),
        )


async def retrieve(
    settings: Settings,
    user_id: UUID,
    query: str,
    k: int,
    document_id: UUID | None = None,
) -> list[Chunk]:
    hits = await _post(
        settings,
        "/v1/retrieve",
        {"query": query, "k": k, "metadata_filter": _tenant_filter(user_id, document_id)},
    )
    return [Chunk.from_hit(hit) for hit in hits]


async def indexed_documents(settings: Settings, user_id: UUID) -> list[IndexedDocument]:
    """Every document of this user's that Pathway holds, uploaded or connected.

    This is the source of truth for what is searchable. Nothing about indexing
    state is mirrored into Postgres, so the two can never disagree — and for a
    connected source there is no Postgres row at all: nothing here ever fetched
    those files, so the index is the only place that has seen them.

    Two Pathway details shape the request:

    * ``/v1/inputs`` reports *file*-level metadata straight from the connector,
      so it carries the ``path`` but not the ``user_id`` the post-processor puts
      on each chunk. Ownership therefore has to come out of the key.
    * The endpoint filters the metadata list but zips the statuses against the
      *unfiltered* one, so asking it to filter would misalign every status.

    ponytail: hence we fetch every file and match the prefix here. Ceiling:
    O(all documents in the index) per call, fine for a personal install.
    Upgrade path: filter server-side once Pathway aligns the two lists.
    """
    entries = await _post(settings, "/v1/inputs", {"return_status": True})

    prefix = user_prefix(user_id)
    documents: list[IndexedDocument] = []
    for entry in entries:
        path = str(entry.get("path", ""))
        if not path.startswith(prefix):
            continue
        if (parsed := parse_key(path)) is None:
            continue
        # The key layout matches hex and dashes, which is nearly but not quite
        # "a UUID". Anything else is not a document this system created, and the
        # caller turns these into UUIDs — where one bad object in the bucket
        # would otherwise take out the whole file list.
        if not _is_uuid(parsed["document_id"]) or not _is_uuid(parsed["source_id"]):
            continue
        documents.append(
            IndexedDocument(
                document_id=parsed["document_id"],
                source_id=parsed["source_id"],
                filename=parsed["filename"],
                web_url=entry.get("web_url") or None,
                size_bytes=_as_int(entry.get("size")),
                modified_at=_as_int(entry.get("modified_at")) or 0,
                ready=entry.get("_indexing_status") == "INDEXED",
            )
        )
    return documents
