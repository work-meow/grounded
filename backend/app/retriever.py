"""Thin client over the Pathway DocumentStoreServer.

Pathway owns parsing, chunking, embeddings, the vector + BM25 hybrid index and
the retrieval itself. This module's only real job is tenant isolation: every
query is narrowed to one user before it leaves the process.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from rag_shared.doc_key import parse_key, user_prefix

from app import http
from app.config import Settings

logger = logging.getLogger(__name__)

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

#: Waits between attempts, and the reason there are waits at all. The two
#: attempts used to run back to back, so a window of even a few hundred
#: milliseconds swallowed both — observed in production right after the
#: indexer's container had been recreated: "All connection attempts failed"
#: twice in a row, and the turn died. A moved container needs a moment, not a
#: second try in the same instant.
#:
#: Deliberately short and deliberately not enough to cover a restart, which
#: takes six or seven seconds: that is what the search page's "индекс
#: перестраивается" exists to say, and holding a request open through it would
#: be worse than saying so.
_BACKOFF_S = (0.2, 0.6)


async def _post(settings: Settings, path: str, payload: dict[str, Any]) -> Any:
    for attempt, wait in enumerate((*_BACKOFF_S, None)):
        try:
            response = await http.client().post(
                f"{settings.pathway_url}{path}",
                json=payload,
                timeout=settings.pathway_timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except _STALE_CONNECTION:
            if wait is None:
                raise
            logger.info(
                "the index did not answer (attempt %d); retrying in %.1fs", attempt + 1, wait
            )
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


#: What a query may contain. Everything else becomes a space.
#:
#: An allowlist, and that is the whole point. A query the engine cannot parse
#: does not come back as an error — it **panics the engine thread**, Pathway
#: exits, and the indexer goes down for everybody until it restarts. Which
#: characters do that is not something to guess at: the first attempt here was
#: a blocklist of Tantivy's query syntax, it missed the backtick, and the
#: crash survived it.
#:
#: The backtick is Pathway's own literal delimiter — the same syntax the tenant
#: filter is built with (``user_id == `uuid` ``) — so the string is parsed as an
#: expression before it is ever a search. Proven by narrowing it down on the
#: deployment: "# тест" answers, "тест ``` тест" takes the process down, and
#: plain words answer. Whatever else that parser owns is now moot.
#:
#: Kept: letters, digits, underscores, whitespace, and the punctuation that
#: carries meaning inside a search term — a decimal comma, a percent, a dot in
#: a filename. Everything else is dropped, and a search loses nothing by it.
_ALLOWED = re.compile(r"[^\w\s.,%№]+", re.UNICODE)


def searchable(query: str) -> str:
    """A query the index is guaranteed to be able to parse.

    An allowlist rather than a blocklist, because the failure is a crash and
    not a bad result: being wrong about one character in a blocklist means the
    whole index goes down, and being wrong in an allowlist means a query is
    slightly coarser than it could have been.

    Anything a person, a fragment or the model typed goes through here.
    """
    return " ".join(_ALLOWED.sub(" ", query).split())


def _tenant_filter(
    user_id: UUID, document_id: UUID | None = None, source_id: UUID | None = None
) -> str:
    """A JMESPath filter pinned to one user.

    Backticks, not quotes. Pathway rewrites the expression before handing it to
    the engine (see document_store._get_jmespath_filter): a backtick becomes a
    single quote, a single quote is escaped to \\' and a double quote is
    dropped. Written with quotes, `user_id == 'x'` reaches the parser as
    `user_id == \\'x\\'` — which does not parse, and takes the whole indexer
    process down with it rather than returning an error.

    All three ids are ``UUID`` instances, so their string form is hex-and-dashes
    only and carries no character that rewrite could turn into syntax. That
    typing *is* the injection guard — do not loosen it to ``str``.

    Dates are deliberately not here. A number written as `` `1700000000` ``
    survives the same rewrite as the string ``'1700000000'``, and JMESPath
    comparing a number to a string evaluates to null — so the filter would
    quietly match nothing. Recency is applied to the results instead, where it
    is arithmetic rather than a rewritten expression.
    """
    clause = f"user_id == `{user_id}`"
    if document_id is not None:
        clause += f" && document_id == `{document_id}`"
    if source_id is not None:
        clause += f" && source_id == `{source_id}`"
    return clause


def _as_int(value: Any) -> int | None:
    """Metadata crosses JSON, so a number can arrive as 14, "14" or 14.0."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


#: The only schemes a link handed to the browser may use. Both a connector's
#: `web_url` and the url of a web search result are strings from somebody
#: else's API that this API passes to the browser, which opens them with
#: window.open — where a `javascript:` URL runs in a document that inherits
#: our origin. None of those services would send one; a trust boundary is not
#: the place to rely on that.
_OPENABLE = ("https://", "http://")


def openable(value: Any) -> str | None:
    """One link, or None if it is not one we may hand to a browser."""
    url = str(value or "").strip()
    return url if url.startswith(_OPENABLE) else None


def _as_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


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
    #: presigned link to the bucket instead, and None for an object store, which
    #: has no page at all — that one is signed on demand from ``external_id``.
    web_url: str | None
    #: What the far end calls this file: a Notion page id, a Drive file id, an
    #: object key. Carried so a source with nothing to link to can still be
    #: linked to.
    external_id: str | None
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
    #: When the far end last changed the document this came from. Zero when the
    #: metadata did not say, which is treated as "old" — a fragment with no date
    #: is not evidence of being recent.
    modified_at: int = 0

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
            modified_at=_as_int(meta.get("modified_at")) or 0,
        )


def since(days: int | None) -> int | None:
    """The instant a "last N days" question reaches back to.

    Nonsense is ignored rather than refused: a caller that passes days=0 or a
    negative meant "no limit", and turning that into an empty result would be a
    worse answer than searching everything.

    ponytail: applied to the fragments after retrieval, not inside the filter —
    see :func:`_tenant_filter` for why a number cannot go into that expression.
    Ceiling: a recency question asks for the k best overall and keeps the recent
    ones, rather than the k best among the recent ones, so on a large corpus it
    could come back thin. Upgrade path: push it into the filter once a numeric
    literal survives Pathway's rewriting.
    """
    if days is None or days <= 0:
        return None
    return int(time.time()) - days * 86_400


async def retrieve(
    settings: Settings,
    user_id: UUID,
    query: str,
    k: int,
    document_id: UUID | None = None,
    source_id: UUID | None = None,
    since: int | None = None,
) -> list[Chunk]:
    """The k best fragments for this query, narrowed if asked.

    Document and source are pushed down to the index, so the k that come back
    are k from inside the narrowing rather than k from everywhere with most of
    them then dropped. Recency is applied here, for the reason in
    :func:`_tenant_filter`.
    """
    hits = await _post(
        settings,
        "/v1/retrieve",
        {
            # Never the raw string: see searchable().
            "query": searchable(query),
            "k": k,
            "metadata_filter": _tenant_filter(user_id, document_id, source_id),
        },
    )
    chunks = [Chunk.from_hit(hit) for hit in hits]
    if since is None:
        return chunks
    return [chunk for chunk in chunks if chunk.modified_at >= since]


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
                web_url=openable(entry.get("web_url")),
                external_id=_as_text(entry.get("external_id")),
                size_bytes=_as_int(entry.get("size")),
                modified_at=_as_int(entry.get("modified_at")) or 0,
                ready=entry.get("_indexing_status") == "INDEXED",
            )
        )
    return documents
