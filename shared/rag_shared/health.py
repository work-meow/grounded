"""How the connected sources are doing, on the wire back from the indexer.

The manifest only goes one way: the API says which sources exist, and nothing
ever came back. So a source whose token had been revoked at the far end kept
looking perfectly healthy in the UI, with documents that quietly stopped being
updated, while the indexer wrote the failure to a log nobody reads.

This is the return channel, and it is the same channel run backwards — a small
document in the bucket both processes already share. The indexer rewrites it
after every polling pass; the API reads it when it lists sources. No new port,
no new dependency, and the indexer stays what it is: a process that talks to
the bucket and to nothing else.

Two things this deliberately is not. It is not authoritative about *which*
sources exist — that is the database's job, and a source with no entry here has
simply not been looked at yet. And it never carries an exception text: those
carry hosts, URLs and occasionally the credential itself, and this document is
read by a browser. The indexer classifies the failure into a sentence first.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

HEALTH_KEY = "state/sources.json"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class SourceHealth:
    """The last thing the indexer learned about one source."""

    source_id: UUID
    ok: bool
    #: Why not, in words the person who connected it can act on. Empty when ok.
    problem: str
    #: Unix seconds of the last attempt, successful or not.
    checked_at: int
    #: Documents the last successful listing found. None before there was one.
    #: Zero is a diagnosis in itself: the folder is empty, or it is the wrong
    #: folder, or the Notion pages were never shared with the integration.
    documents: int | None


@dataclass(frozen=True, slots=True)
class Health:
    sources: dict[UUID, SourceHealth]
    #: How long an entry stays believable, in seconds. Past it the indexer is
    #: presumed gone rather than the source slow: this document is rewritten on
    #: every pass, so silence across several passes is not a slow service, it is
    #: no indexer at all — and that is a failure the UI must not report as
    #: health.
    stale_after_s: int


def dump_health(sources: list[SourceHealth], stale_after_s: int) -> bytes:
    payload = {
        "version": _VERSION,
        "stale_after_s": stale_after_s,
        "sources": [
            {
                "source_id": str(source.source_id),
                "ok": source.ok,
                "problem": source.problem,
                "checked_at": source.checked_at,
                "documents": source.documents,
            }
            for source in sorted(sources, key=lambda s: s.source_id)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode()


def load_health(raw: bytes) -> Health:
    """Parse the health document. Never raises.

    Forgiving for the same reason the manifest reader is: this is written by
    another process that may be a version ahead, and a listing of sources must
    not fail because the health of one of them could not be read. An entry that
    does not parse is one source shown as unchecked.
    """
    try:
        payload = json.loads(raw)
        entries = payload["sources"]
        stale_after_s = int(payload["stale_after_s"])
    except (ValueError, KeyError, TypeError):
        logger.warning(
            "the source health document is unreadable; nothing is known about any source"
        )
        return Health(sources={}, stale_after_s=0)

    sources: dict[UUID, SourceHealth] = {}
    for entry in entries:
        if (health := _health(entry)) is not None:
            sources[health.source_id] = health
    return Health(sources=sources, stale_after_s=stale_after_s)


def _health(entry: Any) -> SourceHealth | None:
    try:
        documents = entry["documents"]
        return SourceHealth(
            source_id=UUID(entry["source_id"]),
            ok=bool(entry["ok"]),
            problem=str(entry["problem"]),
            checked_at=int(entry["checked_at"]),
            documents=None if documents is None else int(documents),
        )
    except (ValueError, KeyError, TypeError):
        logger.warning("skipping an unreadable entry in the source health document")
        return None
