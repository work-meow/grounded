"""Connected sources, from the API's side.

The user edits the source list here; the indexer reads it from the bucket. This
module is the bridge — it checks what a connector needs, seals the credential
once, and republishes the manifest whenever the list changes.

Publishing is the only side effect outside the database, and it is deliberately
the last step of any change. A manifest naming a source the database does not
have would restart the indexer into a state nobody asked for; the reverse — a
row whose manifest entry is one write behind — costs one polling interval and
fixes itself on the next change.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from rag_shared.connectors import (
    MANIFEST_KEY,
    REQUIRED_FIELDS,
    Kind,
    SealedSource,
    dump_manifest,
)
from rag_shared.connectors import fingerprint as compute_fingerprint
from rag_shared.crypto import InvalidToken, Sealer
from rag_shared.health import HEALTH_KEY, SourceHealth, load_health
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.config import Settings
from app.models import Source

logger = logging.getLogger(__name__)

#: A credential is a token, not a document. Anything longer is either a mistake
#: or an attempt to use the source list as storage.
MAX_FIELD_LENGTH = 8192

#: Per user. Each connected source is a polling thread in the indexer and a
#: share of somebody else's API quota, so the number is bounded on purpose.
MAX_SOURCES_PER_USER = 20

_KNOWN_KINDS = {kind.value for kind in Kind}


def sealer(settings: Settings) -> Sealer | None:
    """The seal, or None when this deployment cannot keep a credential safely."""
    if not settings.secrets_key:
        return None
    try:
        return Sealer(settings.secrets_key)
    except ValueError:
        logger.error("SECRETS_KEY is not a usable key; connected sources are disabled")
        return None


def clean_config(kind: Kind, config: dict[str, str]) -> dict[str, str]:
    """Exactly the fields this kind needs, trimmed. Raises ValueError otherwise.

    Extra keys are dropped rather than refused: nothing would ever read them,
    and keeping them would let the sealed blob grow without bound.
    """
    cleaned: dict[str, str] = {}
    for field in REQUIRED_FIELDS[kind]:
        value = str(config.get(field) or "").strip()
        if not value:
            raise ValueError(f"не заполнено поле «{field}»")
        if len(value) > MAX_FIELD_LENGTH:
            raise ValueError(f"поле «{field}» слишком длинное")
        cleaned[field] = value
    return cleaned


async def backfill_fingerprints(sealer: Sealer, session: AsyncSession, user_id: UUID) -> None:
    """Give this user's older sources the fingerprint they were connected without.

    A fingerprint is what makes "this source is already connected" a question
    the database can answer, and sources predating that check have none — which
    makes them exactly the ones a user could still duplicate. It cannot be done
    in the migration: computing one means opening a sealed config, and only a
    process holding SECRETS_KEY can do that.

    A pair that is *already* a duplicate is left alone rather than fixed. There
    is no honest way to choose which of the two to keep, the unique index would
    refuse them both, and the user can see both in the list and delete one.
    """
    rows = (
        (
            await session.execute(
                select(Source).where(Source.user_id == user_id, Source.sealed_config.is_not(None))
            )
        )
        .scalars()
        .all()
    )
    taken = {row.fingerprint for row in rows if row.fingerprint}
    filled = False
    for row in rows:
        if row.fingerprint or row.kind not in _KNOWN_KINDS:
            continue
        try:
            config = sealer.unseal(row.sealed_config or "")
        except InvalidToken:
            # Sealed under a different key. Nothing here can read it, and the
            # indexer will be skipping it for the same reason.
            continue
        finger = compute_fingerprint(Kind(row.kind), config)
        if finger in taken:
            continue
        taken.add(finger)
        row.fingerprint = finger
        filled = True

    if filled:
        # Committed here rather than left to the caller's. This is a derived
        # column catching up, not part of the change being made — and the change
        # being made is quite likely to be refused *because* of what was just
        # filled in, which would roll the fill back and leave the unique index
        # with nothing to enforce.
        await session.commit()


async def duplicate_name(session: AsyncSession, user_id: UUID, finger: str) -> str | None:
    """The name of the source already reading this exact place, if there is one."""
    return await session.scalar(
        select(Source.name).where(Source.user_id == user_id, Source.fingerprint == finger)
    )


async def publish(settings: Settings, session: AsyncSession) -> None:
    """Rewrite the manifest from the database. This restarts the indexer.

    Every user's sources, not just the one who made the change: there is one
    index and one manifest describing all of its inputs.

    The sealed blob is copied through rather than opened and re-sealed. Fernet
    picks a fresh IV each time, so re-sealing would make an unchanged list write
    different bytes and restart the indexer for nothing.
    """
    rows = (
        (await session.execute(select(Source).where(Source.sealed_config.is_not(None))))
        .scalars()
        .all()
    )
    manifest = dump_manifest(
        [
            SealedSource(
                source_id=row.id,
                user_id=row.user_id,
                kind=Kind(row.kind),
                name=row.name,
                sealed_config=row.sealed_config or "",
            )
            for row in rows
            if row.kind in _KNOWN_KINDS
        ]
    )
    await storage.put(settings, MANIFEST_KEY, manifest, "application/json")


class Status(StrEnum):
    """How a connected source is doing, as far as anyone here can tell."""

    OK = "ok"
    ERROR = "error"
    #: Nothing has been heard about it. Normal for the first minute of a
    #: source's life — the indexer restarts to pick it up, then polls — and it
    #: is deliberately not called "ok".
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceStatus:
    status: Status
    #: Why, when something is wrong. Empty otherwise.
    problem: str
    #: When the indexer last looked. None when it never has.
    checked_at: datetime | None
    #: Entries its last successful listing returned. None when there was none.
    documents: int | None


UNCHECKED = SourceStatus(status=Status.UNKNOWN, problem="", checked_at=None, documents=None)

_NO_INDEXER = "индексатор давно не отвечает: источник может быть не в актуальном состоянии"

#: A budget for reading it. This is a kilobyte from a store one hop away, on the
#: path of a page that has to render either way — an unknown status is a worse
#: answer than a fresh one and a far better answer than a spinner.
_TIMEOUT_S = 3.0


async def statuses(settings: Settings) -> dict[UUID, SourceStatus]:
    """What the indexer last said about each source.

    Empty when it has never said anything, which is also what a missing bucket
    or an unreadable document comes back as — every source is then reported as
    unchecked, which is the honest answer and the one the UI can explain.
    """
    try:
        async with asyncio.timeout(_TIMEOUT_S):
            raw = await storage.get(settings, HEALTH_KEY)
    except TimeoutError:
        logger.warning("the source health document did not arrive in time")
        return {}
    if raw is None:
        return {}
    report = load_health(raw)
    now = time.time()
    return {
        source_id: _status(entry, now, report.stale_after_s)
        for source_id, entry in report.sources.items()
    }


def _status(entry: SourceHealth, now: float, stale_after_s: int) -> SourceStatus:
    checked_at = datetime.fromtimestamp(entry.checked_at, tz=UTC)
    # Staleness first, and it overrides a healthy report. The indexer rewrites
    # the document after every pass, so an old "everything is fine" is not
    # evidence that everything is fine — it is evidence that nothing has looked
    # since, which is exactly the case this whole channel exists to stop the UI
    # from painting green.
    if stale_after_s and now - entry.checked_at > stale_after_s:
        return SourceStatus(Status.ERROR, _NO_INDEXER, checked_at, entry.documents)
    if not entry.ok:
        return SourceStatus(Status.ERROR, entry.problem, checked_at, entry.documents)
    return SourceStatus(Status.OK, "", checked_at, entry.documents)
