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

import logging

from rag_shared.connectors import (
    MANIFEST_KEY,
    REQUIRED_FIELDS,
    Kind,
    SealedSource,
    dump_manifest,
)
from rag_shared.crypto import Sealer
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
