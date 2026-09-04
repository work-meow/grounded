"""Connected sources, on the wire between the API and the indexer.

The API owns the source list: it is what the user edits. The indexer owns the
index, and a Pathway dataflow graph is fixed the moment it is built, so the
indexer cannot simply be *told* about a new source over HTTP — the graph has to
be rebuilt from scratch.

So the two agree on a file instead. The API writes this manifest into the bucket
they already share; the indexer reads it at startup to build its connectors, and
watches it while it runs so that adding a source restarts the process.

Every credential in the manifest is sealed (see :mod:`rag_shared.crypto`); the
plaintext exists only inside the two processes that hold ``SECRETS_KEY``.
"""

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from rag_shared.crypto import InvalidToken, Sealer

logger = logging.getLogger(__name__)

MANIFEST_KEY = "config/connectors.json"
_VERSION = 1


class Kind(StrEnum):
    """The connectors a user can add. Uploaded files are not one of these.

    Uploads are always on, need no configuration and have no credential, so they
    are wired in unconditionally rather than described by a manifest entry.
    """

    GDRIVE = "gdrive"
    NOTION = "notion"
    YANDEX = "yandex"
    DROPBOX = "dropbox"
    ONEDRIVE = "onedrive"


#: What each kind needs in its sealed config, and nothing optional.
#:
#: The API validates against this before it writes, and the indexer checks again
#: before it builds: a manifest that has lost a field must cost one source, not
#: the whole index.
#:
#: Dropbox and OneDrive carry the application's own identity alongside the
#: refresh token because their access tokens expire hourly — the connector
#: trades the three for a fresh one, so nothing here goes stale.
REQUIRED_FIELDS: dict[Kind, tuple[str, ...]] = {
    Kind.GDRIVE: ("folder_id",),
    Kind.NOTION: ("token",),
    Kind.YANDEX: ("token", "path"),
    Kind.DROPBOX: ("app_key", "app_secret", "refresh_token", "path"),
    Kind.ONEDRIVE: ("client_id", "client_secret", "refresh_token", "path"),
}


@dataclass(frozen=True, slots=True)
class _Source:
    source_id: UUID
    user_id: UUID
    kind: Kind
    name: str


@dataclass(frozen=True, slots=True)
class SealedSource(_Source):
    """A source as it travels: the credential still sealed.

    What the API holds. It seals a config once, when the user submits it, and
    from then on only ever moves the blob around — so the process that writes
    the manifest never needs the plaintext, and the manifest for an unchanged
    list is byte-identical every time it is written.
    """

    sealed_config: str


@dataclass(frozen=True, slots=True)
class ConnectorSpec(_Source):
    """A source as it is used: the credential in the clear.

    What the indexer holds, and only after opening the seal.
    """

    config: dict[str, Any]

    def missing_fields(self) -> tuple[str, ...]:
        return tuple(f for f in REQUIRED_FIELDS[self.kind] if not self.config.get(f))


def dump_manifest(sources: list[SealedSource]) -> bytes:
    """Serialise the source list for the bucket.

    Sorted by source id, because the indexer restarts on a changed ETag and a
    list that serialises differently on each write would restart it each time.
    """
    payload = {
        "version": _VERSION,
        "sources": [
            {
                "source_id": str(source.source_id),
                "user_id": str(source.user_id),
                "kind": source.kind.value,
                "name": source.name,
                "sealed_config": source.sealed_config,
            }
            for source in sorted(sources, key=lambda s: s.source_id)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode()


def load_manifest(raw: bytes, sealer: Sealer) -> list[ConnectorSpec]:
    """Parse the manifest, dropping entries this build cannot use.

    Deliberately forgiving. This runs at indexer startup, and the manifest is
    written by another process that may be a version ahead: an entry naming a
    connector this build does not have, or one whose seal does not open, must
    cost that one source. Raising here would leave the user with no index at all
    — including the uploads, which have nothing to do with the bad entry.
    """
    try:
        payload = json.loads(raw)
        entries = payload["sources"]
    except (ValueError, KeyError, TypeError):
        logger.exception("connector manifest is unreadable; continuing with uploads only")
        return []

    specs: list[ConnectorSpec] = []
    for entry in entries:
        if (spec := _spec(entry, sealer)) is not None:
            specs.append(spec)
    return specs


def _spec(entry: Any, sealer: Sealer) -> ConnectorSpec | None:
    """One manifest entry, or None with the reason logged — never the credential."""
    try:
        kind = Kind(entry["kind"])
        spec = ConnectorSpec(
            source_id=UUID(entry["source_id"]),
            user_id=UUID(entry["user_id"]),
            kind=kind,
            name=str(entry["name"]),
            config=sealer.unseal(entry["sealed_config"]),
        )
    except ValueError as exc:  # unknown kind, malformed uuid
        logger.error("skipping a manifest entry: %s", exc)
        return None
    except (KeyError, TypeError):
        logger.error("skipping a manifest entry: it is missing required keys")
        return None
    except InvalidToken:
        logger.error(
            "skipping source %s: its credential does not open with this SECRETS_KEY",
            entry.get("source_id"),
        )
        return None

    if missing := spec.missing_fields():
        logger.error("skipping source %s: config has no %s", spec.source_id, ", ".join(missing))
        return None
    return spec
