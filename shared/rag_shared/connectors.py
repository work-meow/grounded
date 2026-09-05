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
from hashlib import sha256
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


#: Which of a kind's fields say *what* is being read, as opposed to merely
#: proving we are allowed to read it. Connecting the same thing twice indexes
#: every fragment of it twice and pays for the embeddings twice, so it is
#: refused — and this is the definition of "the same thing".
#:
#: The credential is in here for four of the five kinds, because it is what
#: names the account: one Notion token is one workspace, and a Yandex token
#: plus a path is one folder belonging to one person. That is a limit as much
#: as a definition. Someone who runs the OAuth dance a second time gets a
#: different refresh token for the same Dropbox, and this will not recognise
#: it; doing better would mean calling each service to resolve an account id
#: while the user waits on the form. Refusing the duplicate somebody actually
#: creates — the same folder added twice — is what this is for.
IDENTITY_FIELDS: dict[Kind, tuple[str, ...]] = {
    Kind.GDRIVE: ("folder_id",),
    Kind.NOTION: ("token",),
    Kind.YANDEX: ("token", "path"),
    Kind.DROPBOX: ("app_key", "refresh_token", "path"),
    Kind.ONEDRIVE: ("client_id", "refresh_token", "path"),
}


def fingerprint(kind: Kind, config: dict[str, str]) -> str:
    """What makes this source *this* source, as 64 hex characters.

    A digest rather than the values themselves, because those values are
    credentials and this one is stored in a plain column beside the sealed blob.
    Sealing exists so that a database dump gives up the source list and not the
    tokens behind it; a fingerprint that undid that would be worse than having
    none. The inputs are opaque ids and high-entropy secrets, so the digest
    gives nothing back.
    """
    material = "\n".join(
        f"{field}={_identity(field, config.get(field, ''))}" for field in IDENTITY_FIELDS[kind]
    )
    return sha256(f"{kind.value}\n{material}".encode()).hexdigest()


def _identity(field: str, value: str) -> str:
    """One identity value, in its canonical spelling.

    Only the folder fields need it: "/Документы", "Документы/" and "Документы"
    are one folder, and typing it a different way the second time must not buy a
    second copy of it. Credentials are left exactly as they are — a token is not
    a path and has no spelling to normalise.
    """
    return value.strip().strip("/") if field == "path" else value


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
