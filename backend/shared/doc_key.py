"""The S3 key layout — the single contract between the API and the indexer.

The API writes keys; the indexer parses them back into metadata so that every
chunk carries ``user_id``/``source_id``/``document_id`` and retrieval can be
filtered on exact equality rather than on string matching over a path.

Zero dependencies on purpose: the two processes live in separate virtualenvs
(the indexer's Pathway pins langchain<0.4, the API needs >=1.0) and this module
is imported by both. Keep it stdlib-only.
"""

import re
from uuid import UUID

PREFIX = "users"
_KEY_RE = re.compile(
    r"^users/(?P<user_id>[0-9a-f-]{36})"
    r"/sources/(?P<source_id>[0-9a-f-]{36})"
    r"/(?P<document_id>[0-9a-f-]{36})"
    r"/(?P<filename>.+)$"
)


def build_key(user_id: UUID, source_id: UUID, document_id: UUID, filename: str) -> str:
    """users/<user>/sources/<source>/<document>/<filename>

    ``document_id`` is part of the path so that uploading the same filename
    twice cannot collide.
    """
    return f"{PREFIX}/{user_id}/sources/{source_id}/{document_id}/{filename}"


def user_prefix(user_id: UUID) -> str:
    return f"{PREFIX}/{user_id}/"


def parse_key(key: str) -> dict[str, str] | None:
    """Split a key back into its parts, or None if it is not one of ours.

    Returns plain ``str`` values (not UUID) because this feeds JSON metadata.
    """
    match = _KEY_RE.match(key.lstrip("/"))
    return match.groupdict() if match else None


def tenant_metadata(text: str, metadata: dict) -> tuple[str, dict]:
    """Pathway document post-processor: lift the ids out of the object key.

    Retrieval filters on these fields with exact equality, which is what makes
    one user's chunks unreachable from another user's query. An object whose key
    does not follow the layout gets null ids, so it can never satisfy a filter.

    Lives here rather than in the indexer so that the code writing the key, the
    code reading it back, and the tests all share one implementation.
    """
    parsed = parse_key(str(metadata.get("path", "")))
    if parsed is None:
        return text, {**metadata, "user_id": None, "document_id": None}
    return text, {**metadata, **parsed}
