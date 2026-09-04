"""Google Drive folders as documents.

Pathway ships this connector, so the work here is only the two things it cannot
know about: which files are worth downloading, and how to say who owns them.

Access is granted the other way round from the usual OAuth flow. The deployment
holds one service account; the user shares a folder with its address. That is
strictly less access than an OAuth consent would give — the service account can
read one folder, read-only, and the user revokes it from Drive's own sharing
dialog without coming back here — and it means this system never holds a
credential that could reach the rest of somebody's Drive.
"""

import logging
from typing import Any

import pathway as pw
from rag_shared.connectors import ConnectorSpec
from rag_shared.formats import is_supported

from rag_indexer.connectors.contract import COMMIT_INTERVAL_MS, conform, describe
from rag_indexer.connectors.http import as_timestamp

logger = logging.getLogger(__name__)

#: Google's own formats have no file extension, and Pathway exports them to
#: Office formats on download. Naming them accordingly is what lets the rest of
#: the system — the format check here, the parser later — treat them as files.
#: Must agree with ``pw.io.gdrive.DEFAULT_MIME_TYPE_MAPPING``.
_GOOGLE_EXPORTS = {
    "application/vnd.google-apps.document": ".docx",
    "application/vnd.google-apps.spreadsheet": ".xlsx",
    "application/vnd.google-apps.presentation": ".pptx",
}


def build(
    spec: ConnectorSpec,
    *,
    credentials_file: str,
    refresh_interval: float,
    size_limit: int,
) -> pw.Table:
    table = pw.io.gdrive.read(
        object_id=spec.config["folder_id"],
        mode="streaming",
        refresh_interval=refresh_interval,
        service_user_credentials_file=credentials_file,
        with_metadata=True,
        # Enforced by the connector before it downloads, so a video somebody
        # dropped in the folder costs one listing entry rather than its size.
        object_size_limit=size_limit,
        name=f"gdrive-{spec.source_id}",
        # Forwarded by pw.io.gdrive.read to the python connector underneath.
        autocommit_duration_ms=COMMIT_INTERVAL_MS,
    )

    @pw.udf
    def readable(metadata: pw.Json) -> bool:
        return is_supported(_filename(metadata.as_dict()))

    def description(metadata: dict[str, Any]) -> dict:
        return describe(
            user_id=spec.user_id,
            source_id=spec.source_id,
            external_id=metadata["id"],
            filename=_filename(metadata),
            modified_at=as_timestamp(metadata.get("modifiedTime")),
            # The canonical "open in Drive" link. Built rather than requested:
            # webViewLink is not among the fields Pathway asks Drive for.
            web_url=f"https://drive.google.com/file/d/{metadata['id']}/view",
            # Drive reports it as a string, and omits it entirely for its own
            # formats, which are not blobs until they are exported.
            size=_int_or_none(metadata.get("size")),
        )

    return conform(table.filter(readable(pw.this._metadata)), description)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filename(metadata: dict[str, Any]) -> str:
    name = str(metadata.get("name") or "")
    suffix = _GOOGLE_EXPORTS.get(str(metadata.get("mimeType") or ""))
    if suffix and not name.endswith(suffix):
        return name + suffix
    return name
