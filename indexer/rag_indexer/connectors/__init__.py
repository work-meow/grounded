"""Every input the index has.

One list of tables, built once, at startup. Pathway's dataflow graph is fixed the
moment ``DocumentStore`` is constructed, which is the single fact that shapes
everything about connected sources: they cannot be added to a running process,
so the manifest that describes them is read here and the process restarts when
it changes (see :mod:`rag_indexer.manifest`).

A source that will not build is skipped, loudly. One user's expired Dropbox
token must not cost another user their uploads.
"""

import json
import logging
import os
import tempfile

import pathway as pw
from rag_shared.connectors import ConnectorSpec, Kind

from rag_indexer.config import IndexerSettings
from rag_indexer.connectors import files, gdrive
from rag_indexer.connectors.dropbox import DropboxSource
from rag_indexer.connectors.notion import NotionSource
from rag_indexer.connectors.onedrive import OneDriveSource
from rag_indexer.connectors.remote import RemoteSource, polling_table
from rag_indexer.connectors.yandex import YandexSource

logger = logging.getLogger(__name__)

#: The kinds served by the shared polling loop. Google Drive is absent because
#: Pathway has a connector for it that does the same job better.
_POLLED: dict[Kind, type[RemoteSource]] = {
    Kind.NOTION: NotionSource,
    Kind.DROPBOX: DropboxSource,
    Kind.ONEDRIVE: OneDriveSource,
    Kind.YANDEX: YandexSource,
}


def build_tables(settings: IndexerSettings, specs: list[ConnectorSpec]) -> list[pw.Table]:
    """The uploads table, plus one table per source that could be connected."""
    tables = [files.build(settings)]
    credentials: str | None = None

    for spec in specs:
        try:
            if spec.kind is Kind.GDRIVE:
                credentials = credentials or _gdrive_credentials(settings)
                table = gdrive.build(
                    spec,
                    credentials_file=credentials,
                    refresh_interval=settings.refresh_interval_s,
                    size_limit=settings.max_document_bytes,
                )
            else:
                table = polling_table(
                    _POLLED[spec.kind](spec),
                    spec=spec,
                    refresh_interval=settings.refresh_interval_s,
                    size_limit=settings.max_document_bytes,
                )
        except Exception:
            logger.exception(
                "could not connect %s source %s; skipping it", spec.kind, spec.source_id
            )
            continue

        tables.append(table)
        logger.info("connected %s source %r (%s)", spec.kind, spec.name, spec.source_id)

    return tables


def _gdrive_credentials(settings: IndexerSettings) -> str:
    """The service account key on disk, because Google's client wants a path.

    Written from an environment variable rather than mounted so that the same
    compose file works with the key held in ``.env`` next to every other secret.
    Created with 0600 by ``mkstemp`` and left in place: Pathway opens it when the
    connector's thread starts, not when the graph is built.
    """
    raw = settings.gdrive_credentials_json.strip()
    if not raw:
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON is not set, so Drive sources cannot be read")
    try:
        json.loads(raw)
    except ValueError as exc:
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON is not valid JSON") from exc

    handle, path = tempfile.mkstemp(prefix="gdrive-", suffix=".json")
    with os.fdopen(handle, "w") as file:
        file.write(raw)
    return path
