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

import pathway as pw
from rag_shared.connectors import ConnectorSpec, Kind

from rag_indexer import health
from rag_indexer.config import IndexerSettings
from rag_indexer.connectors import files
from rag_indexer.connectors.dropbox import DropboxSource
from rag_indexer.connectors.gdrive import GoogleDriveSource
from rag_indexer.connectors.notion import NotionSource
from rag_indexer.connectors.onedrive import OneDriveSource
from rag_indexer.connectors.remote import RemoteSource, polling_table
from rag_indexer.connectors.yandex import YandexSource

logger = logging.getLogger(__name__)

#: The kinds served by the shared polling loop, which is all of them. Google
#: Drive needs a second argument — the deployment's service account key — so it
#: is built in the branch below rather than looked up here.
_POLLED: dict[Kind, type[RemoteSource]] = {
    Kind.NOTION: NotionSource,
    Kind.DROPBOX: DropboxSource,
    Kind.ONEDRIVE: OneDriveSource,
    Kind.YANDEX: YandexSource,
}


def build_tables(
    settings: IndexerSettings,
    specs: list[ConnectorSpec],
    reporter: health.Reporter = health.SILENT,
) -> list[pw.Table]:
    """The uploads table, plus one table per source that could be connected.

    The reporter is passed in rather than made here so that the tests, and the
    static-mode graph, build the same connectors without a bucket to publish to.
    """
    tables = [files.build(settings)]
    credentials: dict | None = None

    for spec in specs:
        try:
            if spec.kind is Kind.GDRIVE:
                # The only kind that needs something from the deployment rather
                # than from the user, so the only one built by hand.
                credentials = credentials or _gdrive_credentials(settings)
                source: RemoteSource = GoogleDriveSource(spec, credentials)
            else:
                source = _POLLED[spec.kind](spec)
            table = polling_table(
                source,
                spec=spec,
                refresh_interval=settings.refresh_interval_s,
                size_limit=settings.max_document_bytes,
                reporter=reporter,
            )
        except Exception:
            logger.exception(
                "could not connect %s source %s; skipping it", spec.kind, spec.source_id
            )
            # A source skipped here is never polled, so it would never report —
            # and an unreported source shows in the UI as one nobody has looked
            # at yet, which for a Drive folder on a deployment with no service
            # account key would stay true for ever.
            reporter.report(
                spec.source_id,
                ok=False,
                problem="источник не удалось подключить: проверьте настройки",
            )
            continue

        tables.append(table)
        logger.info("connected %s source %r (%s)", spec.kind, spec.name, spec.source_id)

    return tables


def _gdrive_credentials(settings: IndexerSettings) -> dict:
    """The service account key, read once from the environment.

    It stays in memory. Pathway's connector wanted a path, and so needed this
    written to disk with the care a private key deserves; ours takes the parsed
    object, so the key never touches the filesystem.
    """
    raw = settings.gdrive_credentials_json.strip()
    if not raw:
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON is not set, so Drive sources cannot be read")
    try:
        credentials = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON is not valid JSON") from exc
    if not isinstance(credentials, dict):
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON is not a service account key")
    return credentials
