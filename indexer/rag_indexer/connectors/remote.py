"""Connectors for services that only offer list-and-download.

Pathway ships connectors for object storage and for Google Drive. For everything
else — Notion, Dropbox, OneDrive, Yandex Disk — the documented extension point is
``ConnectorSubject``, which is what Pathway's own connectors are built on too.

Those four services differ in exactly two ways: how you ask what they hold, and
how you pull one document's bytes. That is :class:`RemoteSource`. Everything
else — polling, diffing against what is already indexed, deletions, skipping
what we cannot read, surviving an outage at the far end — is here, once.

Failure policy, which is the whole reason this is shared: a remote service is
allowed to be down, slow, or lying. Nothing it does may raise out of
:meth:`_PollingSubject.run`, because that thread failing takes the process with
it — and with it the *other* users' sources and their uploads. A failed listing
keeps the previous snapshot; a failed download retries on the next pass.
"""

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

import pathway as pw
from pathway.internals import api
from pathway.internals.api import SessionType
from pathway.io.python import ConnectorSubject
from rag_shared.connectors import ConnectorSpec
from rag_shared.formats import is_supported

from rag_indexer.connectors.contract import clean_name, describe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One document as the far end describes it."""

    #: Stable for the lifetime of the file *at the service*. Whatever it calls
    #: an id — a Notion page id, a Dropbox file id, a Graph item id. Not a path:
    #: a renamed file must stay the same document, or its citations break.
    external_id: str
    filename: str
    #: Unix seconds. The change detector: a document is re-read when this moves.
    modified_at: int
    #: Where a person opens the original. None means there is no such page.
    web_url: str | None = None
    #: Bytes, when the listing says. Lets an oversized file be skipped before it
    #: is downloaded rather than after.
    size: int | None = None


class RemoteSource(Protocol):
    """What a connector has to be able to do."""

    def list(self) -> Iterable[RemoteFile]:
        """Everything the source currently holds. May raise; the caller handles it."""
        ...

    def fetch(self, file: RemoteFile) -> bytes:
        """One document's bytes. May raise; the caller handles it."""
        ...

    def close(self) -> None:
        """Release the HTTP connection pool."""
        ...


class _PollingSubject(ConnectorSubject):
    """Turns a :class:`RemoteSource` into a live Pathway input."""

    def __init__(
        self,
        source: RemoteSource,
        *,
        spec: ConnectorSpec,
        refresh_interval: float,
        size_limit: int,
        mode: Literal["streaming", "static"] = "streaming",
    ) -> None:
        super().__init__(datasource_name=spec.kind.value)
        self._source = source
        self._spec = spec
        self._refresh_interval = refresh_interval
        self._size_limit = size_limit
        self._mode = mode
        self._label = f"{spec.kind.value} source {spec.source_id}"

    @property
    def _session_type(self) -> SessionType:
        # UPSERT, so re-adding a key replaces the document rather than
        # duplicating it. This is what makes an edited file re-index cleanly.
        return SessionType.UPSERT

    @property
    def _with_metadata(self) -> bool:
        return True

    def on_stop(self) -> None:
        self._source.close()

    def run(self) -> None:
        # external_id -> the modified_at we last acted on. Held in memory only:
        # on restart the whole source is listed again, which costs one pass and
        # no embeddings (those are cached on disk by document content).
        indexed: dict[str, int] = {}
        while True:
            started = time.monotonic()
            self._poll(indexed)
            self.commit()
            # Static mode reads the source once and ends the stream, matching
            # what Pathway's own connectors do with it. It is what lets a test
            # run a connector through a real graph rather than around it.
            if self._mode == "static":
                return
            time.sleep(max(0.0, self._refresh_interval - (time.monotonic() - started)))

    def _poll(self, indexed: dict[str, int]) -> None:
        try:
            listing = {file.external_id: file for file in self._source.list()}
        except Exception:
            logger.exception("%s: could not be listed; keeping the last snapshot", self._label)
            return

        for external_id in indexed.keys() - listing.keys():
            self._remove(api.ref_scalar(external_id), b"")
            del indexed[external_id]

        for file in listing.values():
            if indexed.get(file.external_id) == file.modified_at:
                continue
            if self._store(file):
                indexed[file.external_id] = file.modified_at

    def _store(self, file: RemoteFile) -> bool:
        """Index one new or changed document.

        Returns whether the decision is final. A deliberate skip counts as final
        — the file is recorded at this revision so it is not reconsidered, and
        not logged about, every ten minutes for as long as it exists.
        """
        # The cleaned name, because that is the one the document is indexed
        # under: a remote file called "report.pdf " has the suffix ".pdf " until
        # it is cleaned, and would be turned away for a format we do read.
        if not is_supported(clean_name(file.filename)):
            logger.info("%s: skipping %r, not a format we read", self._label, file.filename)
            return True
        if file.size is not None and file.size > self._size_limit:
            logger.warning(
                "%s: skipping %r, %d bytes is over the %d limit",
                self._label,
                file.filename,
                file.size,
                self._size_limit,
            )
            return True

        try:
            payload = self._source.fetch(file)
        except Exception:
            logger.exception(
                "%s: could not read %r; retrying next pass", self._label, file.filename
            )
            return False

        if len(payload) > self._size_limit:
            # The listing under-reported, or did not report at all.
            logger.warning(
                "%s: skipping %r, %d bytes is over the %d limit",
                self._label,
                file.filename,
                len(payload),
                self._size_limit,
            )
            return True

        metadata = describe(
            user_id=self._spec.user_id,
            source_id=self._spec.source_id,
            external_id=file.external_id,
            filename=file.filename,
            modified_at=file.modified_at,
            web_url=file.web_url,
            size=file.size if file.size is not None else len(payload),
        )
        self._add(api.ref_scalar(file.external_id), payload, json.dumps(metadata).encode())
        return True


def polling_table(
    source: RemoteSource,
    *,
    spec: ConnectorSpec,
    refresh_interval: float,
    size_limit: int,
    mode: Literal["streaming", "static"] = "streaming",
) -> pw.Table:
    """The input table for one polled source, already in our metadata shape."""
    return pw.io.python.read(
        _PollingSubject(
            source,
            spec=spec,
            refresh_interval=refresh_interval,
            size_limit=size_limit,
            mode=mode,
        ),
        # Pathway deprecates `format` in favour of a schema and next(**values),
        # but that path has no key, so it cannot express an upsert or a delete —
        # which is the whole point of a source that changes. Pathway's own
        # gdrive and pyfilesystem connectors call it exactly like this, and emit
        # the same warning at graph build; one line per connector, at startup.
        format="binary",
        name=f"{spec.kind.value}-{spec.source_id}",
    )
