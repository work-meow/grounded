"""Telling the API how the connected sources are doing.

The polling loop already knew everything worth knowing — that a token no longer
opens the folder, that a listing came back empty, that the last three attempts
timed out — and put all of it in a log file. This publishes it instead, into the
document described by :mod:`rag_shared.health`, so the person who connected the
source can see it on the page where they connected it.

Two rules shape what goes out. Nothing here may raise: reporting runs inside the
polling thread, and a failure to *describe* an outage must not become an outage.
And no exception text is ever published — those carry internal hostnames, full
request URLs and, from some clients, the credential itself, and this document is
read by a browser. Every failure is classified into a sentence first, which has
the side effect of being far more useful than a stack trace to the person who
has to fix it.
"""

import logging
import threading
import time
from typing import Any, Protocol
from uuid import UUID

import httpx
from botocore.exceptions import BotoCoreError, ClientError
from rag_shared.health import HEALTH_KEY, SourceHealth, dump_health

from rag_indexer import bucket
from rag_indexer.config import IndexerSettings

logger = logging.getLogger(__name__)

#: How many passes a source may miss before the API stops believing its last
#: report. The document is rewritten after every pass, so silence for three of
#: them is not a slow service — it is an indexer that is no longer running, and
#: that must not read as health. The slack on top covers a restart.
_STALE_PASSES = 3
_STALE_SLACK_S = 120


class Reporter(Protocol):
    """Where a polling pass says how it went."""

    def report(
        self, source_id: UUID, *, ok: bool, problem: str = "", documents: int | None = None
    ) -> None: ...


class _Silent:
    """A reporter for tests and for the static-mode graph, which has no bucket."""

    def report(
        self, source_id: UUID, *, ok: bool, problem: str = "", documents: int | None = None
    ) -> None:
        return None


SILENT: Reporter = _Silent()


class BucketReporter:
    """Publishes every source's last outcome into the shared bucket.

    One instance per process, shared by every polling thread — which is why the
    lock is here. Each report rewrites the whole document, because the API reads
    the whole document and a partial one would mean a source whose entry is
    missing looks unchecked.

    Every report writes, including one that says the same thing as the last: the
    timestamp is the point. A source that is fine says so again, and that is what
    lets the API tell "healthy" from "nobody has looked in an hour".

    After a restart it starts empty and fills in as each source completes its
    first pass, so for a few seconds the document names fewer sources than exist.
    That reads in the UI as "not checked yet", which is what they are.
    """

    def __init__(self, settings: IndexerSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._sources: dict[UUID, SourceHealth] = {}
        self._client: Any | None = None
        self._stale_after_s = _STALE_PASSES * settings.refresh_interval_s + _STALE_SLACK_S

    def report(
        self, source_id: UUID, *, ok: bool, problem: str = "", documents: int | None = None
    ) -> None:
        with self._lock:
            previous = self._sources.get(source_id)
            self._sources[source_id] = SourceHealth(
                source_id=source_id,
                ok=ok,
                problem=problem,
                checked_at=int(time.time()),
                # A failed pass learned nothing about the contents, so the last
                # count stands. Reporting None would read as "never checked",
                # which is a different and less honest thing than "this many,
                # when it last worked".
                documents=documents
                if documents is not None
                else (previous.documents if previous is not None else None),
            )
            self._publish()

    def _publish(self) -> None:
        """Write the document. Called under the lock; never raises."""
        try:
            if self._client is None:
                self._client = bucket.client(self._settings)
            self._client.put_object(
                Bucket=self._settings.s3_bucket,
                Key=HEALTH_KEY,
                Body=dump_health(list(self._sources.values()), self._stale_after_s),
                ContentType="application/json",
            )
        except (ClientError, BotoCoreError, OSError):
            # Degrading to what the log has always done is the correct failure
            # here. The alternative — letting this reach the polling loop — would
            # turn a wobble in MinIO into a source that stops being read.
            logger.warning("could not publish source health; it will be republished next pass")


def describe_failure(exc: BaseException) -> str:
    """One sentence a user can act on, from an exception they must never see.

    Keyed on the status code where there is one, because that is the part every
    service agrees about: a 401 from Notion and a 401 from Graph mean the same
    thing to the person holding the token.
    """
    status = _status(exc)
    if status in (401, 403):
        return "нет доступа: токен недействителен или доступ к источнику отозван"
    if status == 404:
        return "источник не найден: папка или страница удалена либо переименована"
    if status == 429:
        return "сервис ограничивает частоту запросов, попробуем позже"
    if status is not None and status >= 500:
        return "сервис временно недоступен"
    if isinstance(exc, httpx.TimeoutException):
        return "сервис не отвечает"
    if isinstance(exc, (httpx.TransportError, OSError)):
        return "не удалось связаться с сервисом"
    return "не удалось прочитать источник"


def _status(exc: BaseException) -> int | None:
    """The HTTP status behind a failure, whichever client raised it.

    httpx puts it on ``response``; Google's client puts it on ``resp``, an
    httplib2 object whose ``status`` has been a string in some versions. Read by
    attribute rather than by isinstance so this module does not have to import
    the Google stack to describe a failure that did not come from it.
    """
    for candidate in (getattr(exc, "response", None), exc):
        value = getattr(candidate, "status_code", None)
        if isinstance(value, int):
            return value
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return None
