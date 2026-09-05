"""What the indexer tells the API about the sources it is polling.

This channel exists because of one failure: a token revoked at the far end left
the source looking perfectly healthy in the UI, with documents that quietly
stopped being updated. So the two things worth testing are the two that failure
turns on — that a bad pass says so, and that saying so can never itself break
the pass it is describing.

The third is the classifier. It is the only place an exception is turned into
something a browser will render, and exceptions from these clients carry urls,
internal hostnames and, from at least one of them, the credential.
"""

import uuid

import httpx
import pytest
from botocore.exceptions import EndpointConnectionError
from rag_shared.health import SourceHealth, dump_health, load_health

from rag_indexer.config import IndexerSettings
from rag_indexer.health import BucketReporter, describe_failure

SOURCE = uuid.UUID("22222222-2222-2222-2222-222222222222")
OTHER = uuid.UUID("33333333-3333-3333-3333-333333333333")


class _Bucket:
    """MinIO, as far as the reporter can tell."""

    def __init__(self):
        self.written: list[bytes] = []
        self.fails = False

    def put_object(self, **kwargs):
        if self.fails:
            raise EndpointConnectionError(endpoint_url="http://minio:9000")
        self.written.append(kwargs["Body"])


@pytest.fixture
def reporter(monkeypatch):
    settings = IndexerSettings(openrouter_api_key="unused")
    bucket = _Bucket()
    built = BucketReporter(settings)
    built._client = bucket
    return built, bucket


def _published(bucket: _Bucket):
    return load_health(bucket.written[-1])


# --- what gets published -----------------------------------------------------


def test_a_failed_pass_is_published_with_its_reason(reporter):
    built, bucket = reporter

    built.report(SOURCE, ok=False, problem="нет доступа")

    entry = _published(bucket).sources[SOURCE]
    assert entry.ok is False
    assert entry.problem == "нет доступа"


def test_a_good_pass_publishes_what_it_found(reporter):
    built, bucket = reporter

    built.report(SOURCE, ok=True, documents=7)

    entry = _published(bucket).sources[SOURCE]
    assert (entry.ok, entry.documents) == (True, 7)


def test_every_source_is_in_every_write(reporter):
    """The API reads the whole document, so a partial one would report the
    sources missing from it as never checked."""
    built, bucket = reporter

    built.report(SOURCE, ok=True, documents=1)
    built.report(OTHER, ok=False, problem="сервис недоступен")

    assert set(_published(bucket).sources) == {SOURCE, OTHER}


def test_a_failure_keeps_the_count_the_last_good_pass_found(reporter):
    """ "Ten documents, and the last check failed" is true; "no idea how many"
    is not, and would read in the UI as a source nobody has ever looked at."""
    built, bucket = reporter
    built.report(SOURCE, ok=True, documents=10)

    built.report(SOURCE, ok=False, problem="сервис недоступен")

    assert _published(bucket).sources[SOURCE].documents == 10


def test_an_unchanged_report_is_still_written(reporter):
    """The timestamp is the point. Two identical "all fine" reports an hour
    apart are different facts, and the second is what stops the API from
    deciding the indexer has stopped."""
    built, bucket = reporter

    built.report(SOURCE, ok=True, documents=1)
    built.report(SOURCE, ok=True, documents=1)

    assert len(bucket.written) == 2


def test_the_staleness_window_leaves_room_for_missed_passes(reporter):
    """One late pass is not a dead indexer; several in a row is."""
    built, bucket = reporter
    built.report(SOURCE, ok=True, documents=1)

    interval = IndexerSettings(openrouter_api_key="unused").refresh_interval_s
    assert _published(bucket).stale_after_s > interval


def test_a_bucket_that_is_down_does_not_break_the_pass(reporter):
    """Reporting an outage must not become one. This runs inside the polling
    thread, and an exception here would stop the source being read at all."""
    built, bucket = reporter
    bucket.fails = True

    built.report(SOURCE, ok=False, problem="нет доступа")

    assert bucket.written == []


# --- the wire format ---------------------------------------------------------


def test_a_health_document_survives_the_round_trip():
    entry = SourceHealth(
        source_id=SOURCE, ok=False, problem="нет доступа", checked_at=1_700_000_000, documents=0
    )

    parsed = load_health(dump_health([entry], 900))

    assert parsed.sources == {SOURCE: entry}
    assert parsed.stale_after_s == 900


@pytest.mark.parametrize("given", [b"", b"{}", b"not json", b'{"sources": 3}'])
def test_an_unreadable_document_reports_nothing_rather_than_raising(given):
    """It is written by another process that may be a version ahead. Listing
    sources must not fail because their health could not be read."""
    assert load_health(given).sources == {}


def test_one_bad_entry_costs_only_itself():
    raw = b"""{"version": 1, "stale_after_s": 900, "sources": [
      {"source_id": "nonsense", "ok": true, "problem": "", "checked_at": 1, "documents": 0},
      {"source_id": "22222222-2222-2222-2222-222222222222",
       "ok": true, "problem": "", "checked_at": 1, "documents": 4}
    ]}"""

    assert set(load_health(raw).sources) == {SOURCE}


# --- turning an exception into a sentence ------------------------------------


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.example.test/v1/list?token=hunter2")
    return httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(status, request=request)
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "нет доступа"), (403, "нет доступа"), (404, "не найден"), (429, "ограничивает")],
)
def test_the_status_decides_what_the_user_is_told(status, expected):
    assert expected in describe_failure(_http_error(status))


def test_a_service_having_a_bad_day_reads_as_a_service_having_a_bad_day():
    assert describe_failure(_http_error(503)) == "сервис временно недоступен"


def test_a_google_style_error_is_read_too():
    """googleapiclient keeps the status somewhere else entirely."""

    class _HttpError(Exception):
        resp = type("Resp", (), {"status": "403"})()

    assert "нет доступа" in describe_failure(_HttpError())


def test_a_network_failure_says_so_without_quoting_the_exception():
    assert describe_failure(httpx.ConnectError("[Errno -2] Name or service not known"))


@pytest.mark.parametrize(
    "exc",
    [
        _http_error(401),
        _http_error(500),
        httpx.ConnectError("nodename nor servname provided"),
        ValueError("token=hunter2 rejected by https://internal.host/v1"),
    ],
)
def test_nothing_from_the_exception_itself_reaches_the_message(exc):
    """The whole reason this is a classifier and not str(exc): these carry urls,
    hostnames and — from clients that log the request — the credential."""
    message = describe_failure(exc)

    assert "hunter2" not in message
    assert "http" not in message
    assert message == message.strip() and message
