"""What the connectors' HTTP layer has to guarantee.

Two things, and both are about a far end this system does not control. It may
be rate-limiting us, in which case retrying hard is how an integration gets
suspended. And it may send more than it said it would — the size in a listing
is the service's word, and Notion gives no number at all — in which case
reading the body into memory would let it decide how much memory this process
uses. That is the failure that once took the indexer from 225 MB to 1.7 GB.
"""

import httpx
import pytest

from rag_indexer.connectors.http import Http, as_timestamp


@pytest.fixture
def http(monkeypatch):
    """An Http whose transport and backoff the test controls."""

    def build(handler):
        client = Http("test", timeout=1.0)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(Http, "_wait", lambda self, attempt, retry_after: None)
        return client

    return build


# --- the download ceiling ----------------------------------------------------


def _serving(body: bytes):
    return lambda request: httpx.Response(200, content=body)


def test_a_body_within_the_limit_comes_back_whole(http):
    client = http(_serving(b"x" * 500))

    assert client.download("GET", "https://example.test/f", limit=1024) == b"x" * 500


def test_a_body_over_the_limit_is_refused_rather_than_returned(http):
    client = http(_serving(b"x" * 5000))

    assert client.download("GET", "https://example.test/f", limit=1024) is None


def test_the_limit_is_exact(http):
    client = http(_serving(b"x" * 1024))

    assert client.download("GET", "https://example.test/f", limit=1024) == b"x" * 1024
    assert client.download("GET", "https://example.test/f", limit=1023) is None


def test_an_error_status_still_raises(http):
    client = http(lambda request: httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        client.download("GET", "https://example.test/f", limit=1024)


# --- retries -----------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_retryable_status_is_retried_then_succeeds(http, status):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return (
            httpx.Response(200, json={"ok": True}) if attempts["n"] > 1 else httpx.Response(status)
        )

    assert http(handler).json("GET", "https://example.test/x") == {"ok": True}
    assert attempts["n"] == 2


def test_a_service_that_keeps_failing_surfaces_rather_than_looping(http):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        http(handler).json("GET", "https://example.test/x")
    assert attempts["n"] == 3, "three attempts, not an unbounded retry against a struggling service"


def test_a_bad_request_is_not_retried(http):
    """A 401 will be a 401 next time too; retrying it only spends quota."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(401)

    with pytest.raises(httpx.HTTPStatusError):
        http(handler).json("GET", "https://example.test/x")
    assert attempts["n"] == 1


def test_a_dropped_connection_is_retried(http):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("dropped")
        return httpx.Response(200, json=[])

    assert http(handler).json("GET", "https://example.test/x") == []


# --- timestamps --------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("2026-03-01T10:00:00.000Z", 1772359200),
        ("2026-03-01T10:00:00+00:00", 1772359200),
        (None, 0),
        ("", 0),
        ("вчера", 0),
    ],
)
def test_an_instant_that_makes_no_sense_reads_as_zero(given, expected):
    """Zero, not "now": this value is the change detector, and one that moved on
    every poll would re-download and re-embed the document on every poll."""
    assert as_timestamp(given) == expected
