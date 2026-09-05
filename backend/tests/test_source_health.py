"""What the source list says about a source that has stopped working.

The bug this replaces: a connector whose token had been revoked went on being
listed exactly like a working one. The indexer knew — it logged the failure
every ten minutes — and nothing carried that back here.

So these are about the reading half of that channel, and mostly about the two
ways it could go on lying. A source nobody has reported on must not read as
healthy, and neither must a report that is hours old: the indexer republishes
after every pass, so an old "all fine" says nothing about now.
"""

import asyncio
import time
import uuid
from datetime import UTC, datetime

import pytest
from rag_shared.health import HEALTH_KEY, SourceHealth, dump_health

from app import connectors, storage
from app.config import Settings
from app.models import Source
from app.routers.connectors import _out

SOURCE = uuid.UUID("22222222-2222-2222-2222-222222222222")
WINDOW = 1_920  # what the indexer publishes for a ten-minute refresh


def settings() -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k")


@pytest.fixture
def bucket(monkeypatch):
    """Whatever the indexer last published, or nothing at all."""

    def publish(raw: bytes | None):
        async def get(_settings, key):
            assert key == HEALTH_KEY
            return raw

        monkeypatch.setattr(storage, "get", get)

    return publish


def _entry(*, ok=True, problem="", age_s=0, documents=3) -> bytes:
    return dump_health(
        [
            SourceHealth(
                source_id=SOURCE,
                ok=ok,
                problem=problem,
                checked_at=int(time.time()) - age_s,
                documents=documents,
            )
        ],
        WINDOW,
    )


async def test_a_source_the_indexer_is_happy_with_reads_as_ok(bucket):
    bucket(_entry())

    health = (await connectors.statuses(settings()))[SOURCE]

    assert health.status is connectors.Status.OK
    assert health.problem == ""
    assert health.documents == 3


async def test_a_failing_source_carries_the_reason_the_indexer_gave(bucket):
    bucket(_entry(ok=False, problem="нет доступа: токен недействителен"))

    health = (await connectors.statuses(settings()))[SOURCE]

    assert health.status is connectors.Status.ERROR
    assert health.problem == "нет доступа: токен недействителен"


async def test_a_failing_source_keeps_showing_what_it_last_held(bucket):
    """ "Ten documents, and the last check failed" is a more useful thing to see
    than a source with no numbers on it at all."""
    bucket(_entry(ok=False, problem="сервис недоступен", documents=10))

    assert (await connectors.statuses(settings()))[SOURCE].documents == 10


async def test_an_old_report_saying_everything_is_fine_is_not_believed(bucket):
    """The indexer republishes after every pass. An "all fine" from three hours
    ago is not evidence that anything is fine — it is evidence that nothing has
    looked since, which is the exact shape of the failure this channel is for.
    """
    bucket(_entry(ok=True, age_s=WINDOW + 60))

    health = (await connectors.statuses(settings()))[SOURCE]

    assert health.status is connectors.Status.ERROR
    assert "индексатор" in health.problem


async def test_a_report_inside_the_window_is_believed(bucket):
    bucket(_entry(ok=True, age_s=WINDOW - 60))

    assert (await connectors.statuses(settings()))[SOURCE].status is connectors.Status.OK


@pytest.mark.parametrize("published", [None, b"", b"not json", b'{"version": 99}'])
async def test_nothing_readable_means_nothing_known(bucket, published):
    """Not an error, either: a deployment whose indexer has never started has
    sources that are genuinely unchecked, and the list must still render."""
    bucket(published)

    assert await connectors.statuses(settings()) == {}


def test_a_source_with_no_report_is_unknown_rather_than_ok():
    """The default that matters. "Nobody has looked" and "it works" are the two
    things this whole channel exists to keep apart."""
    row = Source(id=SOURCE, kind="notion", name="Заметки", created_at=datetime.now(UTC))

    out = _out(row, connectors.UNCHECKED)

    assert out.status is connectors.Status.UNKNOWN
    assert out.checked_at is None
    assert out.documents is None


async def test_a_wedged_object_store_does_not_hold_the_page(monkeypatch):
    """The list of sources has to render either way. Waiting out botocore's
    retries would leave somebody looking at a spinner for a status line."""

    async def never(_settings, _key):
        await asyncio.sleep(30)

    monkeypatch.setattr(storage, "get", never)
    monkeypatch.setattr(connectors, "_TIMEOUT_S", 0.05)

    assert await connectors.statuses(settings()) == {}
