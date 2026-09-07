"""Reading a long chat.

A chat used to be returned whole, which is fine until it is not: the answer to
"open this conversation" grew with every turn ever taken in it. Now it is read
backwards from the end, a page at a time, the way it is looked at.

Two things have to hold for that to be an improvement rather than a rewrite of
the same list. The page has to be the *end* of the chat, because that is what
somebody opening it wants to see. And the cursor has to be exact — a page
boundary that skips a message loses it silently, and one that repeats a message
shows it twice.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.models import Message
from app.routers import chats

CHAT = uuid.UUID("44444444-4444-4444-4444-444444444444")
USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
START = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Session:
    """A database that answers with what the test put in it, and keeps the query.

    The query is half of what is being tested: it is where the ordering, the
    keyset filter and the one-extra-row trick live.
    """

    def __init__(self, rows):
        self._rows = rows
        self.query = None

    async def execute(self, query):
        self.query = query
        return _Rows(self._rows)


@pytest.fixture(autouse=True)
def owned(monkeypatch):
    async def _owned_chat(session, user_id, chat_id):
        return None

    monkeypatch.setattr(chats.conversation, "owned_chat", _owned_chat)


def _message(minute: int) -> Message:
    return Message(
        id=uuid.uuid4(),
        chat_id=CHAT,
        role="user" if minute % 2 == 0 else "assistant",
        content=f"сообщение {minute}",
        citations=[],
        created_at=START + timedelta(minutes=minute),
    )


def _newest_first(count: int) -> list[Message]:
    """What the query returns: newest first, because it reads backwards."""
    return [_message(minute) for minute in range(count, 0, -1)]


async def _page(rows: list[Message], *, limit: int = 2, before: str | None = None):
    session = _Session(rows)
    page = await chats.list_messages(CHAT, USER, session, before=before, limit=limit)
    return page, session


# --- what a page is ----------------------------------------------------------


async def test_an_empty_chat_is_an_empty_page_and_nothing_older():
    page, _ = await _page([])

    assert page.messages == []
    assert page.next_cursor is None


async def test_a_page_is_handed_over_in_reading_order():
    """Read backwards from the end, rendered forwards. Getting this the wrong
    way round would put every conversation in reverse."""
    page, _ = await _page(_newest_first(2))

    assert [m.content for m in page.messages] == ["сообщение 1", "сообщение 2"]


async def test_a_chat_that_ends_exactly_on_a_page_has_nothing_older():
    """Two messages, a page of two: the extra row the query asked for did not
    come back, and that absence is the whole answer."""
    page, _ = await _page(_newest_first(2), limit=2)

    assert len(page.messages) == 2
    assert page.next_cursor is None


async def test_a_longer_chat_offers_the_page_above():
    page, _ = await _page(_newest_first(3), limit=2)

    assert len(page.messages) == 2, "the extra row is a probe, not content"
    assert page.next_cursor is not None


async def test_the_cursor_starts_exactly_where_this_page_starts():
    """Off by one here and a message is either skipped for ever or shown twice."""
    page, _ = await _page(_newest_first(3), limit=2)

    oldest_on_page = page.messages[0]
    assert page.next_cursor is not None
    assert chats._cursor(page.next_cursor) == (oldest_on_page.created_at, oldest_on_page.id)


# --- the query itself --------------------------------------------------------


def _sql(session) -> str:
    return str(session.query.compile(dialect=postgresql.dialect()))


async def test_a_page_without_a_cursor_reads_from_the_end():
    _, session = await _page(_newest_first(2))
    sql = _sql(session)

    assert "ORDER BY messages.created_at DESC" in sql
    assert "messages.id) <" not in sql, "no cursor, no keyset filter"


async def test_a_cursor_becomes_a_keyset_filter_postgres_accepts():
    """A row comparison, not created_at alone: two messages can share a
    timestamp, and a filter that ignored the tiebreaker would loop on them."""
    cursor = chats._encode(_message(5))

    _, session = await _page(_newest_first(2), before=cursor)

    assert "(messages.created_at, messages.id) < " in _sql(session)


# --- cursors that did not come from us ---------------------------------------


@pytest.mark.parametrize("given", ["", "nonsense", "2026-03-01T10:00:00+00:00", "|", "x|y"])
async def test_a_cursor_that_is_not_one_is_refused_rather_than_run(given):
    """It reaches the query builder otherwise, and comes back a 500."""
    with pytest.raises(HTTPException) as raised:
        await _page(_newest_first(1), before=given)

    assert raised.value.status_code == 422


def test_a_cursor_survives_the_round_trip():
    message = _message(7)

    assert chats._cursor(chats._encode(message)) == (message.created_at, message.id)
