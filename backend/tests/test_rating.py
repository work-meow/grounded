"""Marking an answer good or bad.

Small, and worth its own tests for one reason: what it collects is the eval set
for everything else in this system. A rating that lands on the wrong message, or
on somebody else's chat, is worse than no rating at all.
"""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.models import Message
from app.routers import chats

CHAT = uuid.UUID("44444444-4444-4444-4444-444444444444")
USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _Session:
    """Just enough of an AsyncSession to answer one lookup and one commit."""

    def __init__(self, message: Message | None):
        self._message = message
        self.committed = False

    async def execute(self, _statement):
        return self

    def scalar_one_or_none(self):
        return self._message

    async def commit(self):
        self.committed = True


def _answer(role: str = "assistant", rating: int | None = None) -> Message:
    return Message(
        id=uuid.uuid4(),
        chat_id=CHAT,
        role=role,
        content="сорок",
        citations=[],
        rating=rating,
        created_at=datetime.now(UTC),
    )


@pytest.fixture(autouse=True)
def owned(monkeypatch):
    async def _owned_chat(session, user_id, chat_id):
        return None

    monkeypatch.setattr(chats, "_owned_chat", _owned_chat)


async def _rate(message: Message | None, rating: int) -> _Session:
    session = _Session(message)
    await chats.rate(CHAT, uuid.uuid4(), chats.RatingIn(rating=rating), USER, session)
    return session


@pytest.mark.parametrize("rating", [1, -1])
async def test_an_answer_can_be_marked(rating):
    message = _answer()

    session = await _rate(message, rating)

    assert message.rating == rating
    assert session.committed


async def test_zero_takes_it_back_rather_than_recording_indifference():
    """Null is "no opinion". An answer nobody rated is not an answer nobody
    minded, and storing zero would make those two the same row."""
    message = _answer(rating=1)

    await _rate(message, 0)

    assert message.rating is None


async def test_a_question_is_not_something_to_rate():
    """The reader wrote it. Saying so beats storing an opinion nothing reads."""
    with pytest.raises(HTTPException) as raised:
        await _rate(_answer(role="user"), 1)

    assert raised.value.status_code == 404


async def test_a_message_that_is_not_in_this_chat_is_not_found():
    with pytest.raises(HTTPException) as raised:
        await _rate(None, 1)

    assert raised.value.status_code == 404


@pytest.mark.parametrize("rating", [2, -2, 5, 100])
def test_a_scale_is_refused_at_the_door(rating):
    """Three values, because there is nothing here to average."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        chats.RatingIn(rating=rating)
