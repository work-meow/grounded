"""The SSE relay: what reaches the client, and what survives a lost client.

Both cases here have cost a real answer before. A browser tab that closes does
not raise ``Exception`` — the generator is thrown ``GeneratorExit`` — so the
tokens already produced are only kept if the save sits in ``finally``. And the
error event goes to a browser, so it must carry no internal detail.
"""

import uuid
from collections.abc import AsyncIterator

import pytest

from app.routers import chats


@pytest.fixture
def saved(monkeypatch) -> list[tuple[str, list]]:
    """Capture what would be written to the messages table."""
    written: list[tuple[str, list]] = []

    async def fake_save(chat_id, text, citations, trace=None):
        written.append((text, citations, trace))

    monkeypatch.setattr(chats.conversation, "save_answer", fake_save)
    return written


def agent_yielding(*events, fail: bool = False):
    async def answer(*_args, **_kwargs) -> AsyncIterator[tuple[str, object]]:
        for event in events:
            yield event
        if fail:
            raise RuntimeError("postgresql://rag:hunter2@postgres1:5432/rag is unreachable")

    return answer


def stream():
    return chats._stream(
        settings=None,
        user_id=uuid.uuid4(),
        chat_id=uuid.uuid4(),
        question="q",
        history=[],
        web=False,
    )


async def test_completed_answer_is_streamed_and_saved(monkeypatch, saved):
    monkeypatch.setattr(
        chats.conversation.agent,
        "answer",
        agent_yielding(("token", "45 "), ("token", "дней"), ("citations", [{"n": 1}])),
    )

    records = [record async for record in stream()]

    assert "event: done" in records[-1]
    assert [(text, citations) for text, citations, _ in saved] == [("45 дней", [{"n": 1}])]


async def test_a_client_that_disappears_mid_answer_does_not_lose_it(monkeypatch, saved):
    monkeypatch.setattr(
        chats.conversation.agent,
        "answer",
        agent_yielding(("token", "начало"), ("token", " ответа")),
    )

    generator = stream()
    await anext(generator)  # one token reaches the client
    await generator.aclose()  # the browser tab closes

    assert [(text, citations) for text, citations, _ in saved] == [("начало", [])]


async def test_the_error_event_carries_no_internals(monkeypatch, saved):
    monkeypatch.setattr(
        chats.conversation.agent, "answer", agent_yielding(("token", "част"), fail=True)
    )

    records = [record async for record in stream()]
    error = next(record for record in records if record.startswith("event: error"))

    assert "postgresql" not in error and "hunter2" not in error
    # The partial answer is still kept rather than thrown away with the failure,
    # and so is its trace: a turn that broke is the one worth looking at later.
    text, citations, trace = saved[0]
    assert (text, citations) == ("част", [])
    assert set(trace) == {"steps", "shown", "usage"}
