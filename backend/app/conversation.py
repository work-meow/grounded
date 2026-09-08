"""One turn of a conversation, and the state a conversation keeps.

Three callers run a turn now: the browser's SSE stream, the public API's JSON
answer, and the OpenAI-compatible endpoint. They differ in what they emit and
in whether they persist anything. What they share is here — who owns a chat,
how much history the model is given, what becomes of an answer, and what the
turn cost.

Sharing it is not tidiness. The title rule, the size of the history window and
the save-whatever-arrived discipline are each a decision that was arrived at
once; three copies of them would drift, and the drift would be silent.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent
from app.config import Settings
from app.db import Session
from app.models import Chat, Message
from app.spend import Spend

logger = logging.getLogger(__name__)

#: What a chat is called until its first question names it.
NEW_CHAT = "Новый чат"

#: How much of the first question becomes the title.
TITLE_CHARS = 80


@dataclass(frozen=True, slots=True)
class Answer:
    """A finished turn."""

    text: str
    #: The sources the answer's own [n] markers point at.
    citations: list[dict[str, Any]]
    #: Every fragment the model was handed, cited or not, without its text.
    #: The denominator of precision: eight shown and two cited is a different
    #: search from three shown and two cited.
    shown: list[dict[str, Any]]
    #: What the agent did on the way, in order.
    steps: list[dict[str, Any]]
    #: Tokens and dollars — see :mod:`app.spend`.
    usage: dict[str, Any]
    #: Wall clock, which for a turn is nearly all provider latency.
    took_ms: int


async def owned_chat(session: AsyncSession, user_id: uuid.UUID, chat_id: uuid.UUID) -> Chat:
    """The chat, or 404.

    Not 403: whether a chat exists is itself something only its owner should
    learn, and two different answers here would say which ids are real.
    """
    chat = (
        await session.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
    ).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Чат не найден")
    return chat


async def open_turn(
    session: AsyncSession,
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    question: str,
    window: int,
) -> list[Message]:
    """Record the question and hand back the history it was asked against.

    In that order, and read before the write: the history the model sees is the
    conversation up to this question, and inserting first would put the question
    in its own context twice.
    """
    chat = await owned_chat(session, user_id, chat_id)
    history = (
        (
            await session.execute(
                select(Message)
                .where(Message.chat_id == chat_id)
                .order_by(Message.created_at.desc())
                .limit(window)
            )
        )
        .scalars()
        .all()
    )
    session.add(Message(chat_id=chat_id, role="user", content=question, citations=[]))
    if chat.title == NEW_CHAT:
        chat.title = question[:TITLE_CHARS]
    await session.commit()
    return list(reversed(history))


def as_history(turns: Sequence[tuple[str, str]]) -> list[Message]:
    """History a caller keeps itself, in the shape the agent reads.

    These rows are never added to a session — they exist to carry a role and a
    string, which is all the agent asks of history. Building them as the same
    type the stored path produces is what keeps one code path behind both.
    """
    return [Message(role=role, content=content, citations=[]) for role, content in turns]


def trace(
    steps: list[dict[str, Any]],
    shown: list[dict[str, Any]],
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """What the turn did, in the shape it is stored in.

    Written once here rather than at each call site: three of them save an
    answer, and a trace that means different things in different rows is worse
    than no trace at all.

    Deliberately not the answer's own text or its fragments' — those are the
    two columns next to it. This is the part that used to be computed and
    thrown away: which tools ran with which queries, what the provider
    charged, and how many fragments the model was actually given.
    """
    return {"steps": steps, "shown": shown, "usage": usage}


async def save_answer(
    chat_id: uuid.UUID,
    text: str,
    citations: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
) -> uuid.UUID | None:
    """Keep whatever was produced — a partial answer beats a lost turn.

    Its own session, opened here and closed here: a turn is far too long to
    hold a pooled connection for, so nothing is held across it.

    Returns the id of the row, or None if there was nothing to write or the
    write failed. A failure must never replace the answer the caller is
    reading, so it is logged and swallowed.
    """
    if not text:
        return None
    try:
        async with Session() as session:
            message = Message(
                chat_id=chat_id,
                role="assistant",
                content=text,
                citations=citations,
                trace=trace,
            )
            session.add(message)
            await session.commit()
            return message.id
    except Exception:
        logger.exception("could not save the answer for chat %s", chat_id)
        return None


@dataclass
class Collected:
    """A turn being assembled from the agent's events."""

    parts: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    shown: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def take(self, kind: str, payload: Any) -> None:
        if kind == "token":
            self.parts.append(payload)
        elif kind == "citations":
            self.citations = payload
        elif kind == "shown":
            self.shown = payload
        elif kind == "step":
            self._step(payload)

    def _step(self, payload: dict[str, Any]) -> None:
        """One tool call, announced twice and recorded once.

        The stream says a tool's name the moment it is chosen and its query a
        few hundred milliseconds later, because that is the earliest each is
        known — good for a screen, noise in a list of what happened. So the
        second announcement replaces the first rather than following it.
        """
        if (
            self.steps
            and self.steps[-1]["tool"] == payload.get("tool")
            and not self.steps[-1]["query"]
        ):
            self.steps[-1] = dict(payload)
            return
        self.steps.append(dict(payload))


async def stream(
    settings: Settings,
    user_id: uuid.UUID,
    question: str,
    history: list[Message],
    web: bool,
    collected: Collected,
    spend: Spend | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """The agent's events, collected on the way past.

    The caller gets to relay them and, when the iteration ends however it ends,
    finds everything that arrived in ``collected``. ``spend`` is optional: the
    browser's chat does not report a bill, and an accumulator nobody reads is
    two dictionary lookups per model call.
    """
    async for kind, payload in agent.answer(settings, user_id, question, history, web, spend):
        collected.take(kind, payload)
        yield kind, payload


async def run(
    settings: Settings,
    user_id: uuid.UUID,
    question: str,
    history: list[Message],
    web: bool = False,
) -> Answer:
    """The whole turn, for a caller that wants one answer rather than a stream.

    The agent streams either way — that is the only mode it has — so this is
    the same run with nobody watching it happen. Exceptions are not caught
    here: a caller that has promised a JSON answer has to turn a failure into a
    status code, and there is no partial answer worth returning from a call
    that was supposed to return one.
    """
    spend = Spend()
    collected = Collected()
    started = time.perf_counter()
    async for _ in stream(settings, user_id, question, history, web, collected, spend):
        pass
    return Answer(
        text=collected.text,
        citations=collected.citations,
        shown=collected.shown,
        steps=collected.steps,
        usage=spend.report(),
        took_ms=round((time.perf_counter() - started) * 1000),
    )
