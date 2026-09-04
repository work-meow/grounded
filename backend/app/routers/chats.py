import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent
from app.config import Settings
from app.db import Session
from app.deps import SessionDep, SettingsDep, UserDep
from app.models import Chat, Message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chats", tags=["chats"])


class ChatOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


@router.get("")
async def list_chats(user_id: UserDep, session: SessionDep) -> list[ChatOut]:
    rows = (
        (
            await session.execute(
                select(Chat).where(Chat.user_id == user_id).order_by(Chat.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [ChatOut(id=c.id, title=c.title, created_at=c.created_at) for c in rows]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat(user_id: UserDep, session: SessionDep) -> ChatOut:
    chat = Chat(user_id=user_id)
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return ChatOut(id=chat.id, title=chat.title, created_at=chat.created_at)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: uuid.UUID, user_id: UserDep, session: SessionDep) -> None:
    await _owned_chat(session, user_id, chat_id)
    await session.execute(sql_delete(Chat).where(Chat.id == chat_id))
    await session.commit()


@router.get("/{chat_id}/messages")
async def list_messages(
    chat_id: uuid.UUID, user_id: UserDep, session: SessionDep
) -> list[MessageOut]:
    await _owned_chat(session, user_id, chat_id)
    rows = (
        (
            await session.execute(
                select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at)
            )
        )
        .scalars()
        .all()
    )
    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            citations=m.citations,
            created_at=m.created_at,
        )
        for m in rows
    ]


@router.post("/{chat_id}/messages")
async def ask(
    chat_id: uuid.UUID, body: AskRequest, user_id: UserDep, settings: SettingsDep
) -> StreamingResponse:
    """Answer a question, streaming tokens over SSE.

    Sessions are opened and closed around the stream rather than held for its
    whole duration — an LLM turn is far too long to occupy a pooled connection.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой вопрос")

    async with Session() as session:
        chat = await _owned_chat(session, user_id, chat_id)
        history = (
            (
                await session.execute(
                    select(Message)
                    .where(Message.chat_id == chat_id)
                    .order_by(Message.created_at.desc())
                    .limit(settings.history_window)
                )
            )
            .scalars()
            .all()
        )
        session.add(Message(chat_id=chat_id, role="user", content=question, citations=[]))
        if chat.title == "Новый чат":
            chat.title = question[:80]
        await session.commit()

    return StreamingResponse(
        _stream(settings, user_id, chat_id, question, list(reversed(history))),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream(
    settings: Settings,
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    question: str,
    history: list[Message],
) -> AsyncIterator[str]:
    """Relay the agent's events as SSE, saving the answer however the turn ends.

    The save sits in ``finally`` because a closed browser tab does not raise
    ``Exception`` — Starlette throws ``GeneratorExit`` into this generator, and
    an ``except Exception`` around the loop would let the tokens already
    produced disappear. Awaiting during ``aclose()`` is allowed; yielding is
    not, which is why the ``done`` event stays outside.
    """
    parts: list[str] = []
    citations: list[dict[str, Any]] = []
    try:
        async for kind, payload in agent.answer(settings, user_id, question, history):
            if kind == "token":
                parts.append(payload)
                yield _sse("token", payload)
            elif kind == "citations":
                citations = payload
                yield _sse("citations", payload)
    except Exception:
        # The client must learn the turn failed; a bare 500 mid-stream would
        # just look like the answer stopped. The reason goes to the log, not
        # over the wire: an httpx or driver error carries internal hosts and
        # keys in its text.
        logger.exception("agent run failed for chat %s", chat_id)
        yield _sse("error", "Не удалось получить ответ. Попробуйте ещё раз.")
    finally:
        await _save_answer(chat_id, "".join(parts), citations)

    yield _sse("done", {"citations": citations})


async def _save_answer(chat_id: uuid.UUID, text: str, citations: list[dict[str, Any]]) -> None:
    """Keep whatever was produced — a partial answer beats a lost turn."""
    if not text:
        return
    try:
        async with Session() as session:
            session.add(
                Message(chat_id=chat_id, role="assistant", content=text, citations=citations)
            )
            await session.commit()
    except Exception:
        # Never let a write failure replace the answer the user is reading.
        logger.exception("could not save the answer for chat %s", chat_id)


async def _owned_chat(session: AsyncSession, user_id: uuid.UUID, chat_id: uuid.UUID) -> Chat:
    chat = (
        await session.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
    ).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Чат не найден")
    return chat
