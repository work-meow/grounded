"""The public API: this service, called by another program.

Everything under ``/api`` was written for one caller — our own frontend, which
ships in the same commit and can be changed with it. That freedom is exactly
what an external integration must not inherit: a field renamed to suit a
component would break somebody's job at three in the morning.

So this is a second, deliberate surface. ``/api/v1`` is a contract: what it
returns keeps its shape, and a change that cannot keep it becomes ``/api/v2``.
Underneath, it calls the same functions the frontend does — there is one
implementation of a search and one of a turn, and this module is thin on
purpose. Where a response is the same shape as the frontend's, it is literally
the same model, and the day the two need to differ is the day this file grows
its own.

Two things live here rather than being borrowed:

* **An answer without a stream.** A program that wants one answer should not
  have to parse server-sent events to get it, and a program that wants tokens
  as they are written should not have to wait for the last one. ``stream``
  picks. The final SSE event is the same object the non-streaming call returns,
  so a client can ignore the tokens and read ``done`` — the concatenation bug
  that costs everybody an afternoon is not available.
* **The bill.** Every paid call in a turn is counted and priced (see
  :mod:`app.spend`), because a service that answers questions for money is one
  whose caller needs to know what a question costs.

Authentication is the same JWT the browser uses, in ``Authorization: Bearer``.
Mint one with ``uv run rag-token --days 365``.

ponytail: no rate limiting and no per-token quota. Ceiling: a token is a
personal credential on a personal install, and one turn is already bounded —
by the per-tool call limits and by ``agent_timeout_s``, so a single request
cannot run away with the bill. What is unbounded is *requests*, so a leaked
token could spend real money in a loop. Upgrade path: a counter in Postgres
keyed by the token's subject, checked in the dependency, once this is exposed
to anything but its owner.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from app import conversation
from app.config import Settings
from app.db import Session
from app.deps import SessionDep, SettingsDep, UserDep
from app.models import Message
from app.routers import chats as chats_router
from app.routers import connectors as connectors_router
from app.routers import search as search_router
from app.routers import sources as sources_router
from app.spend import Spend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])

#: How many turns of caller-supplied history are accepted. Well past the window
#: the agent is given anyway (``history_window``), so the cap is a bound on the
#: request body rather than a limit anyone meets.
MAX_HISTORY_TURNS = 200


class Who(BaseModel):
    user_id: uuid.UUID


class Turn(BaseModel):
    """One earlier message, for a caller keeping its own history."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class AnswerIn(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    #: Server-sent events as the answer is written, instead of one JSON object
    #: when it is finished. Same content either way.
    stream: bool = False
    #: Whether the agent may search the open web for this question. Off by
    #: default because it costs about $0.00125 a search, and per request rather
    #: than per token because the caller is the one who knows whether the answer
    #: is in the documents or in today's news.
    web: bool = False
    #: Continue a stored conversation: its history is loaded, and both the
    #: question and the answer are saved to it. Create one with POST /chats.
    chat_id: uuid.UUID | None = None
    #: Or keep the history yourself and pass it in. Nothing is stored then —
    #: which is what a stateless integration usually wants.
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @model_validator(mode="after")
    def _one_source_of_history(self) -> "AnswerIn":
        # Both would be two conversations claiming to be one. Refused rather
        # than merged or silently preferred, because either choice would
        # sometimes answer against context the caller did not send.
        if self.chat_id is not None and self.history:
            raise ValueError("chat_id и history вместе не имеют смысла — выберите одно")
        return self


class Citation(BaseModel):
    """A source the answer's own [n] marker points at."""

    n: int
    #: Set for a fragment from the knowledge base; null for a page from the web.
    document_id: str | None = None
    #: The file's name, or the page's title.
    filename: str | None = None
    page: int | None = None
    #: Set for a page from the web. Documents are opened through
    #: GET /documents/{document_id}/link, which knows how to sign one.
    url: str | None = None
    snippet: str = ""


class Step(BaseModel):
    """Something the agent did before answering."""

    tool: str
    query: str = ""


class StageUsage(BaseModel):
    #: "answer" | "rerank" | "web_search"
    stage: str
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class Usage(BaseModel):
    """What the request used, and what the provider charged for it."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    #: US dollars, summed from what the provider reported per call — not
    #: estimated from a price table.
    cost_usd: float
    #: False when some call did not report a price, which makes ``cost_usd`` a
    #: floor rather than the bill.
    cost_complete: bool
    calls: int
    #: The same numbers split by what they were spent on. Usually the
    #: interesting part: one web search costs more than the answer.
    stages: list[StageUsage]


class AnswerOut(BaseModel):
    answer: str
    citations: list[Citation]
    #: In order. Two searches here mean the first came back with nothing useful.
    steps: list[Step]
    usage: Usage
    took_ms: int
    #: Both null unless the question was asked into a stored chat.
    chat_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None


@router.get("/me")
async def me(user_id: UserDep) -> Who:
    """Whose token this is. The cheapest way to check one is valid."""
    return Who(user_id=user_id)


@router.post("/answer", response_model=AnswerOut)
async def answer(body: AnswerIn, user_id: UserDep, settings: SettingsDep) -> Any:
    """Answer a question from the knowledge base.

    With ``stream: false`` (the default) the response is one ``AnswerOut``.
    With ``stream: true`` it is ``text/event-stream``: ``step`` as the agent
    works, ``token`` as the answer is written, ``citations``, ``usage``, and a
    final ``done`` carrying the whole ``AnswerOut``. A turn that fails mid-
    stream sends ``error`` and no ``done``.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой вопрос")

    if body.chat_id is None:
        history = conversation.as_history(
            [(turn.role, turn.content) for turn in body.history][-settings.history_window :]
        )
    else:
        # Its own session, opened and closed before the turn begins: a turn
        # takes seconds, and a pooled connection held across one is a
        # connection the rest of the process cannot have. The stateless case
        # opens none at all — it has nothing to read and nothing to record.
        async with Session() as session:
            history = await conversation.open_turn(
                session, user_id, body.chat_id, question, settings.history_window
            )

    if body.stream:
        return StreamingResponse(
            _events(settings, user_id, question, history, body.web, body.chat_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await conversation.run(settings, user_id, question, history, body.web)
    except TimeoutError as exc:
        # The turn's own ceiling (agent_timeout_s), which is a statement about
        # the provider rather than about this request being wrong.
        logger.warning("the turn timed out for user %s", user_id)
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "Модель не ответила за отведённое время"
        ) from exc
    except Exception as exc:
        # The reason goes to the log and not over the wire: an httpx or driver
        # error carries internal hosts, and sometimes an API key, in its text.
        logger.exception("the turn failed for user %s", user_id)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Не удалось получить ответ. Попробуйте ещё раз."
        ) from exc

    stored = await _persist(
        body.chat_id, result, conversation.trace(result.steps, result.shown, result.usage)
    )
    return _out(result, body.chat_id, stored)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _events(
    settings: Settings,
    user_id: uuid.UUID,
    question: str,
    history: list[Message],
    web: bool,
    chat_id: uuid.UUID | None,
) -> AsyncIterator[str]:
    """The same turn, relayed as it happens.

    The save sits in ``finally`` for the reason it does everywhere here: a
    client that disconnects does not raise ``Exception`` — the generator is
    thrown ``GeneratorExit`` — and an ``except Exception`` around the loop would
    let a half-written answer disappear. Awaiting during ``aclose()`` is
    allowed; yielding is not, which is why ``done`` stays outside.
    """
    spend = Spend()
    collected = conversation.Collected()
    started = time.perf_counter()
    failed = False
    try:
        async for kind, payload in conversation.stream(
            settings, user_id, question, history, web, collected, spend
        ):
            if kind in ("token", "step", "citations"):
                yield _sse(kind, payload)
    except Exception:
        failed = True
        logger.exception("the streamed turn failed for user %s", user_id)
        yield _sse("error", "Не удалось получить ответ. Попробуйте ещё раз.")
    finally:
        message_id = await _persist(
            chat_id,
            collected,
            conversation.trace(collected.steps, collected.shown, spend.report()),
        )

    if failed:
        return
    result = conversation.Answer(
        text=collected.text,
        citations=collected.citations,
        shown=collected.shown,
        steps=collected.steps,
        usage=spend.report(),
        took_ms=round((time.perf_counter() - started) * 1000),
    )
    yield _sse("usage", result.usage)
    # Everything again, in one object: a client that ignored the tokens has the
    # whole answer here, and one that did not can check what it assembled.
    yield _sse("done", _out(result, chat_id, message_id).model_dump(mode="json"))


async def _persist(
    chat_id: uuid.UUID | None,
    result: conversation.Answer | conversation.Collected,
    trace: dict[str, Any],
) -> uuid.UUID | None:
    """Save the answer, if this question belonged to a stored chat.

    Takes either shape because a streamed turn only ever has the half-assembled
    one: both carry the text and the citations, which is all a row needs
    besides the trace built by the caller that owns the bill.
    """
    if chat_id is None:
        return None
    return await conversation.save_answer(chat_id, result.text, result.citations, trace)


def _out(
    result: conversation.Answer, chat_id: uuid.UUID | None, message_id: uuid.UUID | None
) -> AnswerOut:
    return AnswerOut(
        answer=result.text,
        citations=[Citation(**item) for item in result.citations],
        steps=[Step(**step) for step in result.steps],
        usage=Usage(**result.usage),
        took_ms=result.took_ms,
        chat_id=chat_id,
        message_id=message_id,
    )


# --- the rest is the frontend's own API, under a contract ---------------------
#
# Delegation rather than reimplementation: one search, one upload, one list of
# documents. What the version buys is that these signatures are now promises,
# and the handlers behind them can be refactored without breaking one.


@router.get("/search")
async def find(
    user_id: UserDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=500, description="Поисковый запрос")],
    limit: Annotated[int, Query(ge=1, le=50)] = search_router.DEFAULT_LIMIT,
    source: Annotated[uuid.UUID | None, Query(description="Только в этом источнике")] = None,
    days: Annotated[int | None, Query(ge=1, le=3650, description="Изменённое за N дней")] = None,
) -> search_router.SearchOut:
    """Fragments matching a query, with no model involved.

    A tenth of a second and no tokens, against seconds and a bill through
    /answer. This says where the words are; it never explains or summarises.
    """
    return await search_router.find(
        user_id=user_id, settings=settings, q=q, limit=limit, source=source, days=days
    )


@router.get("/documents")
async def list_documents(
    user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> list[sources_router.DocumentOut]:
    """Everything searchable for this user, uploads and connected sources alike.

    ``status`` is read live from the index, so "ready" means the document can
    actually be found rather than that it was accepted.
    """
    return await sources_router.list_documents(user_id=user_id, session=session, settings=settings)


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> sources_router.DocumentOut:
    """Upload a file (multipart/form-data).

    It comes back as ``processing`` and becomes searchable on its own, usually
    within seconds. Formats are decided by suffix; an unsupported one is a 415
    rather than a document that would sit at ``processing`` forever.
    """
    return await sources_router.upload(
        file=file, user_id=user_id, session=session, settings=settings
    )


@router.get("/documents/{document_id}/link")
async def document_link(
    document_id: uuid.UUID,
    user_id: UserDep,
    session: SessionDep,
    settings: SettingsDep,
    page: Annotated[int | None, Query(ge=1, le=10_000, description="Открыть на странице")] = None,
    quote: Annotated[
        str | None, Query(max_length=400, description="Открыть на этих словах")
    ] = None,
) -> dict[str, str]:
    """Where to open the original: a presigned link, or the service's own page.

    Pass a citation's `page` or `snippet` back and the link opens there —
    `#page=3` for a PDF, a text fragment for markdown and plain text. Both are
    hints a viewer may ignore; neither can make the link worse.
    """
    return await sources_router.document_link(
        document_id=document_id,
        user_id=user_id,
        session=session,
        settings=settings,
        page=page,
        quote=quote,
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(
    document_id: uuid.UUID, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> None:
    """Delete an upload. A document from a connected source is a 404 — it is not
    ours to delete, and the next poll would bring it straight back."""
    await sources_router.remove(
        document_id=document_id, user_id=user_id, session=session, settings=settings
    )


@router.get("/sources")
async def list_sources(
    user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> list[connectors_router.ConnectorOut]:
    """Connected sources, each with what the indexer last saw in it.

    ``status`` is "unknown" until the indexer has looked once — never "ok",
    because a source nobody has read is not a working one.
    """
    listing = await connectors_router.list_connectors(
        user_id=user_id, session=session, settings=settings
    )
    return listing.sources


@router.get("/chats")
async def list_chats(user_id: UserDep, session: SessionDep) -> list[chats_router.ChatOut]:
    return await chats_router.list_chats(user_id=user_id, session=session)


@router.post("/chats", status_code=status.HTTP_201_CREATED)
async def create_chat(user_id: UserDep, session: SessionDep) -> chats_router.ChatOut:
    """Start a stored conversation, to pass as ``chat_id`` to /answer."""
    return await chats_router.create_chat(user_id=user_id, session=session)


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: uuid.UUID, user_id: UserDep, session: SessionDep) -> None:
    await chats_router.delete_chat(chat_id=chat_id, user_id=user_id, session=session)


@router.get("/chats/{chat_id}/messages")
async def list_messages(
    chat_id: uuid.UUID,
    user_id: UserDep,
    session: SessionDep,
    before: Annotated[str | None, Query(description="Курсор next_cursor: страница старше")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = chats_router.MESSAGE_PAGE,
) -> chats_router.MessagesOut:
    """The end of a chat, or the page before ``before``. Oldest message first."""
    return await chats_router.list_messages(
        chat_id=chat_id, user_id=user_id, session=session, before=before, limit=limit
    )
