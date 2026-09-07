"""The same service, speaking OpenAI's chat-completions dialect.

Not because that protocol fits — this is a retrieval service, not a model, and
``messages[]`` has nowhere to put a source filter or a citation. It is here
because of what already speaks it: the official SDKs in every language, n8n,
LangChain, LlamaIndex, LibreChat, Open WebUI, editor plugins. Pointing any of
them at ``base_url`` costs the integrator one line, and this file is what makes
that line work.

    client = OpenAI(base_url="https://host/api/v1", api_key="<jwt>")

The native API next door (``/api/v1/answer``) is the honest one and stays the
place where new capability lands. This is a translation, and translations lose
things. What it loses, deliberately:

* **A system message from the caller is ignored.** The prompt here is what
  keeps answers grounded and citations numbered; letting a caller replace it
  would quietly turn this into a general-purpose model that happens to have
  read some documents.
* **Sampling parameters are accepted and ignored.** ``temperature``,
  ``top_p``, ``max_tokens`` and friends belong to a model call; a turn here is
  several of those plus retrieval, and honouring them halfway would be worse
  than not at all.
* **Citations arrive in a non-standard field**, alongside ``choices``, the way
  Perplexity and OpenRouter extend the same shape. The answer's own [1] and [2]
  markers are in the text, so a client that ignores the field still shows a
  coherent answer.

What it gains is the one option a fixed protocol can still carry: the model
name. ``grounded`` answers from the documents; ``grounded-web`` may also search
the open web. That is the whole reason two names exist.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app import conversation
from app.config import Settings
from app.deps import SettingsDep, UserDep
from app.models import Message
from app.spend import Spend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["openai"])

#: Answers from the user's documents.
MODEL = "grounded"
#: The same, allowed to search the open web when the documents come up empty.
MODEL_WEB = "grounded-web"

#: When this build started. ``created`` is required by the shape and means
#: nothing here; the process start is at least true.
_STARTED = int(time.time())

#: How much of a message list is read as history. The agent is given
#: ``history_window`` of it; this is the bound on the request body.
MAX_MESSAGES = 200


class ChatMessage(BaseModel):
    # Extra keys are ignored rather than refused: SDKs send `name`, `tool_calls`
    # and whatever the next version adds, and a 422 for a field we do not read
    # would make this useless with the very clients it exists for.
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    #: A string, or OpenAI's list of content parts. Null for a message that
    #: carried only tool calls.
    content: str | list[dict[str, Any]] | None = None


class CompletionIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = MODEL
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    stream: bool = False


@router.get("/models")
async def models(user_id: UserDep) -> dict[str, Any]:
    """The two names this endpoint answers to.

    ``user_id`` is unused and required on purpose: an SDK lists models to check
    its credentials, and a listing that answered without one would report a
    dead token as healthy.
    """
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": _STARTED, "owned_by": "grounded"}
            for name in (MODEL, MODEL_WEB)
        ],
    }


@router.post("/chat/completions")
async def completions(body: CompletionIn, user_id: UserDep, settings: SettingsDep) -> Any:
    """A chat completion, answered from the user's knowledge base.

    ``model`` is ``grounded`` or ``grounded-web``; the latter may search the
    open web. ``stream`` behaves as OpenAI's does, ending with a usage chunk
    and ``data: [DONE]``.
    """
    web = _web(body.model)
    question, history = _split(body.messages, settings)
    if not question:
        raise _error(
            status.HTTP_400_BAD_REQUEST,
            "последнее сообщение должно быть от пользователя и не быть пустым",
            "invalid_request_error",
        )

    if body.stream:
        return StreamingResponse(
            _chunks(settings, user_id, question, history, web, body.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await conversation.run(settings, user_id, question, history, web)
    except TimeoutError as exc:
        raise _error(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "модель не ответила за отведённое время",
            "timeout",
        ) from exc
    except Exception as exc:
        # The reason to the log, never to the caller: these carry internal
        # hosts and sometimes a key.
        logger.exception("a chat completion failed for user %s", user_id)
        raise _error(
            status.HTTP_502_BAD_GATEWAY, "не удалось получить ответ", "upstream_error"
        ) from exc

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(result.usage),
        "citations": result.citations,
    }


def _web(model: str) -> bool:
    """Whether this name asks for the web, or is not a name we answer to."""
    if model in (MODEL, MODEL_WEB):
        return model == MODEL_WEB
    raise _error(
        status.HTTP_404_NOT_FOUND,
        f"нет такой модели: {model}. Доступны: {MODEL}, {MODEL_WEB}",
        "model_not_found",
    )


def _split(messages: list[ChatMessage], settings: Settings) -> tuple[str, list[Message]]:
    """The question, and the conversation before it.

    The last user message is the question — not the last message: a client that
    replays a trailing assistant turn is asking for a continuation of a
    conversation whose question is still the one above it. Everything before
    that becomes history; system and tool messages are dropped, because this
    service's own prompt is not something a caller replaces.
    """
    turns = [
        (message.role, text)
        for message in messages
        if message.role in ("user", "assistant") and (text := _text(message.content))
    ]
    for index in range(len(turns) - 1, -1, -1):
        if turns[index][0] == "user":
            return turns[index][1], conversation.as_history(
                turns[:index][-settings.history_window :]
            )
    return "", []


def _text(content: str | list[dict[str, Any]] | None) -> str:
    """One message's text, from either shape OpenAI allows."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(part.get("text") or "").strip()
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def _usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Usage in OpenAI's three fields, plus the price.

    ``cost`` is not OpenAI's — it is OpenRouter's extension to the same object,
    which is where anyone reading a cost off a chat completion already looks.
    The full breakdown by stage is on /api/v1/answer.
    """
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost": usage["cost_usd"],
    }


def _id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _error(code: int, message: str, kind: str) -> HTTPException:
    """An error in the shape an OpenAI client knows how to read.

    ``{"error": {...}}`` rather than FastAPI's ``{"detail": ...}``, which is
    what ``main.py`` rewrites for this path. Built as a dict detail so that one
    handler can do it for everything raised under here, dependencies included.
    """
    return HTTPException(code, {"message": message, "type": kind})


async def _chunks(
    settings: Settings,
    user_id: uuid.UUID,
    question: str,
    history: list[Message],
    web: bool,
    model: str,
) -> AsyncIterator[str]:
    """The answer as OpenAI streams one.

    A first chunk with the role, a chunk per piece of text, a chunk with
    ``finish_reason``, a usage chunk with no choices, and ``[DONE]``. The usage
    chunk is what OpenAI sends under ``stream_options.include_usage``; it is
    sent unconditionally here, because a client that does not expect it ignores
    a chunk with an empty ``choices`` list, and a caller who is paying should
    not have to ask what for.
    """
    completion_id = _id()
    spend = Spend()
    collected = conversation.Collected()

    def envelope(**extra: Any) -> str:
        return _data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                **extra,
            }
        )

    yield envelope(choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
    try:
        async for kind, payload in conversation.stream(
            settings, user_id, question, history, web, collected, spend
        ):
            if kind == "token":
                yield envelope(
                    choices=[{"index": 0, "delta": {"content": payload}, "finish_reason": None}]
                )
    except Exception:
        # There is no error frame in this protocol once the stream has started.
        # OpenAI's own answer to that is `finish_reason` — "length" for a cut
        # answer — and an incomplete one is the honest reading here: the client
        # keeps what arrived and can see it did not finish.
        logger.exception("a streamed chat completion failed for user %s", user_id)
        yield envelope(choices=[{"index": 0, "delta": {}, "finish_reason": "length"}])
        yield _data("[DONE]")
        return

    yield envelope(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
    yield envelope(choices=[], usage=_usage(spend.report()), citations=collected.citations)
    yield _data("[DONE]")


def _data(payload: Any) -> str:
    """One SSE frame in OpenAI's format: no event name, just data.

    ``[DONE]`` is the one payload that is not JSON, which is why this takes a
    string as well.
    """
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"
