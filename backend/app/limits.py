"""A ceiling on a request body, enforced before anything parses it.

Measured, not suspected: a POST of 40 MB of JSON to this API — with no token
at all — took resident memory from 290 MB to 520 MB, and a second one to
827 MB. FastAPI reads and parses the body *before* it solves dependencies, so
authentication happens after the memory is already spent, and a field with
``max_length=8000`` is validated against a string that has already been built.
On the deployment box — measured: 8 GB total, 4.6 GB of it available, other
people's services on the same host — a handful of concurrent requests is an
out-of-memory kill, from an attacker who needs no credentials.

There is no framework knob for this. Starlette and uvicorn do not limit body
size, and neither does Traefik unless a buffering middleware is configured. So
it lives here, as the outermost layer: below it, everything can go on assuming
a body it can afford to hold.

Two paths need more than the rest — the two that accept a file — and they get
exactly the upload limit plus enough slack for multipart framing.
"""

import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: Enough for the largest legitimate JSON body: two hundred chat messages of a
#: few thousand characters each, in a language where a character is two bytes.
#: Two orders of magnitude below the measurement above.
JSON_LIMIT = 4 * 1024 * 1024

#: Multipart adds a boundary, a couple of headers and the field name around the
#: file. A megabyte is far more than that and far less than anything that
#: matters next to a 64 MB upload.
MULTIPART_SLACK = 1024 * 1024

#: Only these carry a body worth measuring. A GET with one is malformed, and
#: whatever reads it would fail on its own.
_METHODS = frozenset({"POST", "PUT", "PATCH"})


class BodyLimit:
    """Reject an oversized body with 413, before the app is called."""

    def __init__(self, app: Any, *, default: int, uploads: int, upload_paths: frozenset[str]):
        self.app = app
        self.default = default
        self.uploads = uploads
        self.upload_paths = upload_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in _METHODS:
            await self.app(scope, receive, send)
            return

        limit = self.uploads if scope.get("path") in self.upload_paths else self.default
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            # The honest case: the client said how much it was about to send,
            # and it is too much. Nothing has been read yet.
            await _too_large(send, limit)
            return

        await self.app(scope, _counting(receive, limit), send)


def _counting(receive: Receive, limit: int) -> Receive:
    """``receive``, with a running total and a cut-off.

    For a request that declared no length — chunked, which any client may use
    and a malicious one certainly will. Once the total passes the limit the
    body ends as a disconnect rather than a 413: the application is already
    running by then, holding the response, and a client that went away is the
    one thing every layer above already knows how to abandon.
    """
    received = 0

    async def counted() -> MutableMapping[str, Any]:
        nonlocal received
        message = await receive()
        if message["type"] != "http.request":
            return message
        received += len(message.get("body") or b"")
        if received > limit:
            logger.warning("a request body passed %d bytes without declaring its length", limit)
            return {"type": "http.disconnect"}
        return message

    return counted


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers") or []:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                # A malformed length is not something to reason about; let the
                # server that parses the request reject it.
                return None
    return None


async def _too_large(send: Send, limit: int) -> None:
    body = json.dumps(
        {"detail": f"Тело запроса больше {limit // 1024 // 1024} МБ"}, ensure_ascii=False
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
                # Nothing about the body was read, so the connection cannot be
                # reused for the next request on it.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
