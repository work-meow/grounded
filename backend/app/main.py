"""The API process: auth, documents, connected sources, chats, and the public API.

Two surfaces, on purpose. Everything under ``/api`` was written for our own
frontend, which ships in the same commit and can be changed with it.
``/api/v1`` is the contract external callers get — see ``app/routers/v1.py``.

The interactive documentation lives under the same prefix (``/api/docs``,
``/api/openapi.json``) rather than at the root. Not cosmetic: Traefik routes
``/api`` here and everything else to the frontend, so ``/docs`` at the root was
served by Next.js and 404'd in the deployment — working only locally, which is
the one place it was least needed.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import http
from app.config import get_settings
from app.db import engine
from app.routers import auth, chats, compat, connectors, search, sources, v1

DESCRIPTION = """\
Отвечает **только по вашим документам** — со ссылкой на файл и страницу.

* `POST /api/v1/answer` — спросить: одним JSON или потоком, со стоимостью запроса.
* `GET /api/v1/search` — найти фрагменты без модели: десятая доля секунды, ноль токенов.
* `POST /api/v1/chat/completions` — то же самое в формате OpenAI, для готовых SDK.

Авторизация — JWT в `Authorization: Bearer`, выпускается командой
`uv run rag-token --days 365`. Полное описание: [docs/api.md](https://github.com/work-meow/grounded/blob/main/docs/api.md).
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # try/finally, not a plain sequence: a failure during shutdown would
    # otherwise skip the teardown and leak the connection pool.
    async with httpx.AsyncClient() as client:
        http.set_client(client)
        try:
            yield
        finally:
            http.set_client(None)
            await engine.dispose()


settings = get_settings()

app = FastAPI(
    title="grounded",
    version="1.0.0",
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sources.router)
app.include_router(connectors.router)
app.include_router(chats.router)
app.include_router(search.router)
app.include_router(v1.router)
app.include_router(compat.router)

#: The two paths that must answer in OpenAI's error shape rather than
#: FastAPI's. Listed exactly rather than matched by prefix, because
#: ``/api/v1/chats`` starts with ``/api/v1/chat``.
_OPENAI_PATHS = frozenset({"/api/v1/chat/completions", "/api/v1/models"})


@app.exception_handler(StarletteHTTPException)
async def _errors(request: Request, exc: StarletteHTTPException) -> Response:
    """``{"detail": ...}`` everywhere, except where a client expects otherwise.

    An OpenAI client reads ``error.message`` and nothing else, so the compat
    endpoints answer in that shape — including for failures raised before their
    own code runs, which is why this is one handler here rather than a wrapper
    around each of them. Everything else keeps FastAPI's default, which the
    frontend already reads.
    """
    if request.url.path not in _OPENAI_PATHS:
        return await http_exception_handler(request, exc)

    detail: Any = exc.detail
    error = (
        dict(detail)
        if isinstance(detail, dict)
        else {"message": str(detail), "type": "invalid_request_error"}
    )
    if exc.status_code == 401:
        error["code"] = "invalid_api_key"
    return JSONResponse(
        {"error": {"param": None, "code": None, **error}},
        status_code=exc.status_code,
        headers=exc.headers,
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Liveness, not readiness: is this process answering at all.

    Deliberately touches neither PostgreSQL nor the indexer. Docker restarts a
    container that fails its healthcheck, and restarting the API because the
    database blinked would turn one outage into two.
    """
    return {"status": "ok"}
