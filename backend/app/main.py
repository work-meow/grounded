"""The API process: auth, documents, connected sources, chats.

Everything it serves lives under /api, which is also how Traefik tells it apart
from the frontend on the same hostname. One consequence worth knowing: /docs and
/openapi.json sit outside that prefix, so in the deployed setup they are routed
to the frontend and 404. They work locally, which is where they are useful.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import retriever
from app.config import get_settings
from app.db import engine
from app.routers import auth, chats, connectors, search, sources


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # try/finally, not a plain sequence: a failure during shutdown would
    # otherwise skip the teardown and leak the connection pool.
    async with httpx.AsyncClient() as client:
        retriever.set_client(client)
        try:
            yield
        finally:
            retriever.set_client(None)
            await engine.dispose()


settings = get_settings()

app = FastAPI(title="RAG", version="0.1.0", lifespan=lifespan)
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


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Liveness, not readiness: is this process answering at all.

    Deliberately touches neither PostgreSQL nor the indexer. Docker restarts a
    container that fails its healthcheck, and restarting the API because the
    database blinked would turn one outage into two.
    """
    return {"status": "ok"}
