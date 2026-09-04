from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import retriever
from app.config import get_settings
from app.db import engine
from app.routers import auth, chats, sources


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient() as client:
        retriever.set_client(client)
        yield
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
app.include_router(chats.router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
