"""Finding something without asking the model about it.

The agent is the good way to get an answer and an expensive way to get a
location. "Где я это записывал" costs a couple of model calls and several
seconds through the chat, and one request to the index — a tenth of a second,
no tokens — through here.

Same index, same tenant filter, same ranking. What is missing on purpose is the
model: this endpoint never explains, never summarises and never invents. It
says where the words are.
"""

import time
import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app import expansion, relevance, retriever, search
from app.config import Settings
from app.deps import SettingsDep, UserDep

router = APIRouter(prefix="/api/search", tags=["search"])

#: Results per query. Twenty is what fits a scroll without turning the page
#: into the corpus; the index is asked for exactly this many.
DEFAULT_LIMIT = 20


class Piece(BaseModel):
    """A run of the snippet, marked or not.

    The split is done here rather than in the browser so the rule that decides
    what counts as a match lives in one place — and so the client never has to
    put anything but text into the DOM.
    """

    text: str
    hit: bool


class Hit(BaseModel):
    document_id: str | None
    filename: str | None
    page: int | None
    snippet: list[Piece]


class SearchOut(BaseModel):
    hits: list[Hit]
    #: How many the index put forward before anything judged them. Shown next to
    #: the number kept, because "3 из 20" says the search worked and the corpus
    #: has three things about this — where a bare "3" could be either.
    found: int
    #: The whole lookup, query embedding included — what the user actually
    #: waited. Shown, because a tenth of a second is the entire argument for
    #: this page existing next to the chat. It is the embedding that moves:
    #: measured on the deployment, 50 ms when the query is cached upstream and
    #: 400 ms when it is not.
    took_ms: int


class Related(BaseModel):
    hits: list[Hit]
    took_ms: int


@router.get("/related")
async def related(
    user_id: UserDep,
    settings: SettingsDep,
    text: Annotated[str, Query(min_length=20, max_length=2000, description="Текст фрагмента")],
    document_id: Annotated[uuid.UUID | None, Query(description="Исключить этот документ")] = None,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> Related:
    """Other fragments about the same thing as this one.

    From a search hit or a citation: "где ещё об этом написано". No new index
    capability is needed — /v1/retrieve takes a string and the fragment's own
    text is the best possible query for finding things like it, which is what
    a vector index is for.

    The fragment's own document is excluded rather than filtered out
    afterwards, because half a page of a long document is otherwise the whole
    answer: a document is related to itself more than to anything else.
    """
    started = time.perf_counter()
    try:
        found = await retriever.retrieve(
            settings, user_id, _as_query(text), settings.rerank_candidates
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Поиск сейчас недоступен, индекс перестраивается. Попробуйте через минуту.",
        ) from exc

    elsewhere = [chunk for chunk in found if str(chunk.document_id) != str(document_id)]
    # One fragment per document: this answers "which other documents talk
    # about this", and five fragments of one document is not that answer.
    chunks = relevance.cap_per_document(elsewhere, 1)[:limit]
    return Related(
        hits=[
            Hit(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                snippet=[
                    Piece(text=piece, hit=marked)
                    for piece, marked in search.snippet(chunk.text, text)
                ],
            )
            for chunk in chunks
        ],
        took_ms=round((time.perf_counter() - started) * 1000),
    )


#: Words of a fragment used as a query. A whole chunk is five hundred tokens
#: of prose, most of it beside the point of what the fragment is *about* — and
#: BM25 scores a long query by everything in it. The opening lines carry the
#: heading and the subject.
_QUERY_WORDS = 40


def _as_query(text: str) -> str:
    """A fragment reduced to something worth searching for.

    Sanitised by `retriever.searchable` on the way to the index in any case —
    a fragment is full of punctuation the query parser owns, and it crashed the
    engine before that existed. This shortens it as well, which is a matter of
    result quality rather than safety.
    """
    return " ".join(retriever.searchable(text).split()[:_QUERY_WORDS])


def _candidates(settings: Settings, limit: int) -> int:
    """How many to ask the index for before judging them."""
    return max(settings.rerank_candidates, limit) if settings.rerank_enabled else limit


@router.get("")
async def find(
    user_id: UserDep,
    settings: SettingsDep,
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=50),
    source: uuid.UUID | None = None,
    days: int | None = Query(None, ge=1, le=3650),
) -> SearchOut:
    """Fragments matching a query, narrowed to one source or to recent changes."""
    query = q.strip()
    if not query:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Пустой запрос")

    started = time.perf_counter()
    try:
        candidates = await expansion.search(
            settings,
            user_id,
            query,
            _candidates(settings, limit),
            source_id=source,
            since=retriever.since(days),
        )
    except httpx.HTTPError as exc:
        # The index being unreachable is not this request being wrong. A 500
        # here would read in the UI as "search is broken" rather than "the
        # indexer is restarting", which is what it usually is — adding a source
        # restarts it, and that takes a few seconds.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Поиск сейчас недоступен, индекс перестраивается. Попробуйте через минуту.",
        ) from exc

    # Judged, then thinned. The order is deliberate: relevance decides what is
    # worth showing, and only then does the page stop showing one document five
    # times — a list somebody is scanning for *where* something is written wants
    # five documents, which is the opposite of what an answer wants.
    chunks = relevance.cap_per_document(
        await relevance.keep_relevant(settings, query, candidates),
        settings.max_chunks_per_document,
    )[:limit]

    return SearchOut(
        found=len(candidates),
        hits=[
            Hit(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                snippet=[
                    Piece(text=text, hit=hit) for text, hit in search.snippet(chunk.text, query)
                ],
            )
            for chunk in chunks
        ],
        took_ms=round((time.perf_counter() - started) * 1000),
    )
