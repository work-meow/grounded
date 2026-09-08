"""Other fragments about the same thing as this one.

From a search hit or a citation: «где ещё об этом написано». No new index
capability was needed — the fragment's own text is the best query for finding
things like it, which is what a vector index is for.

The property worth testing is the exclusion. A document is more related to
itself than to anything else, so without it the answer to "where else" is
"four more paragraphs of the page you are already reading" — which is not an
answer to that question at all.
"""

from datetime import timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import retriever
from app.config import get_settings
from app.main import app
from app.security import issue_token

TEXT = "Основной ежегодный отпуск — 28 календарных дней в год"


def chunk(document: str, page: int | None = None, text: str = TEXT) -> retriever.Chunk:
    return retriever.Chunk(
        text=text, score=0.9, document_id=document, filename=f"{document}.md", page=page
    )


@pytest.fixture
def client():
    token = issue_token(get_settings(), uuid4(), timedelta(minutes=5))
    with TestClient(app, base_url="https://testserver") as opened:
        opened.headers["Authorization"] = f"Bearer {token}"
        yield opened


@pytest.fixture
def index(monkeypatch):
    def serve(chunks, *, fails: bool = False):
        async def retrieve(_settings, _user, query, _k, **_narrowing):
            if fails:
                raise httpx.ConnectError("index is down")
            serve.asked = query
            return chunks

        monkeypatch.setattr(retriever, "retrieve", retrieve)

    serve.asked = None
    return serve


def test_the_fragments_own_text_is_the_query(client, index):
    index([chunk("другой")])

    client.get("/api/v1/related", params={"text": TEXT})

    assert index.asked == TEXT


def test_the_document_it_came_from_is_left_out(client, index):
    """A document is more related to itself than to anything else, so without
    this the answer to "where else" is the page you are already reading."""
    same = str(uuid4())
    other = str(uuid4())
    index([chunk(same, page=1), chunk(same, page=2), chunk(other)])

    body = client.get(
        "/api/v1/related", params={"text": TEXT, "document_id": same, "limit": 5}
    ).json()

    assert [hit["document_id"] for hit in body["hits"]] == [other]


def test_one_fragment_per_document(client, index):
    """The question is which other documents talk about this; five fragments
    of one document is not that answer."""
    index([chunk("а", page=1), chunk("а", page=2), chunk("б"), chunk("в")])

    body = client.get("/api/v1/related", params={"text": TEXT}).json()

    assert [hit["document_id"] for hit in body["hits"]] == ["а", "б", "в"]


def test_the_words_of_the_fragment_are_marked_in_what_comes_back(client, index):
    index([chunk("другой", text="Отпуск переносится по согласованию")])

    body = client.get(
        "/api/v1/related", params={"text": "отпуск переносится по согласованию"}
    ).json()

    marked = [piece["text"] for piece in body["hits"][0]["snippet"] if piece["hit"]]
    assert marked


def test_an_index_that_is_rebuilding_says_so(client, index):
    """503 and not 500: adding a source restarts the indexer on purpose, and a
    page that says "перестраивается" is the difference between a wait and a
    bug report."""
    index([], fails=True)

    assert client.get("/api/v1/related", params={"text": TEXT}).status_code == 503


def test_a_snippet_too_short_to_be_a_query_is_refused(client, index):
    """Three words match everything; the endpoint exists for a fragment."""
    index([chunk("другой")])

    assert client.get("/api/v1/related", params={"text": "отпуск"}).status_code == 422


def test_nothing_is_charged_for_it(client, index):
    """No model is involved: it is one request to a local index, and it should
    not appear on anybody's bill or count against a daily ceiling."""
    index([chunk("другой")])

    body = client.get("/api/v1/related", params={"text": TEXT}).json()

    assert "usage" not in body
