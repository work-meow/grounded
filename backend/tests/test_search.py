"""Search without the model.

Two things are being tested and they fail in different ways. The snippet is
cosmetic until it is wrong — a run dropped or duplicated silently rewrites what
the user reads, and the text is the one part of a search result nobody
double-checks against the source. And the endpoint is the first thing a person
touches when the indexer is restarting, which happens every time a source is
connected, so what it says then matters more than what it says when all is well.
"""

from datetime import timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import retriever, search
from app.config import get_settings
from app.main import app
from app.security import issue_token

#: Numbered, so a test can tell which end of the chunk it is looking at.
_LEAD = " ".join(f"Предложение номер {n}, ничего интересного." for n in range(1, 9))
_TAIL = " ".join(f"Хвост номер {n}, тоже неинтересный." for n in range(1, 9))
TEXT = f"{_LEAD} В вишлисте записано: купить айфон 10-11 и перепрошить редми. {_TAIL}"


def _visible(runs: list[tuple[str, bool]]) -> str:
    return "".join(text for text, _ in runs)


def _marked(runs: list[tuple[str, bool]]) -> list[str]:
    return [text for text, hit in runs if hit]


# --- the snippet -------------------------------------------------------------


def test_the_runs_reassemble_into_exactly_what_is_shown():
    """The invariant the whole shape rests on. A dropped or duplicated run
    silently rewrites the sentence the user reads."""
    runs = search.snippet(TEXT, "вишлист айфон")
    visible = _visible(runs)

    assert visible.strip("…") in " ".join(TEXT.split())
    assert "".join(_marked(runs)) == "вишлистеайфон"


def test_the_window_is_cut_around_the_match_not_from_the_top():
    """The match sits well into the chunk. A snippet taken from the top would
    show eight filler sentences and none of the reason the result is here."""
    visible = _visible(search.snippet(TEXT, "айфон"))

    assert "айфон" in visible
    assert "Предложение номер 1," not in visible, "the head was skipped, not shown"
    assert visible.startswith("…")


def test_a_short_chunk_is_shown_whole_without_ellipses():
    runs = search.snippet("Купить айфон 10-11.", "айфон")

    assert _visible(runs) == "Купить айфон 10-11."


def test_an_inflected_word_is_matched_and_marked_in_full():
    """The index found «вишлисте» for a query of «вишлист»; a highlighter that
    insisted on the exact string would mark nothing on a perfectly good hit."""
    assert _marked(search.snippet(TEXT, "вишлист")) == ["вишлисте"]


def test_a_query_word_longer_than_the_text_form_still_matches():
    assert _marked(search.snippet("Подписать договор аренды.", "договоры")) == ["договор"]


def test_common_short_words_are_not_marked():
    """Otherwise every result comes back painted on «что», «как» and «для»."""
    assert _marked(search.snippet(TEXT, "что там про айфон")) == ["айфон"]


def test_a_vector_hit_with_no_matching_words_still_reads():
    """Embeddings match meaning, so a result routinely shares no word with the
    query. It gets the head of the chunk rather than nothing."""
    runs = search.snippet(TEXT, "мобильная техника")

    assert _marked(runs) == []
    assert _visible(runs).startswith("Предложение номер 1,")


def test_the_cut_lands_on_word_boundaries():
    """A snippet that opens «…исление НДС» reads as a bug even when it is not."""
    visible = _visible(search.snippet(TEXT, "айфон")).strip("…").strip()

    assert not visible.startswith(" ")
    for word in (visible.split()[0], visible.split()[-1]):
        assert word in TEXT


def test_a_word_repeated_throughout_does_not_produce_endless_runs():
    runs = search.snippet("айфон " * 500, "айфон")

    assert len(runs) < 60


@pytest.mark.parametrize("query", ["", "   ", "?!", "а"])
def test_a_query_with_nothing_to_match_on_is_not_an_error(query):
    assert _visible(search.snippet(TEXT, query))


# --- the endpoint ------------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(app, base_url="https://testserver") as opened:
        token = issue_token(get_settings(), uuid4(), timedelta(minutes=5))
        assert opened.post("/api/auth/login", json={"token": token}).status_code == 200
        yield opened


def _found(monkeypatch, chunks=None, *, fails=False):
    async def retrieve(_settings, _user_id, query, k, document_id=None):
        if fails:
            raise httpx.ConnectError("the indexer is restarting")
        return chunks or []

    monkeypatch.setattr(retriever, "retrieve", retrieve)


def _chunk(text=TEXT, **overrides):
    return retriever.Chunk(
        **{
            "text": text,
            "score": 0.9,
            "document_id": str(uuid4()),
            "filename": "Wishlist.md",
            "page": None,
            **overrides,
        }
    )


def test_a_search_returns_marked_snippets(client, monkeypatch):
    _found(monkeypatch, [_chunk()])

    body = client.get("/api/search", params={"q": "вишлист"}).json()

    (hit,) = body["hits"]
    assert hit["filename"] == "Wishlist.md"
    assert [piece["text"] for piece in hit["snippet"] if piece["hit"]] == ["вишлисте"]
    assert body["took_ms"] >= 0


def test_an_empty_index_is_an_empty_list_not_an_error(client, monkeypatch):
    _found(monkeypatch, [])

    response = client.get("/api/search", params={"q": "что угодно"})

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_an_indexer_that_is_restarting_says_so(client, monkeypatch):
    """It restarts every time a source is connected. A 500 here would read as
    "search is broken" instead of "wait ten seconds"."""
    _found(monkeypatch, fails=True)

    response = client.get("/api/search", params={"q": "вишлист"})

    assert response.status_code == 503
    assert "индекс" in response.json()["detail"]


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "x" * 501}, {"q": "x", "limit": 0}])
def test_a_query_that_is_not_one_is_refused(client, monkeypatch, params):
    _found(monkeypatch, [])

    assert client.get("/api/search", params=params).status_code == 422


def test_the_limit_reaches_the_index(client, monkeypatch):
    asked = {}

    async def retrieve(_settings, _user_id, query, k, document_id=None):
        asked.update(k=k, query=query)
        return []

    monkeypatch.setattr(retriever, "retrieve", retrieve)
    client.get("/api/search", params={"q": "  вишлист  ", "limit": 7})

    assert asked == {"k": 7, "query": "вишлист"}, "trimmed, and not silently capped at retrieve_k"


def test_search_needs_a_session():
    with TestClient(app, base_url="https://testserver") as anonymous:
        assert anonymous.get("/api/search", params={"q": "вишлист"}).status_code == 401
