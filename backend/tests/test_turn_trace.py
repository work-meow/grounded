"""What the turn did, kept with its answer.

Everything here was computed and thrown away before: which tools ran with
which queries, what the provider charged, and how many fragments the model was
actually handed. The last one is the reason this exists — precision is a
question about the fragments a turn was *given*, and only the cited ones used
to survive it, so "eight shown, two cited" and "three shown, two cited" were
the same row.

Both paths that save an answer are checked, because they are different code:
the browser streams and saves in a `finally`, the API can answer in one piece.
"""

from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import conversation
from app.config import get_settings
from app.main import app
from app.security import issue_token

SHOWN = [
    {"n": 1, "document_id": "d-1", "filename": "Политика.pdf", "page": 3},
    {"n": 2, "document_id": "d-2", "filename": "Регламент.md", "page": None},
]
CITED = [{"n": 1, "document_id": "d-1", "filename": "Политика.pdf", "page": 3, "snippet": "28"}]


def agent_that_answers(*, fail: bool = False):
    async def answer(settings, user_id, question, history, web=False, spend=None):
        yield "step", {"tool": "search_knowledge", "query": ""}
        yield "step", {"tool": "search_knowledge", "query": "отпуск"}
        if spend is not None:
            spend.add("rerank", "judge", prompt_tokens=900, completion_tokens=12, cost_usd=1.7e-4)
        yield "token", "28 дней [1]."
        if fail:
            raise RuntimeError("provider said no")
        if spend is not None:
            spend.add("answer", "gpt", prompt_tokens=4100, completion_tokens=48, cost_usd=4.2e-4)
        yield "citations", CITED
        yield "shown", SHOWN

    return answer


@pytest.fixture
def saved(monkeypatch) -> list[dict]:
    """Capture the row instead of writing it.

    Opening the turn is stubbed with it: asking into a stored chat reads its
    history from PostgreSQL, and no test here has one. What is under test is
    what gets written, not where the history came from.
    """
    rows: list[dict] = []

    async def fake_save(chat_id, text, citations, trace=None):
        rows.append({"text": text, "citations": citations, "trace": trace})
        return uuid4()

    async def fake_open(session, user_id, chat_id, question, window):
        return []

    monkeypatch.setattr(conversation, "save_answer", fake_save)
    monkeypatch.setattr(conversation, "open_turn", fake_open)
    return rows


@pytest.fixture
def client():
    token = issue_token(get_settings(), uuid4(), timedelta(minutes=5))
    with TestClient(app, base_url="https://testserver") as opened:
        opened.headers["Authorization"] = f"Bearer {token}"
        yield opened


def test_the_shape_is_one_shape(monkeypatch):
    """Three call sites save an answer; a trace that means different things in
    different rows would be worse than no trace at all."""
    assert conversation.trace([], [], None) == {"steps": [], "shown": [], "usage": None}


def test_the_api_stores_what_it_searched_what_it_saw_and_what_it_cost(client, monkeypatch, saved):
    monkeypatch.setattr(conversation.agent, "answer", agent_that_answers())

    client.post(
        "/api/v1/answer", json={"question": "сколько дней отпуска", "chat_id": str(uuid4())}
    )

    trace = saved[0]["trace"]
    # One step, not two: the query replaces the announcement of the tool that
    # was about to run it.
    assert trace["steps"] == [{"tool": "search_knowledge", "query": "отпуск"}]
    assert trace["usage"]["cost_usd"] == 0.00059
    # Two fragments were handed over and one was cited — which is exactly the
    # pair of numbers that used to be unrecoverable.
    assert len(trace["shown"]) == 2
    assert len(saved[0]["citations"]) == 1


def test_the_fragments_are_kept_without_their_text(client, monkeypatch, saved):
    """The cited ones already carry snippets in their own column; twenty
    fragments of prose per message would be a table nobody reads back."""
    monkeypatch.setattr(conversation.agent, "answer", agent_that_answers())

    client.post("/api/v1/answer", json={"question": "вопрос", "chat_id": str(uuid4())})

    for fragment in saved[0]["trace"]["shown"]:
        assert set(fragment) == {"n", "document_id", "filename", "page"}


def test_a_streamed_turn_stores_the_same_thing(client, monkeypatch, saved):
    monkeypatch.setattr(conversation.agent, "answer", agent_that_answers())

    client.post(
        "/api/v1/answer",
        json={"question": "вопрос", "chat_id": str(uuid4()), "stream": True},
    )

    trace = saved[0]["trace"]
    assert trace["steps"] and trace["shown"]
    assert trace["usage"]["cost_usd"] == 0.00059


def test_a_turn_that_broke_is_still_traced(client, monkeypatch, saved):
    """The one worth looking at later. The answer stops mid-sentence, so
    `shown` is empty — and that is itself the useful fact."""
    monkeypatch.setattr(conversation.agent, "answer", agent_that_answers(fail=True))

    client.post(
        "/api/v1/answer",
        json={"question": "вопрос", "chat_id": str(uuid4()), "stream": True},
    )

    trace = saved[0]["trace"]
    assert saved[0]["text"] == "28 дней [1]."
    assert trace["shown"] == []
    # The rerank that did happen before the failure is still on the bill.
    assert trace["usage"]["cost_usd"] == 0.00017


def test_a_question_asked_without_a_chat_stores_nothing(client, monkeypatch, saved):
    monkeypatch.setattr(conversation.agent, "answer", agent_that_answers())

    body = client.post("/api/v1/answer", json={"question": "вопрос"}).json()

    assert saved == []
    assert body["message_id"] is None
