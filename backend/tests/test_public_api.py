"""The public answer endpoint: both ways of asking, and the bill.

Two promises are tested here because both are the kind that break quietly. The
first is that ``done`` carries the same object the non-streaming call returns —
that is the whole reason a client may ignore the tokens, and if the two drift
nobody finds out until somebody's integration is subtly missing citations. The
second is that a failed turn says so and does not send ``done``: a stream that
simply stops looks exactly like an answer that ended.
"""

import json
from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import conversation
from app.config import get_settings
from app.main import app
from app.security import issue_token

CITATION = {
    "n": 1,
    "document_id": "0f0f9d3c-9c4e-4f4b-9d61-0b1d1c1c1c1c",
    "filename": "Политика.pdf",
    "page": 3,
    "url": None,
    "snippet": "28 календарных дней",
    # When the far end last changed the document, so the chip can say how old
    # the answer's source is.
    "modified_at": 1788000000,
}


@pytest.fixture
def client():
    user_id = uuid4()
    token = issue_token(get_settings(), user_id, timedelta(minutes=5))
    with TestClient(app, base_url="https://testserver") as test_client:
        test_client.headers["Authorization"] = f"Bearer {token}"
        yield test_client


def agent_that(*, fail: bool = False, timeout: bool = False):
    """A turn that spends a little and cites one fragment."""

    async def answer(settings, user_id, question, history, web=False, spend=None):
        # Both halves of a step, as the real stream sends them: the tool's name
        # the moment it is chosen, the query once its arguments have parsed.
        yield "step", {"tool": "search_knowledge", "query": ""}
        yield "step", {"tool": "search_knowledge", "query": question}
        if spend is not None:
            spend.add("rerank", "judge", prompt_tokens=900, completion_tokens=12, cost_usd=1.7e-4)
        yield "token", "Отпуск 28 дней "
        if timeout:
            raise TimeoutError
        if fail:
            raise RuntimeError("postgresql://rag:hunter2@postgres1:5432/rag is unreachable")
        yield "token", "[1]."
        if spend is not None:
            spend.add("answer", "gpt", prompt_tokens=4100, completion_tokens=48, cost_usd=4.2e-4)
        yield "citations", [CITATION]

    return answer


@pytest.fixture
def working(monkeypatch):
    monkeypatch.setattr(conversation.agent, "answer", agent_that())


def ask(client, **body):
    return client.post("/api/v1/answer", json={"question": "сколько дней отпуска", **body})


def test_one_call_returns_the_answer_its_sources_and_what_it_cost(client, working):
    body = ask(client).json()

    assert body["answer"] == "Отпуск 28 дней [1]."
    assert body["citations"] == [CITATION]
    # One step, not two: the query replaces the announcement of the tool that
    # was about to run it. A list of what happened wants the finished version.
    assert body["steps"] == [{"tool": "search_knowledge", "query": "сколько дней отпуска"}]
    assert body["usage"]["cost_usd"] == 0.00059
    assert body["usage"]["cost_complete"] is True
    assert [row["stage"] for row in body["usage"]["stages"]] == ["rerank", "answer"]
    # Nothing was stored, because nothing was asked to be.
    assert body["chat_id"] is None and body["message_id"] is None


def test_the_last_event_is_the_answer_a_client_could_have_skipped_the_stream_for(client, working):
    plain = ask(client).json()

    streamed = ask(client, stream=True)
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = _events(streamed.text)

    assert [name for name, _ in events] == [
        "step",
        "step",
        "token",
        "token",
        "citations",
        "usage",
        "done",
    ]
    done = dict(events[-1][1])
    # took_ms is the one field that cannot match: it is measured twice.
    assert done.pop("took_ms") >= 0
    assert done == {key: value for key, value in plain.items() if key != "took_ms"}


def test_a_failed_turn_is_a_502_that_carries_no_internals(client, monkeypatch):
    monkeypatch.setattr(conversation.agent, "answer", agent_that(fail=True))

    response = ask(client)

    assert response.status_code == 502
    assert "postgres" not in response.text and "hunter2" not in response.text


def test_a_turn_that_outlasts_its_ceiling_is_a_504(client, monkeypatch):
    monkeypatch.setattr(conversation.agent, "answer", agent_that(timeout=True))

    assert ask(client).status_code == 504


def test_a_stream_that_fails_says_so_and_does_not_pretend_to_be_done(client, monkeypatch):
    monkeypatch.setattr(conversation.agent, "answer", agent_that(fail=True))

    events = _events(ask(client, stream=True).text)

    assert [name for name, _ in events] == ["step", "step", "token", "error"]
    assert "hunter2" not in ask(client, stream=True).text


def test_two_histories_are_refused_rather_than_merged(client, working):
    response = ask(client, chat_id=str(uuid4()), history=[{"role": "user", "content": "привет"}])

    assert response.status_code == 422


def test_a_question_of_only_spaces_is_not_a_question(client, working):
    assert client.post("/api/v1/answer", json={"question": "   "}).status_code == 400


def test_caller_supplied_history_reaches_the_model(client, monkeypatch):
    seen: list[list[str]] = []

    async def answer(settings, user_id, question, history, web=False, spend=None):
        seen.append([f"{m.role}:{m.content}" for m in history])
        yield "token", "ок"

    monkeypatch.setattr(conversation.agent, "answer", answer)
    ask(client, history=[{"role": "user", "content": "а"}, {"role": "assistant", "content": "б"}])

    assert seen == [["user:а", "assistant:б"]]


def test_the_history_a_caller_sends_is_still_capped_to_the_window(client, monkeypatch):
    """A caller keeping its own history is not a reason to replay a year of it
    into every prompt — the window exists to bound what a turn costs."""
    seen: list[int] = []

    async def answer(settings, user_id, question, history, web=False, spend=None):
        seen.append(len(history))
        yield "token", "ок"

    monkeypatch.setattr(conversation.agent, "answer", answer)
    ask(client, history=[{"role": "user", "content": f"{n}"} for n in range(60)])

    assert seen == [get_settings().history_window]


def test_only_the_two_compat_paths_answer_in_openai_shape(client):
    """The rewrite is by exact path, not by prefix: /api/v1/chats starts with
    /api/v1/chat, and a chat endpoint answering in OpenAI's error shape would
    break the frontend's error handling instead."""
    with TestClient(app, base_url="https://testserver") as anonymous:
        assert "error" in anonymous.get("/api/v1/models").json()
        assert "detail" in anonymous.get("/api/v1/me").json()

    # A validation error on a native endpoint keeps FastAPI's shape too.
    assert "detail" in client.post("/api/v1/answer", json={}).json()


def test_search_is_delegated_to_the_one_implementation_of_it(client, monkeypatch):
    """The public endpoint calls the frontend's handler directly, which works
    because a FastAPI decorator returns the function unchanged. That is an
    assumption about somebody else's library, so it is checked rather than
    trusted — and the narrowing arguments are checked with it."""
    from app import retriever

    asked: list[dict] = []

    async def retrieve(_settings, _user_id, query, k, document_id=None, source_id=None, since=None):
        asked.append({"query": query, "source_id": source_id, "since": since})
        return [
            retriever.Chunk(
                text="28 календарных дней отпуска",
                score=0.9,
                document_id=str(uuid4()),
                filename="Политика.pdf",
                page=3,
            )
        ]

    monkeypatch.setattr(retriever, "retrieve", retrieve)
    source = uuid4()
    body = client.get(
        "/api/v1/search",
        params={"q": "отпуск", "limit": 5, "source": str(source), "days": 30},
    ).json()

    assert asked[0]["query"] == "отпуск"
    assert asked[0]["source_id"] == source
    assert asked[0]["since"] is not None
    assert body["found"] == 1
    assert "".join(piece["text"] for piece in body["hits"][0]["snippet"]).startswith("28")


def test_the_whole_public_surface_is_registered(client):
    """A list, so that a route quietly disappearing from the app is a failing
    test rather than somebody's 404 in production. It is also the shortest
    honest description of what /api/v1 is."""
    paths = client.get("/api/openapi.json").json()["paths"]

    assert {path for path in paths if path.startswith("/api/v1")} == {
        "/api/v1/me",
        "/api/v1/answer",
        "/api/v1/search",
        "/api/v1/verify",
        "/api/v1/related",
        "/api/v1/changes",
        "/api/v1/documents/url",
        "/api/v1/documents",
        "/api/v1/documents/{document_id}",
        "/api/v1/documents/{document_id}/link",
        "/api/v1/sources",
        "/api/v1/chats",
        "/api/v1/chats/{chat_id}",
        "/api/v1/chats/{chat_id}/messages",
        "/api/v1/chat/completions",
        "/api/v1/models",
    }


def test_every_endpoint_but_three_requires_a_token(client):
    """The whole app, not only /api/v1: an endpoint that forgets its user
    dependency is an open door, and nothing about the response would look
    wrong from the outside. Three are public by design and named here, so
    adding a fourth is a decision somebody has to write down.
    """
    public = {
        ("/api/health", "get"),
        ("/api/auth/login", "post"),  # trades a signed token for a cookie
        ("/api/auth/logout", "post"),  # clears a cookie; nothing to protect
    }
    paths = client.get("/api/openapi.json").json()["paths"]

    unguarded = {
        (path, method)
        for path, methods in paths.items()
        for method, spec in methods.items()
        if not spec.get("security")
    }
    assert unguarded == public


def _events(payload: str) -> list[tuple[str, object]]:
    """The SSE stream, parsed back into (event, data) pairs."""
    events = []
    for frame in payload.strip().split("\n\n"):
        name, _, data = frame.partition("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events
