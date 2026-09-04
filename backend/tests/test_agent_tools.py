"""What the agent can see.

The tools are bound to one user by closure, and two of them enumerate that
user's documents. Where that list comes from is the whole point: the database
knows about uploads, the index knows about those *and* everything reached
through a connected source. A Notion page has no row anywhere, so a tool asking
the database would tell the model that a document it can perfectly well search
does not exist.
"""

import asyncio
import uuid

import httpx
import pytest

from app import agent, retriever
from app.config import Settings

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
UPLOADED = uuid.uuid4()
FROM_NOTION = uuid.uuid4()


def settings() -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k")


def _document(document_id, filename, *, ready=True):
    return retriever.IndexedDocument(
        document_id=str(document_id),
        source_id=str(uuid.uuid4()),
        filename=filename,
        web_url=None,
        size_bytes=None,
        modified_at=1,
        ready=ready,
    )


def tools(monkeypatch, found=None, *, fails=False):
    async def indexed_documents(_settings, _user_id):
        if fails:
            raise httpx.ConnectError("indexer is down")
        return found or []

    monkeypatch.setattr(retriever, "indexed_documents", indexed_documents)
    built = agent._build_tools(settings(), USER, agent._Citations())
    return {tool.name: tool for tool in built}


async def test_a_document_from_a_connected_source_is_listed(monkeypatch):
    found = [_document(UPLOADED, "Договор.pdf"), _document(FROM_NOTION, "Заметка.md")]

    listed = await tools(monkeypatch, found)["list_sources"].ainvoke({})

    assert str(UPLOADED) in listed
    assert str(FROM_NOTION) in listed
    assert "Заметка.md" in listed


async def test_a_document_still_indexing_says_so(monkeypatch):
    """Otherwise the model reads an empty search as the document being wrong."""
    found = [_document(UPLOADED, "Договор.pdf", ready=False)]

    listed = await tools(monkeypatch, found)["list_sources"].ainvoke({})

    assert "индексируется" in listed


async def test_an_empty_knowledge_base_says_so_plainly(monkeypatch):
    assert "пуста" in await tools(monkeypatch, [])["list_sources"].ainvoke({})


async def test_an_unreachable_indexer_does_not_end_the_turn(monkeypatch):
    """A tool that raises takes the whole answer with it; a tool that explains
    itself leaves the model able to fall back on search_knowledge."""
    listed = await tools(monkeypatch, fails=True)["list_sources"].ainvoke({})

    assert "search_knowledge" in listed


@pytest.mark.parametrize("given", ["not-a-uuid", "'; drop table", "../../etc", ""])
async def test_read_document_refuses_an_id_that_is_not_one(monkeypatch, given):
    """The UUID parse is the guard: it is what keeps anything but hex and
    dashes out of the filter expression handed to the index."""
    read = tools(monkeypatch, [])["read_document"]

    answer = await read.ainvoke({"document_id": given, "query": "срок"})

    assert answer.startswith("Некорректный document_id")


async def test_read_document_narrows_to_the_owner_and_the_document(monkeypatch):
    seen = {}

    async def retrieve(_settings, user_id, query, k, document_id=None):
        seen.update(user_id=user_id, document_id=document_id, query=query)
        return []

    monkeypatch.setattr(retriever, "retrieve", retrieve)
    read = tools(monkeypatch, [])["read_document"]

    await read.ainvoke({"document_id": str(FROM_NOTION), "query": "срок"})

    # Both, and both as UUID rather than str — the typing is the injection guard.
    assert seen["user_id"] == USER
    assert seen["document_id"] == FROM_NOTION


def test_one_model_client_for_the_process():
    """Built per request it would open a connection pool per question and never
    close one. The cache is what keeps that from happening."""
    first = agent._model("m", "k", 0.0, "minimal")
    second = agent._model("m", "k", 0.0, "minimal")

    assert first is second
    assert agent._model("other", "k", 0.0, "minimal") is not first


def test_the_reasoning_effort_reaches_the_client():
    assert agent._model("m", "k", 0.0, "minimal").reasoning == {"effort": "minimal"}
    assert agent._model("m", "k", 0.0, "").reasoning is None


def test_the_client_is_built_without_a_timeout_on_purpose():
    """Setting one hangs it. ChatOpenRouter accepts `timeout` and
    `request_timeout` without complaint and then never completes a request —
    a call that answers in a second sat past five minutes on the deployment
    with either set. The turn is bounded in answer() instead."""
    client = agent._model("m", "k", 0.0, "minimal")

    assert client.request_timeout is None


async def test_a_turn_that_never_finishes_is_cut_off(monkeypatch):
    """A provider that accepts the connection and then says nothing would hold
    the SSE stream open behind it for as long as it liked. The call-count
    middleware bounds how many requests a turn makes, never how long one takes.
    """

    class _NeverAnswers:
        async def astream(self, _inputs, **_kwargs):
            await asyncio.sleep(30)
            yield None, {}  # pragma: no cover - never reached

    monkeypatch.setattr(agent, "create_agent", lambda **_kwargs: _NeverAnswers())
    settings = Settings(jwt_secret="x" * 40, openrouter_api_key="k", agent_timeout_s=0.05)

    with pytest.raises(TimeoutError):
        async for _ in agent.answer(settings, USER, "вопрос", []):
            pass


async def test_a_turn_that_answers_is_left_alone(monkeypatch):
    class _Answers:
        async def astream(self, _inputs, **_kwargs):
            yield type("Chunk", (), {"content": "сорок"})(), {"langgraph_node": "model"}

    monkeypatch.setattr(agent, "create_agent", lambda **_kwargs: _Answers())
    settings = Settings(jwt_secret="x" * 40, openrouter_api_key="k", agent_timeout_s=5.0)

    events = [event async for event in agent.answer(settings, USER, "вопрос", [])]
    assert events[0] == ("token", "сорок")
    assert events[-1][0] == "citations"
