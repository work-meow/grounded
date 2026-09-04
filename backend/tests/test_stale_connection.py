"""A connection the indexer already closed must cost one retry, not one answer.

Seen in production: the first request after an idle period failed with
"Server disconnected without sending a response", and the user got a chat turn
that simply did not work — or a file list that flipped back to "processing".
"""

from uuid import UUID

import httpx
import pytest

from app import retriever
from app.config import Settings

ALICE = UUID("11111111-1111-1111-1111-111111111111")


def settings() -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k")


@pytest.fixture
def flaky_indexer(request):
    """Fail the first N calls the way a dead keep-alive connection does."""
    failures, payload = request.param
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= failures:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, json=payload)

    retriever.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    yield calls
    retriever.set_client(None)


HIT = [{"text": "срок уведомления — 45 дней", "dist": 0.1, "metadata": {"page_number": 3}}]


@pytest.mark.parametrize("flaky_indexer", [(1, HIT)], indirect=True)
async def test_a_dropped_connection_is_retried_once(flaky_indexer):
    chunks = await retriever.retrieve(settings(), ALICE, "уведомление", k=4)

    assert flaky_indexer["n"] == 2
    assert [chunk.page for chunk in chunks] == [3]


@pytest.mark.parametrize("flaky_indexer", [(1, [])], indirect=True)
async def test_the_document_list_is_retried_too(flaky_indexer):
    """Otherwise one dead connection shows every file as still processing."""
    assert await retriever.indexed_documents(settings(), ALICE) == []
    assert flaky_indexer["n"] == 2


@pytest.mark.parametrize("flaky_indexer", [(2, HIT)], indirect=True)
async def test_an_indexer_that_is_really_down_still_fails(flaky_indexer):
    """One retry, not a loop: a down indexer must surface, not hang the turn."""
    with pytest.raises(httpx.RemoteProtocolError):
        await retriever.retrieve(settings(), ALICE, "уведомление", k=4)

    assert flaky_indexer["n"] == 2
