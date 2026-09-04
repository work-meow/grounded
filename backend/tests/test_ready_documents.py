"""Readiness comes from Pathway, so pin the shape we expect back from it.

Uses a mock transport rather than a stubbed function: the request body, the
prefix matching and the status handling are all part of what can break.
"""

import json
from uuid import UUID, uuid4

import httpx
import pytest

from app import retriever
from app.config import Settings
from shared.doc_key import build_key

ALICE = UUID("11111111-1111-1111-1111-111111111111")
BOB = UUID("22222222-2222-2222-2222-222222222222")


def settings() -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k")


def entry(user_id: UUID, document_id: UUID, status: str) -> dict:
    return {
        "path": build_key(user_id, uuid4(), document_id, "doc.pdf"),
        "_indexing_status": status,
        "modified_at": 1,
    }


@pytest.fixture
def pathway(request):
    """Serve a canned /v1/inputs payload and record the request body."""
    payload, seen = request.param, {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen["url"] = str(http_request.url)
        seen["body"] = json.loads(http_request.content)
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    retriever.set_client(client)
    yield seen
    retriever.set_client(None)


ALICE_READY = uuid4()
ALICE_PENDING = uuid4()
BOB_READY = uuid4()

PAYLOAD = [
    entry(ALICE, ALICE_READY, "INDEXED"),
    entry(ALICE, ALICE_PENDING, "INGESTED"),
    entry(BOB, BOB_READY, "INDEXED"),
    {"path": "junk/loose.pdf", "_indexing_status": "INDEXED"},
]


@pytest.mark.parametrize("pathway", [PAYLOAD], indirect=True)
async def test_only_own_indexed_documents_are_ready(pathway):
    ready = await retriever.ready_document_ids(settings(), ALICE)

    assert ready == {str(ALICE_READY)}
    # Not another user's file, not one still ingesting, not a stray object.
    assert str(BOB_READY) not in ready
    assert str(ALICE_PENDING) not in ready


@pytest.mark.parametrize("pathway", [PAYLOAD], indirect=True)
async def test_status_is_requested_without_a_server_side_filter(pathway):
    """Pathway zips statuses against the unfiltered list, so we must not filter there."""
    await retriever.ready_document_ids(settings(), ALICE)

    assert pathway["url"].endswith("/v1/inputs")
    assert pathway["body"] == {"return_status": True}


@pytest.mark.parametrize("pathway", [[]], indirect=True)
async def test_nothing_is_ready_when_the_indexer_knows_no_files(pathway):
    assert await retriever.ready_document_ids(settings(), ALICE) == set()
