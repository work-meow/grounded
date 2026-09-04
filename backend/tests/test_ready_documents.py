"""The document list comes from Pathway, so pin the shape we expect back.

Uses a mock transport rather than a stubbed function: the request body, the
prefix matching and the status handling are all part of what can break.
"""

import json
from uuid import UUID, uuid4

import httpx
import pytest
from rag_shared.doc_key import build_key

from app import retriever
from app.config import Settings

ALICE = UUID("11111111-1111-1111-1111-111111111111")
BOB = UUID("22222222-2222-2222-2222-222222222222")


def settings() -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k")


def entry(user_id: UUID, document_id: UUID, status: str, **extra) -> dict:
    return {
        "path": build_key(user_id, uuid4(), document_id, "doc.pdf"),
        "_indexing_status": status,
        "modified_at": 1,
    } | extra


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
CONNECTED = uuid4()
ALICE_PENDING = uuid4()
BOB_READY = uuid4()

PAYLOAD = [
    entry(ALICE, ALICE_READY, "INDEXED"),
    entry(ALICE, ALICE_PENDING, "INGESTED"),
    entry(BOB, BOB_READY, "INDEXED"),
    {"path": "junk/loose.pdf", "_indexing_status": "INDEXED"},
]


async def documents_of(user_id: UUID) -> dict[str, retriever.IndexedDocument]:
    found = await retriever.indexed_documents(settings(), user_id)
    return {document.document_id: document for document in found}


@pytest.mark.parametrize("pathway", [PAYLOAD], indirect=True)
async def test_only_this_user_s_documents_come_back(pathway):
    documents = await documents_of(ALICE)

    # Not another user's file, and not a stray object outside the key layout.
    assert set(documents) == {str(ALICE_READY), str(ALICE_PENDING)}
    assert documents[str(ALICE_READY)].ready
    assert not documents[str(ALICE_PENDING)].ready


@pytest.mark.parametrize("pathway", [PAYLOAD], indirect=True)
async def test_status_is_requested_without_a_server_side_filter(pathway):
    """Pathway zips statuses against the unfiltered list, so we must not filter there."""
    await documents_of(ALICE)

    assert pathway["url"].endswith("/v1/inputs")
    assert pathway["body"] == {"return_status": True}


@pytest.mark.parametrize(
    "pathway",
    [[entry(ALICE, CONNECTED, "INDEXED", web_url="https://notion.so/p", size=4096)]],
    indirect=True,
)
async def test_a_connected_document_carries_where_to_open_it(pathway):
    """Uploads get a presigned link; everything else has to say so itself."""
    document = (await documents_of(ALICE))[str(CONNECTED)]

    assert document.web_url == "https://notion.so/p"
    assert document.size_bytes == 4096


@pytest.mark.parametrize(
    "pathway", [[entry(ALICE, CONNECTED, "INDEXED", size="not a number")]], indirect=True
)
async def test_metadata_that_makes_no_sense_is_dropped_not_fatal(pathway):
    """Metadata crosses JSON from a connector we do not control."""
    document = (await documents_of(ALICE))[str(CONNECTED)]

    assert document.size_bytes is None
    assert document.web_url is None


# 36 characters of hex and dashes, and not a UUID: the key layout matches it,
# uuid.UUID does not.
IMPOSSIBLE = {
    "path": f"users/{ALICE}/sources/{uuid4()}/{'-' * 36}/x.pdf",
    "_indexing_status": "INDEXED",
}


@pytest.mark.parametrize("pathway", [[IMPOSSIBLE]], indirect=True)
async def test_one_impossible_object_does_not_take_out_the_file_list(pathway):
    assert await documents_of(ALICE) == {}


@pytest.mark.parametrize("pathway", [[]], indirect=True)
async def test_an_empty_index_lists_nothing(pathway):
    assert await documents_of(ALICE) == {}
