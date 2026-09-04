"""The check that matters: one user's query must never reach another's chunks.

The filter we send to Pathway is JMESPath, and Pathway evaluates it against each
chunk's metadata. So we evaluate the real filter against real metadata shapes
here — if the expression or the metadata contract ever drifts, this fails.
"""

from uuid import UUID, uuid4

import jmespath
import pytest
from rag_shared.doc_key import build_key, parse_key, tenant_metadata

from app.retriever import Chunk, _tenant_filter

ALICE = UUID("11111111-1111-1111-1111-111111111111")
BOB = UUID("22222222-2222-2222-2222-222222222222")


def chunk_metadata(user_id: UUID, filename: str = "contract.pdf") -> dict:
    """Metadata exactly as the indexer would produce it for an uploaded file."""
    key = build_key(user_id, uuid4(), uuid4(), filename)
    return tenant_metadata("text", {"path": key, "page_number": 14})[1]


def as_pathway_rewrites_it(expression: str) -> str:
    """Reproduce document_store._get_jmespath_filter verbatim.

    Pathway does not pass the filter through untouched, and evaluating the raw
    string here would test a dialect the engine never sees — which is exactly
    how a filter that crashed the indexer once passed these tests.
    """
    return expression.replace("'", r"\'").replace("`", "'").replace('"', "")


def matches(user_id: UUID, metadata: dict, document_id: UUID | None = None) -> bool:
    expression = as_pathway_rewrites_it(_tenant_filter(user_id, document_id))
    return bool(jmespath.compile(expression).search(metadata))


def test_filter_uses_backticks_so_pathway_rewrites_it_into_valid_syntax():
    """Quotes would arrive at the engine as \\' and take the process down."""
    raw = _tenant_filter(ALICE)
    assert "`" in raw and "'" not in raw
    assert "\\" not in as_pathway_rewrites_it(raw)


def test_user_matches_own_chunk():
    assert matches(ALICE, chunk_metadata(ALICE))


def test_user_never_matches_another_users_chunk():
    assert not matches(BOB, chunk_metadata(ALICE))
    assert not matches(ALICE, chunk_metadata(BOB))


def test_unparseable_key_is_unreachable():
    """An object that does not follow our layout must match nobody."""
    metadata = tenant_metadata("text", {"path": "junk/stray.pdf"})[1]
    for user in (ALICE, BOB):
        assert not matches(user, metadata)


def test_document_scope_narrows_within_the_same_user():
    key = build_key(ALICE, uuid4(), uuid4(), "a.pdf")
    metadata = tenant_metadata("t", {"path": key})[1]
    wanted = UUID(metadata["document_id"])

    assert matches(ALICE, metadata, document_id=wanted)
    assert not matches(ALICE, metadata, document_id=uuid4())
    # Scoping to a document still cannot cross the tenant boundary.
    assert not matches(BOB, metadata, document_id=wanted)


@pytest.mark.parametrize(
    "filename", ["contract.pdf", "договор аренды.pdf", "a b'c\".pdf", "отчёт (2026).docx"]
)
def test_key_round_trips_for_awkward_filenames(filename):
    key = build_key(ALICE, BOB, ALICE, filename)
    parsed = parse_key(key)
    assert parsed is not None
    assert parsed["filename"] == filename
    assert parsed["user_id"] == str(ALICE)
    # A quote in the filename must not leak into the filter's string literal.
    assert matches(ALICE, tenant_metadata("t", {"path": key})[1])


def test_chunk_reads_page_and_flips_distance_into_a_score():
    hit = {
        "text": "срок уведомления — 30 дней",
        "dist": 0.25,
        "metadata": {"document_id": str(ALICE), "filename": "c.pdf", "page_number": 14},
    }
    chunk = Chunk.from_hit(hit)
    assert chunk.page == 14
    assert chunk.filename == "c.pdf"
    # Pathway sorts ascending by distance, so a smaller dist must score higher.
    assert chunk.score > Chunk.from_hit({**hit, "dist": 0.9}).score


def test_chunk_survives_metadata_without_a_page():
    chunk = Chunk.from_hit({"text": "t", "dist": 0.1, "metadata": {}})
    assert chunk.page is None
    assert chunk.document_id is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(14, 14), ("14", 14), (14.0, 14), (None, None), ("стр. 14", None), ({}, None)],
)
def test_page_number_survives_the_json_round_trip(raw, expected):
    """Metadata crosses JSON, so the page can come back as any of these."""
    chunk = Chunk.from_hit({"text": "t", "dist": 0.1, "metadata": {"page_number": raw}})
    assert chunk.page == expected
