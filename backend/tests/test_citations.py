"""Agent behaviour worth pinning.

Citation numbers are what the model is told to cite as [1], [2]. If numbering
drifts between what the tool returns and what is stored on the message, every
source link in the UI points at the wrong document.
"""

from app.agent import _Citations, _model, _render
from app.retriever import Chunk


def test_chat_client_is_shared_across_requests():
    """Built per request, each client would leak an HTTP connection pool."""
    first = _model("openai/gpt-4o-mini", "key", 0.0)
    second = _model("openai/gpt-4o-mini", "key", 0.0)
    assert first is second


def chunk(document_id: str, page: int | None, text: str = "текст") -> Chunk:
    return Chunk(
        text=text, score=1.0, document_id=document_id, filename=f"{document_id}.pdf", page=page
    )


def test_numbering_is_stable_across_repeated_tool_calls():
    citations = _Citations()

    first = _render([chunk("a", 1), chunk("b", 2)], citations)
    assert "[1] (a.pdf, стр. 1)" in first
    assert "[2] (b.pdf, стр. 2)" in first

    # A second search that re-finds "a" must reuse [1], not renumber it.
    second = _render([chunk("b", 2), chunk("a", 1), chunk("c", 3)], citations)
    assert "[1] (a.pdf, стр. 1)" in second
    assert "[2] (b.pdf, стр. 2)" in second
    assert "[3] (c.pdf, стр. 3)" in second

    assert [item["n"] for item in citations.items] == [1, 2, 3]
    assert [item["document_id"] for item in citations.items] == ["a", "b", "c"]


def test_same_document_different_pages_are_separate_citations():
    citations = _Citations()
    _render([chunk("a", 1), chunk("a", 7)], citations)

    assert len(citations.items) == 2
    assert {item["page"] for item in citations.items} == {1, 7}


def test_pageless_document_renders_without_a_page_suffix():
    citations = _Citations()
    rendered = _render([chunk("a", None)], citations)

    assert "[1] (a.pdf)" in rendered
    assert citations.items[0]["page"] is None


def test_empty_result_tells_the_model_to_retry_rather_than_inventing():
    citations = _Citations()
    assert "Ничего не найдено" in _render([], citations)
    assert citations.items == []


def test_snippet_is_bounded():
    citations = _Citations()
    _render([chunk("a", 1, "щ" * 5000)], citations)

    assert len(citations.items[0]["snippet"]) == 300
