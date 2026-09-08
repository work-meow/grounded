"""A line in front of each chunk saying what it is part of.

A chunk from the middle of a leave policy can read, in full, "28 календарных
дней" — nothing in it says leave, or policy. This puts that back, and the
failure that matters is not the call going wrong: it is a description landing
on the wrong chunk. A fragment described as something it is not becomes
findable under the wrong words, which is worse than one described as nothing,
so a batch is all or nothing and the test says so.

The other rule under test is the one this whole codebase holds to: a
degradation must never cost the document. Every path here returns the chunk.
"""

import asyncio
import json

import httpx
import pytest
from pathway.xpacks.llm.splitters import BaseSplitter

from rag_indexer.config import IndexerSettings
from rag_indexer.contextual import (
    PROMPT_VERSION,
    ContextualSplitter,
    _contexts,
    _VersionedCache,
)


def settings(**overrides) -> IndexerSettings:
    return IndexerSettings(
        **{
            "s3_bucket": "b",
            "s3_access_key_id": "k",
            "s3_secret_access_key": "s",
            "openrouter_api_key": "key",
            "contextual_chunks": True,
            **overrides,
        }
    )


class Cut(BaseSplitter):
    """A splitter that cuts on a marker, so the test controls the chunks."""

    def chunk(self, text: str, metadata: dict | None = None, **kwargs):
        return [(part, dict(metadata or {})) for part in text.split("|") if part]


def served(payload, *, status: int = 200, capture: list | None = None):
    """A transport standing in for the describing model."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(json.loads(request.content))
        if isinstance(payload, Exception):
            raise payload
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


@pytest.fixture
def model(monkeypatch):
    """Answer the describing call with whatever the test says."""

    def serve(payload, *, status: int = 200):
        seen: list = []
        client = httpx.AsyncClient(transport=served(payload, status=status, capture=seen))
        monkeypatch.setattr("rag_indexer.contextual._client", lambda _timeout: client)
        return seen

    return serve


def _said(*contexts: str) -> dict:
    return {"choices": [{"message": {"content": json.dumps({"contexts": list(contexts)})}}]}


def split(text: str, config: IndexerSettings | None = None):
    """One part through the splitter.

    `asyncio.run` rather than an async test: this indexer is synchronous by
    design — Pathway owns the threads — and it has no async test plugin. One
    loop per call is also closer to how Pathway drives the UDF.
    """
    splitter = ContextualSplitter(Cut(), config or settings())
    return asyncio.run(splitter.__wrapped__(text))


def test_each_chunk_gets_its_own_line(model):
    model(_said("Про отпуск в политике компании.", "Про больничный."))

    chunks = split("28 календарных дней|оплачивается с первого дня")

    assert chunks[0][0] == "Про отпуск в политике компании.\n\n28 календарных дней"
    assert chunks[1][0] == "Про больничный.\n\nоплачивается с первого дня"


def test_the_lines_are_never_slid_onto_other_chunks(model):
    """A short list would otherwise be padded and every description after the
    gap would belong to the chunk before it — findable under the wrong words,
    which is worse than not findable at all."""
    model(_said("только одна фраза"))

    chunks = split("первый|второй|третий")

    assert [body for body, _ in chunks] == ["первый", "второй", "третий"]


def test_a_part_that_is_one_chunk_is_not_described(model):
    """It already opens with the document's name and there is nothing to
    situate it against — and this is most of a corpus of short notes, so it is
    also where the money is saved."""
    seen = model(_said("не должно понадобиться"))

    chunks = split("один кусок и всё")

    assert chunks == [("один кусок и всё", {})]
    assert seen == [], "ни одного вызова"


def test_a_part_of_too_many_chunks_is_left_alone(model):
    seen = model(_said("не понадобится"))
    many = "|".join(f"кусок {n}" for n in range(6))

    chunks = split(many, settings(context_max_chunks=5))

    assert len(chunks) == 6
    assert all(body.startswith("кусок") for body, _ in chunks)
    assert seen == []


def test_chunks_are_described_in_batches(model):
    """One call per part is the point; one call per chunk would pay for the
    document's text over and over."""
    seen = model(_said("а", "б"))

    split("один|два|три|четыре", settings(context_batch=2))

    assert len(seen) == 2, "четыре куска по два в запросе"


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": "не json"}}]},
        {"choices": [{"message": {"content": '{"contexts": "строка"}'}}]},
        {"choices": [{"message": {"content": '{"other": []}'}}]},
        {"choices": []},
        {},
    ],
)
def test_an_answer_in_an_unknown_shape_costs_the_context_only(model, payload):
    model(payload)

    chunks = split("первый|второй")

    assert [body for body, _ in chunks] == ["первый", "второй"]


def test_a_failed_call_costs_the_context_only(model):
    model(httpx.ConnectError("no route"))

    chunks = split("первый|второй")

    assert [body for body, _ in chunks] == ["первый", "второй"]


def test_an_http_error_costs_the_context_only(model):
    model({"error": "rate limited"}, status=429)

    chunks = split("первый|второй")

    assert [body for body, _ in chunks] == ["первый", "второй"]


def test_the_metadata_of_every_chunk_survives(model):
    """Tenant isolation, citations and indexing status all hang off it: a
    chunk that lost its metadata is a document belonging to nobody."""
    model(_said("а", "б"))
    splitter = ContextualSplitter(Cut(), settings())

    chunks = asyncio.run(splitter.__wrapped__("первый|второй"))

    assert all(isinstance(metadata, dict) for _, metadata in chunks)


def test_the_document_reaches_the_model_truncated(model):
    seen = model(_said("а", "б"))
    long = "х" * 9000 + "|" + "второй"

    split(long, settings(context_document_chars=500))

    asked = seen[0]["messages"][1]["content"]
    assert asked.count("х") <= 1000, "и документ, и фрагмент подрезаны"


def test_the_splitter_can_be_built_more_than_once():
    """Naming the cache after the prompt version made this raise: a named
    DiskCache must be unique for the life of the process. Harmless in
    production, where it is built once, and a landmine anywhere else."""
    ContextualSplitter(Cut(), settings())
    ContextualSplitter(Cut(), settings())


def test_the_instructions_version_is_part_of_the_cache_key():
    """Otherwise the key is the part's text alone, and editing the prompt would
    leave every chunk described to the old brief with nothing to show for it."""
    cache = _VersionedCache()

    key = cache.make_key(("какой-то текст",), {})

    assert key.startswith(f"v{PROMPT_VERSION}:")


def test_a_line_that_is_not_a_string_is_no_line():
    assert _contexts({"choices": [{"message": {"content": '{"contexts": [1, "б"]}'}}]}, 2) == [
        "",
        "б",
    ]
