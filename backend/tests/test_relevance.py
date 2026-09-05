"""What of the retrieved set reaches the model.

The index is a ranked retriever: it answers every query with its k best
fragments whether or not any of them is about the question. Measured across
thirteen questions on the deployment, that came to nineteen useful fragments out
of eighty, and a question the documents cannot answer still received eight.

So a second pass judges the candidates. Most of what is worth testing is what
happens when that judgement is unusable — because the fallback has to be the
behaviour it replaced, never a failed answer.
"""

import httpx
import pytest

from app import http, relevance, retriever
from app.config import Settings


def settings(**overrides) -> Settings:
    return Settings(
        **{"jwt_secret": "x" * 40, "openrouter_api_key": "k", "rerank_enabled": True, **overrides}
    )


def chunk(n: int, document: str = "Wishlist.md") -> retriever.Chunk:
    return retriever.Chunk(
        text=f"фрагмент {n}", score=0.01, document_id=document, filename=document, page=None
    )


CANDIDATES = [chunk(n) for n in range(8)]


@pytest.fixture
def judge():
    """The reranking model, answering whatever the test puts in it."""

    def serve(payload, *, status: int = 200):
        def handler(_request: httpx.Request) -> httpx.Response:
            if isinstance(payload, Exception):
                raise payload
            return httpx.Response(status, json=payload)

        http.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    yield serve
    http.set_client(None)


def _verdict(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# --- what it keeps -----------------------------------------------------------


async def test_only_what_bears_on_the_question_comes_back(judge):
    judge(_verdict('{"relevant": [3, 1]}'))

    kept = await relevance.keep_relevant(settings(), "вишлист", CANDIDATES)

    assert [c.text for c in kept] == ["фрагмент 3", "фрагмент 1"], "и в порядке судьи"


async def test_nothing_relevant_means_nothing(judge):
    """The point of the whole pass. "В базе этого нет" stops being the model's
    opinion about eight paragraphs and becomes a fact about the retrieval."""
    judge(_verdict('{"relevant": []}'))

    assert await relevance.keep_relevant(settings(), "налоговый вычет", CANDIDATES) == []


async def test_more_than_the_ceiling_is_trimmed(judge):
    judge(_verdict('{"relevant": [0, 1, 2, 3, 4, 5, 6, 7]}'))

    kept = await relevance.keep_relevant(settings(rerank_keep=3), "вишлист", CANDIDATES)

    assert len(kept) == 3


# --- what it does with an unusable verdict -----------------------------------


@pytest.mark.parametrize(
    "content",
    ['{"relevant": "все"}', "{}", "не json вовсе", '{"relevant": null}'],
)
async def test_a_verdict_that_makes_no_sense_leaves_the_candidates_alone(judge, content):
    """No verdict is not the same answer as an empty one: falling back to the
    ranked order is exactly the behaviour this replaces, so it costs precision
    and never an answer."""
    judge(_verdict(content))

    kept = await relevance.keep_relevant(settings(rerank_keep=6), "вишлист", CANDIDATES)

    assert len(kept) == 6


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("нет связи"), httpx.ReadTimeout("долго")],
)
async def test_a_judge_that_is_not_answering_does_not_take_the_turn_with_it(judge, failure):
    judge(failure)

    assert len(await relevance.keep_relevant(settings(rerank_keep=6), "q", CANDIDATES)) == 6


async def test_an_error_from_the_provider_is_survivable(judge):
    judge({"error": "нет денег"}, status=402)

    assert len(await relevance.keep_relevant(settings(rerank_keep=6), "q", CANDIDATES)) == 6


async def test_an_index_the_judge_invented_is_ignored(judge):
    """It would otherwise carry an unrelated fragment into the answer, or throw."""
    judge(_verdict('{"relevant": [1, 99, -3, "два", 1]}'))

    kept = await relevance.keep_relevant(settings(), "вишлист", CANDIDATES)

    assert [c.text for c in kept] == ["фрагмент 1"], "в диапазоне, без повторов"


async def test_nothing_to_judge_is_not_a_call(judge):
    judge(httpx.ConnectError("судью звать не должны"))

    assert await relevance.keep_relevant(settings(), "вишлист", []) == []


async def test_turning_it_off_asks_nobody(judge):
    judge(httpx.ConnectError("судью звать не должны"))

    kept = await relevance.keep_relevant(
        settings(rerank_enabled=False, rerank_keep=4), "q", CANDIDATES
    )

    assert len(kept) == 4


# --- one document filling the page -------------------------------------------


def test_a_long_document_does_not_take_the_whole_list():
    """For browsing. Five documents beat the same document five times when
    somebody is scanning for *where* something is written."""
    chunks = [chunk(0, "Импульсивные.docx"), chunk(1, "Импульсивные.docx"), chunk(2, "Wishlist.md")]

    kept = relevance.cap_per_document(chunks, 1)

    assert [c.document_id for c in kept] == ["Импульсивные.docx", "Wishlist.md"]


def test_the_cap_keeps_the_order_it_was_given():
    chunks = [chunk(0, "a"), chunk(1, "b"), chunk(2, "a"), chunk(3, "c")]

    assert [c.text for c in relevance.cap_per_document(chunks, 2)] == [
        "фрагмент 0",
        "фрагмент 1",
        "фрагмент 2",
        "фрагмент 3",
    ]
