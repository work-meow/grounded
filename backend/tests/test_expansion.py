"""Asking the index the same question in more than one way.

The value is not that a paraphrase beats the question — it is that BM25 and
the embedder fail on different wordings, so the union is what the judge should
be choosing from. Which makes the failure modes the interesting part: a
rewrite that came back as prose must not reach the index, one phrasing failing
must not lose the others, and the same fragment found twice must not become
two fragments in the answer.

Fusion is by rank and not by score on purpose. What comes back from the index
is a position in a list, not a measure of anything, so two searches' scores
cannot be compared — but their ranks can.
"""

import json

import httpx
import pytest

from app import expansion, http, retriever
from app.config import Settings


def settings(**overrides) -> Settings:
    return Settings(
        **{
            "jwt_secret": "x" * 40,
            "openrouter_api_key": "k",
            "expand_queries": True,
            **overrides,
        }
    )


def chunk(document: str, page: int | None = None, text: str = "фрагмент") -> retriever.Chunk:
    return retriever.Chunk(
        text=text, score=0.01, document_id=document, filename=f"{document}.md", page=page
    )


@pytest.fixture
def rewriter():
    """The rephrasing model, answering whatever the test puts in it."""

    def serve(payload, *, status: int = 200):
        def handler(_request: httpx.Request) -> httpx.Response:
            if isinstance(payload, Exception):
                raise payload
            return httpx.Response(status, json=payload)

        http.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    yield serve
    http.set_client(None)


def _said(*queries: str) -> dict:
    return {"choices": [{"message": {"content": json.dumps({"queries": list(queries)})}}]}


# --- what reaches the index --------------------------------------------------


async def test_the_question_is_asked_as_well_as_rephrased(rewriter):
    rewriter(_said("продолжительность ежегодного отпуска", "сколько длится отпуск"))

    asked = await expansion.variants(settings(), "сколько дней отпуска?")

    assert asked[0] == "сколько дней отпуска?", "оригинал всегда первый"
    assert len(asked) == 3


async def test_a_rewrite_that_is_the_question_again_is_dropped(rewriter):
    rewriter(_said("Сколько  дней   отпуска?", "продолжительность отпуска"))

    asked = await expansion.variants(settings(), "сколько дней отпуска?")

    assert asked == ["сколько дней отпуска?", "продолжительность отпуска"]


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": "не json"}}]},
        {"choices": [{"message": {"content": '{"queries": "строка"}'}}]},
        {"choices": [{"message": {"content": '{"other": ["а"]}'}}]},
        {"choices": []},
        {},
    ],
)
async def test_an_unusable_rewrite_leaves_the_question_alone(rewriter, payload):
    rewriter(payload)

    assert await expansion.variants(settings(), "вопрос") == ["вопрос"]


async def test_a_failed_rewrite_leaves_the_question_alone(rewriter):
    rewriter(httpx.ConnectError("no route"))

    assert await expansion.variants(settings(), "вопрос") == ["вопрос"]


async def test_nothing_is_asked_twice_when_expansion_is_off(rewriter):
    rewriter(_said("не должно понадобиться"))

    assert await expansion.variants(settings(expand_queries=False), "вопрос") == ["вопрос"]


async def test_a_rewrite_is_bounded_like_a_query_is(rewriter):
    """A model that decided to write an essay must not send one to the index."""
    rewriter(_said("х" * 900))

    asked = await expansion.variants(settings(), "вопрос")

    assert len(asked[1]) == 500


async def test_only_as_many_rewrites_as_asked_for(rewriter):
    rewriter(_said("раз", "два", "три", "четыре"))

    assert len(await expansion.variants(settings(expand_to=2), "вопрос")) == 3


# --- merging what came back --------------------------------------------------


def test_a_fragment_found_by_two_phrasings_outranks_one_found_by_either():
    """The whole reason to ask twice."""
    both = chunk("оба")
    fused = expansion.fuse(
        [[chunk("первый"), both], [both, chunk("второй")]],
        k=10,
    )

    assert fused[0].document_id == "оба"


def test_the_same_fragment_retrieved_twice_is_one_fragment():
    """Otherwise the answer cites the same paragraph twice and the judge spends
    its budget reading a duplicate."""
    fused = expansion.fuse([[chunk("а", page=3)], [chunk("а", page=3)]], k=10)

    assert len(fused) == 1


def test_two_pages_of_one_document_stay_two_fragments():
    fused = expansion.fuse([[chunk("а", page=1), chunk("а", page=2)]], k=10)

    assert len(fused) == 2


def test_the_merged_list_is_cut_to_k():
    fused = expansion.fuse([[chunk(f"д{n}") for n in range(20)]], k=5)

    assert len(fused) == 5


def test_one_ranking_alone_comes_back_in_its_own_order():
    ranking = [chunk("а"), chunk("б"), chunk("в")]

    assert [c.document_id for c in expansion.fuse([ranking], k=10)] == ["а", "б", "в"]


def test_the_constant_matches_the_one_the_index_fuses_with():
    """Pathway's HybridIndexFactory uses 60 on the other side of this call;
    two different constants fusing the same ranks would be an arbitrary
    disagreement."""
    assert expansion.RRF_K == 60


# --- searching ----------------------------------------------------------------


async def test_one_phrasing_failing_does_not_lose_the_others(rewriter, monkeypatch):
    rewriter(_said("другая формулировка"))
    calls: list[str] = []

    async def retrieve(_settings, _user, query, _k, **_narrowing):
        calls.append(query)
        if query == "вопрос":
            raise httpx.ConnectError("first one failed")
        return [chunk("нашлось")]

    monkeypatch.setattr(retriever, "retrieve", retrieve)

    found = await expansion.search(settings(), "u", "вопрос", 8)

    assert len(calls) == 2
    assert [c.document_id for c in found] == ["нашлось"]


async def test_every_phrasing_failing_reaches_the_caller(rewriter, monkeypatch):
    """The index being unreachable is what the caller's own error handling is
    for — a search page that says "перестраивается" instead of showing an
    empty result."""
    rewriter(_said("другая формулировка"))

    async def retrieve(*_args, **_kwargs):
        raise httpx.ConnectError("index is down")

    monkeypatch.setattr(retriever, "retrieve", retrieve)

    with pytest.raises(httpx.ConnectError):
        await expansion.search(settings(), "u", "вопрос", 8)


async def test_with_expansion_off_it_is_exactly_one_search(rewriter, monkeypatch):
    """Nothing here may make a search worse than not having it, which starts
    with not changing it when it is switched off."""
    rewriter(_said("не понадобится"))
    calls: list[str] = []

    async def retrieve(_settings, _user, query, _k, **_narrowing):
        calls.append(query)
        return [chunk("а")]

    monkeypatch.setattr(retriever, "retrieve", retrieve)

    await expansion.search(settings(expand_queries=False), "u", "вопрос", 8)

    assert calls == ["вопрос"]


async def test_the_narrowing_is_passed_to_every_phrasing(rewriter, monkeypatch):
    """A source filter that applied to the question and not to its rewrites
    would quietly return fragments from outside the chosen source."""
    rewriter(_said("другая формулировка"))
    seen: list[dict] = []

    async def retrieve(_settings, _user, _query, _k, **narrowing):
        seen.append(narrowing)
        return []

    monkeypatch.setattr(retriever, "retrieve", retrieve)

    await expansion.search(settings(), "u", "вопрос", 8, source_id="src", since=1700000000)

    assert len(seen) == 2
    assert all(one == {"source_id": "src", "since": 1700000000} for one in seen)
