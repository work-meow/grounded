"""Checking a written text against the documents.

The chat's promise turned around: instead of answering only from the
documents, this says which parts of what somebody already wrote the documents
support, contradict, or have never heard of.

Three verdicts and not two, and that is what most of this tests. "Absent" and
"contradicted" are different situations for the reader — one means go and write
the document, the other means somebody is about to say something that is on
record as false — and "unknown" is neither: a judge that failed has told us
nothing about the documents, and reporting that as "absent" would send
somebody off to write a document that already exists.
"""

import json
from uuid import uuid4

import httpx
import pytest

from app import expansion, http, relevance, retriever, verify
from app.config import Settings
from app.spend import Spend

USER = uuid4()


def settings(**overrides) -> Settings:
    return Settings(**{"jwt_secret": "x" * 40, "openrouter_api_key": "k", **overrides})


def chunk(text: str = "Отпуск — 28 календарных дней.") -> retriever.Chunk:
    return retriever.Chunk(text=text, score=0.9, document_id="d1", filename="Политика.pdf", page=3)


@pytest.fixture
def model():
    """The cheap model, answering whatever the test puts in it, in order."""

    def serve(*payloads):
        answers = list(payloads)

        def handler(_request: httpx.Request) -> httpx.Response:
            payload = answers.pop(0) if len(answers) > 1 else answers[0]
            if isinstance(payload, Exception):
                raise payload
            return httpx.Response(200, json=payload)

        http.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    yield serve
    http.set_client(None)


def _json(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


@pytest.fixture
def found(monkeypatch):
    """What retrieval hands back for a claim."""

    def serve(chunks):
        async def search(*_args, **_kwargs):
            return chunks

        async def keep(_settings, _question, candidates, _spend=None):
            return candidates

        monkeypatch.setattr(expansion, "search", search)
        monkeypatch.setattr(relevance, "keep_relevant", keep)

    return serve


# --- splitting ---------------------------------------------------------------


async def test_a_paragraph_becomes_separate_claims(model):
    model(_json({"claims": ["отпуск 28 дней", "премия раз в год"]}))

    assert await verify.claims(settings(), "текст") == ["отпуск 28 дней", "премия раз в год"]


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": "не json"}}]},
        {"choices": [{"message": {"content": '{"claims": "строка"}'}}]},
        {"choices": [{"message": {"content": '{"claims": []}'}}]},
        {"choices": []},
    ],
)
async def test_a_split_that_failed_checks_the_text_whole(model, payload):
    """A paragraph checked as one lump is a worse answer than the same
    paragraph in pieces, and a far better one than an error."""
    model(payload)

    assert await verify.claims(settings(), "весь текст") == ["весь текст"]


async def test_a_failed_call_checks_the_text_whole(model):
    model(httpx.ConnectError("no route"))

    assert await verify.claims(settings(), "весь текст") == ["весь текст"]


async def test_the_number_of_claims_is_bounded(model):
    """Each one costs a retrieval and a judgement, so a pasted book would
    otherwise be a bill nobody agreed to."""
    model(_json({"claims": [f"утверждение {n}" for n in range(50)]}))

    assert len(await verify.claims(settings(), "текст")) == verify.MAX_CLAIMS


# --- judging -----------------------------------------------------------------


async def test_a_claim_the_documents_confirm(model, found):
    found([chunk()])
    model(_json({"verdict": "supported", "why": "28 календарных дней"}))

    checked = await verify.check(settings(), USER, "отпуск 28 дней")

    assert checked.verdict == "supported"
    assert checked.why == "28 календарных дней"
    assert checked.citations[0]["filename"] == "Политика.pdf"


async def test_a_claim_the_documents_disagree_with(model, found):
    """Not the same as absent, and the more expensive of the two: somebody is
    about to say something that is on record as false."""
    found([chunk()])
    model(_json({"verdict": "contradicted", "why": "во фрагменте 28, а не 35"}))

    checked = await verify.check(settings(), USER, "отпуск 35 дней")

    assert checked.verdict == "contradicted"


async def test_nothing_relevant_is_absent_without_asking(model, found):
    """The judge is not called at all: the relevance pass already answered the
    question, and paying to be told so twice would be a call for nothing."""
    found([])
    model(httpx.ConnectError("не должен вызываться"))

    checked = await verify.check(settings(), USER, "про налоговый вычет")

    assert checked.verdict == "absent"
    assert checked.citations == []


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": '{"verdict": "может быть"}'}}]},
        {"choices": [{"message": {"content": "не json"}}]},
        {"choices": [{"message": {"content": '{"why": "без вердикта"}'}}]},
        {"choices": []},
    ],
)
async def test_a_judge_that_did_not_answer_is_unknown_and_not_absent(model, found, payload):
    """The mistake that would matter: reporting "nobody could tell" as "the
    documents do not mention this" sends somebody off to write a document that
    already exists."""
    found([chunk()])
    model(payload)

    checked = await verify.check(settings(), USER, "отпуск 28 дней")

    assert checked.verdict == "unknown"
    # The fragments are still returned: the reader can look for themselves,
    # which is the whole fallback.
    assert checked.citations


async def test_a_failed_judge_is_unknown_too(model, found):
    found([chunk()])
    model(httpx.ConnectError("no route"))

    assert (await verify.check(settings(), USER, "утверждение")).verdict == "unknown"


async def test_an_essay_instead_of_a_reason_is_cut_short(model, found):
    found([chunk()])
    model(_json({"verdict": "supported", "why": "х" * 900}))

    assert len((await verify.check(settings(), USER, "утверждение")).why) == 300


async def test_what_it_costs_is_counted(model, found):
    """The same accounting as an answer: this spends real money per claim, and
    a caller paying for it should be able to see for what."""
    found([chunk()])
    model(
        {
            "model": "google/gemini-2.5-flash-lite",
            "usage": {"prompt_tokens": 900, "completion_tokens": 12, "cost": 2.1e-5},
            "choices": [{"message": {"content": '{"verdict": "supported", "why": "да"}'}}],
        }
    )
    spend = Spend()

    await verify.check(settings(), USER, "утверждение", spend)

    assert spend.report()["cost_usd"] == 2.1e-5
