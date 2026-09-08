"""Whether the answer said anything its fragments do not.

This is the metric that exists because of one production failure: asked about
the central bank's rate, the agent answered 8,50% — a figure in none of the
sources it had been handed. It looked like a good answer, it had a citation
beside it, and every number in the harness said the run was perfect.

So most of what is worth testing here is the difference between the two ways
of saying nothing. An empty tuple means "checked, all of it is in the
fragments"; `None` means "no verdict" — and a caller that reads the second as
the first would report a hallucination as a clean run, which is worse than
having no metric.

`evals/` is run as a script inside the container and is not a package, so the
module is loaded by path.
"""

import importlib.util
from pathlib import Path

import httpx
import pytest

from app import http, retriever
from app.config import Settings

_spec = importlib.util.spec_from_file_location(
    "grounded", Path(__file__).parent.parent / "evals" / "grounded.py"
)
assert _spec and _spec.loader
grounded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grounded)

FRAGMENTS = [
    retriever.Chunk(
        text="Основной ежегодный отпуск — 28 календарных дней.",
        score=0.9,
        document_id="d1",
        filename="Политика.pdf",
        page=3,
    )
]
ANSWER = "Отпуск — 28 календарных дней [1]."


def settings(**overrides) -> Settings:
    return Settings(**{"jwt_secret": "x" * 40, "openrouter_api_key": "k", **overrides})


@pytest.fixture
def judge():
    """The judging model, answering whatever the test puts in it."""
    seen: list[dict] = []

    def serve(payload, *, status: int = 200):
        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            seen.append(_json.loads(request.content))
            if isinstance(payload, Exception):
                raise payload
            return httpx.Response(status, json=payload)

        http.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return seen

    yield serve
    http.set_client(None)


def _verdict(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


async def test_an_answer_out_of_its_fragments_is_clean(judge):
    judge(_verdict('{"unsupported": []}'))

    assert await grounded.unsupported(settings(), "отпуск", ANSWER, FRAGMENTS) == ()


async def test_a_claim_the_fragments_do_not_carry_comes_back(judge):
    judge(_verdict('{"unsupported": ["премия 15 000 рублей"]}'))

    found = await grounded.unsupported(settings(), "отпуск", ANSWER, FRAGMENTS)

    assert found == ("премия 15 000 рублей",)


@pytest.mark.parametrize(
    "payload",
    [
        _verdict("не json"),
        _verdict('{"relevant": []}'),
        _verdict('{"unsupported": "строка"}'),
        {"choices": []},
        {},
    ],
)
async def test_a_judge_that_did_not_answer_usefully_leaves_no_verdict(judge, payload):
    """None, and never an empty tuple: "nobody checked" read as "nothing wrong"
    is how a metric starts lying about the thing it exists for."""
    judge(payload)

    assert await grounded.unsupported(settings(), "отпуск", ANSWER, FRAGMENTS) is None


async def test_a_judge_that_failed_costs_the_verdict_and_not_the_run(judge):
    judge(httpx.ConnectError("no route"))

    assert await grounded.unsupported(settings(), "отпуск", ANSWER, FRAGMENTS) is None


async def test_an_http_error_is_the_same_as_no_verdict(judge):
    judge({"error": "rate limited"}, status=429)

    assert await grounded.unsupported(settings(), "отпуск", ANSWER, FRAGMENTS) is None


async def test_nothing_to_check_is_not_a_clean_bill():
    assert await grounded.unsupported(settings(), "отпуск", "", FRAGMENTS) is None
    assert await grounded.unsupported(settings(), "отпуск", "   ", FRAGMENTS) is None
    assert await grounded.unsupported(settings(), "отпуск", ANSWER, []) is None


async def test_junk_inside_a_good_shape_is_cleaned(judge):
    judge(
        _verdict(
            '{"unsupported": ["  премия   15 000 ", 42, null, "премия 15 000", '
            '"' + "х" * 300 + '"]}'
        )
    )

    found = await grounded.unsupported(settings(), "отпуск", ANSWER, FRAGMENTS)

    # Whitespace collapsed, so the same claim twice is one claim; non-strings
    # dropped; a judge that quoted an essay back is cut short.
    assert found is not None
    assert found[0] == "премия 15 000"
    assert len(found) == 2
    assert all(len(claim) <= 120 for claim in found)


async def test_only_a_few_fragments_reach_the_judge(judge):
    """One long answer with twenty fragments behind it must not turn into a
    call that costs real money."""
    seen = judge(_verdict('{"unsupported": []}'))
    many = [
        retriever.Chunk(
            text=f"фрагмент {n}", score=0.1, document_id=f"d{n}", filename="Х.md", page=None
        )
        for n in range(30)
    ]

    await grounded.unsupported(settings(), "отпуск", ANSWER, many)

    asked = seen[0]["messages"][1]["content"]
    assert asked.count("] Х.md") == grounded.MAX_FRAGMENTS
