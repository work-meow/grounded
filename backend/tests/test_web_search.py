"""Looking things up outside the knowledge base.

Three things are worth pinning. What reaches the agent — because a search
result is text from strangers, and the url in it ends up in window.open. What
it costs — the tool is bound only when the person asked for it, and bounded
when it is. And what happens when the search fails, which must be an answer
from the documents rather than a failed turn.
"""

import json
import uuid

import httpx
import pytest

from app import agent, http, retriever, websearch
from app.config import Settings

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


def settings(**overrides) -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k", **overrides)


def _body(content: str = "Курс 86,58 рубля.", annotations: list | None = None) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "annotations": annotations
                    if annotations is not None
                    else [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://www.cbr.ru/currency/",
                                "title": "Официальные курсы валют",
                                "content": "Курс доллара США на 5 сентября — 86,5857 рубля.",
                            },
                        }
                    ],
                }
            }
        ]
    }


@pytest.fixture
def openrouter():
    """OpenRouter, answering whatever the test puts in it."""

    def serve(handler):
        http.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    yield serve
    http.set_client(None)


def _returning(body: dict, *, status: int = 200, seen: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.update(url=str(request.url), payload=json.loads(request.content))
        return httpx.Response(status, json=body)

    return handler


# --- what comes back ---------------------------------------------------------


async def test_a_search_returns_its_summary_and_its_sources(openrouter):
    openrouter(_returning(_body()))

    found = await websearch.search(settings(), "курс доллара")

    assert found.summary == "Курс 86,58 рубля."
    (source,) = found.sources
    assert source.url == "https://www.cbr.ru/currency/"
    assert "86,5857" in source.text


async def test_a_link_that_is_not_a_link_is_dropped(openrouter):
    """The url goes into window.open, where a `javascript:` one would run in a
    document inheriting our origin. A search engine would not send one; a trust
    boundary is not the place to rely on that."""
    openrouter(
        _returning(
            _body(
                annotations=[
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "javascript:alert(1)",
                            "title": "x",
                            "content": "y",
                        },
                    },
                    {
                        "type": "url_citation",
                        "url_citation": {"url": "https://ok.test/a", "title": "x", "content": "y"},
                    },
                ]
            )
        )
    )

    found = await websearch.search(settings(), "что угодно")

    assert [source.url for source in found.sources] == ["https://ok.test/a"]


async def test_the_same_page_twice_is_one_source(openrouter):
    same = {
        "type": "url_citation",
        "url_citation": {"url": "https://ok.test/a", "title": "x", "content": "y"},
    }
    openrouter(_returning(_body(annotations=[same, same])))

    assert len((await websearch.search(settings(), "q")).sources) == 1


async def test_markdown_links_do_not_survive_into_the_answer(openrouter):
    """The interface renders answers as plain text, so `[домен](url)` would
    reach the reader as those characters — and the numbered citations this
    system uses are the same brackets, which is the worse confusion."""
    openrouter(_returning(_body(content="По данным [cbr.ru](https://cbr.ru/x) курс вырос.")))

    found = await websearch.search(settings(), "курс")

    assert found.summary == "По данным cbr.ru курс вырос."


async def test_a_source_is_trimmed_before_it_reaches_the_agent(openrouter):
    """Four sources at three thousand characters each is most of a context
    window spent on one tool call."""
    openrouter(
        _returning(
            _body(
                annotations=[
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "https://ok.test/a",
                            "title": "x",
                            "content": "я" * 9000,
                        },
                    }
                ]
            )
        )
    )

    (source,) = (await websearch.search(settings(), "q")).sources

    assert len(source.text) == websearch.SOURCE_CHARS


@pytest.mark.parametrize(
    "body", [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": None}]}]
)
async def test_a_shape_this_build_does_not_know_is_empty_rather_than_fatal(openrouter, body):
    openrouter(_returning(body))

    found = await websearch.search(settings(), "q")

    assert found.summary == "" and found.sources == []


async def test_an_http_failure_reaches_the_caller(openrouter):
    """The tool above catches it; this layer must not swallow it silently."""
    openrouter(_returning({}, status=500))

    with pytest.raises(httpx.HTTPStatusError):
        await websearch.search(settings(), "q")


# --- what is asked for -------------------------------------------------------


async def test_the_request_names_the_engine_and_the_model(openrouter):
    seen: dict = {}
    openrouter(_returning(_body(), seen=seen))

    await websearch.search(settings(), "курс доллара")

    assert seen["url"].endswith("/chat/completions")
    assert seen["payload"]["model"] == "google/gemini-2.5-flash-lite"
    (plugin,) = seen["payload"]["plugins"]
    assert plugin["id"] == "web"
    assert plugin["engine"] == "parallel"
    # Turbo is the tier that was measured, and five times cheaper than the rest.
    assert plugin["mode"] == "turbo"


async def test_another_engine_is_not_sent_parallels_own_option(openrouter):
    seen: dict = {}
    openrouter(_returning(_body(), seen=seen))

    await websearch.search(settings(web_search_engine="exa"), "q")

    (plugin,) = seen["payload"]["plugins"]
    assert plugin["engine"] == "exa"
    assert "mode" not in plugin


# --- the tool the agent sees -------------------------------------------------


def _tools(web: bool) -> dict:
    built = agent._build_tools(settings(), USER, agent._Citations(), web)
    return {tool.name: tool for tool in built}


def test_the_tool_exists_only_when_it_was_asked_for():
    """It costs money per call, so a turn that did not ask for it must not be
    able to spend any — and a model is never told about a tool it does not
    have, which is how one ends up apologising for not calling it."""
    assert "search_web" not in _tools(web=False)
    assert "search_web" in _tools(web=True)


def test_the_web_is_bounded_more_tightly_than_the_index():
    """A knowledge search is a tenth of a cent's worth of embedding; a web
    search is about $0.00125 of somebody else's search API."""
    limits = agent._limits(settings(), web=True)
    per_tool = {
        getattr(limit, "tool_name", None): limit
        for limit in limits
        if getattr(limit, "tool_name", None)
    }

    assert set(per_tool) == {"search_knowledge", "search_web"}
    assert len(agent._limits(settings(), web=False)) == len(limits) - 1


async def test_a_failed_search_leaves_the_turn_answerable(openrouter):
    """A tool that raises takes the whole answer with it. The documents are
    still there, and saying so is better than losing the turn."""
    openrouter(_returning({}, status=503))

    answer = await _tools(web=True)["search_web"].ainvoke({"query": "курс"})

    assert "не сработал" in answer
    assert "базе знаний" in answer


async def test_web_results_are_numbered_alongside_the_documents(openrouter):
    """An answer that leans on a document and a news item cites [1] and [2]
    without the reader needing to know which is which."""
    openrouter(_returning(_body()))
    citations = agent._Citations()
    citations.add(
        retriever.Chunk(
            text="из документа",
            score=1.0,
            document_id=str(uuid.uuid4()),
            filename="Договор.pdf",
            page=2,
        )
    )
    tools = {tool.name: tool for tool in agent._build_tools(settings(), USER, citations, True)}

    rendered = await tools["search_web"].ainvoke({"query": "курс"})

    assert "[2]" in rendered
    assert [item["n"] for item in citations.items] == [1, 2]
    assert citations.items[1]["url"] == "https://www.cbr.ru/currency/"
    assert citations.items[0]["url"] is None
