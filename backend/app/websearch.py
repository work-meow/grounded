"""Looking something up on the open web.

The knowledge base answers questions about the user's own documents and nothing
else. Everything time-bound — a rate today, a law as amended, who won on
Tuesday — is outside it by construction, and the honest answer used to be "в
базе нет информации" for facts that are three seconds away.

OpenRouter runs the search for us: a chat completion with the ``web`` plugin
gets the query, does the searching, and comes back with a grounded summary plus
the pages it read. That means one request rather than a search API, a scraper
and a summariser of our own.

Three things were measured before being settled on, not read off a page:

* Gemini has **no** native search on OpenRouter — the request is refused with a
  404 naming the models that do. So an external engine it is.
* Parallel in turbo mode costs $0.00125 a call against Exa's $0.0072, comes
  back in 2.2 s against 3.3 s, and returns more text per source (400–3000
  characters against 150–1700). Nearly six times cheaper and better.
* The ``plugins`` form beats the newer ``tools: [{"type":
  "openrouter:web_search"}]`` one, which is the interesting result, because the
  tool form is what the documentation now recommends. Across six questions with
  a checkable answer, the plugin was right six times; the tool form got the
  central bank's rate and today's own date wrong, and twice answered with no
  sources at all and a bill of four thousandths of a cent — it simply decided
  not to search.

That last one is not really about search quality. Under ``plugins`` the search
happens before the model runs; under ``tools`` it is the model's decision, and
this is a small cheap model being asked to second-guess a decision that has
already been made. By the time this module is called, the agent has decided a
web search is warranted — visibly, as a step on screen, against a ceiling — and
handing that judgement back down to a cheaper model is both redundant and worse.
It would also take the queries off the screen, which is most of what makes the
agent legible while it works.

A small model does the searching — the agent above it is what writes the
answer, and it only needs the findings.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from app import http, retriever
from app.config import Settings

logger = logging.getLogger(__name__)

#: How much of one page reaches the agent. Sources come back at up to three
#: thousand characters each; four of those unabridged is most of a context
#: window spent on one tool call, and the part that answers a question is
#: almost always near the top of what the search engine chose to return.
SOURCE_CHARS = 1200

#: A ceiling on the searching model's own answer. It is relaying findings, not
#: writing the reply, so a long one is tokens spent on prose nobody reads.
SUMMARY_TOKENS = 700

_INSTRUCTION = (
    "Ты — поисковый помощник. Кратко изложи, что нашлось в интернете по запросу, "
    "фактами и числами, с датами там, где они есть. Не рассуждай и не советуй."
)

#: Markdown links in the summary, which the searching model adds unprompted.
#: The interface renders answers as plain text, so `[домен](url)` would reach
#: the user as those exact characters — and the numbered citations this system
#: uses are the same brackets, which is the more expensive confusion of the two.
_MD_LINK = re.compile(r"\[([^\]\n]{1,120})\]\((https?://[^)\s]+)\)")


@dataclass(frozen=True, slots=True)
class Source:
    """One page the search read."""

    url: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Result:
    #: What the searching model made of it. May be empty.
    summary: str
    sources: list[Source]


async def search(settings: Settings, query: str) -> Result:
    """Search the web for one query. Raises on transport or HTTP failure."""
    response = await http.client().post(
        f"{settings.openrouter_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={
            "model": settings.web_search_model,
            "messages": [
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": query},
            ],
            "plugins": [_plugin(settings)],
            "max_tokens": SUMMARY_TOKENS,
            "temperature": 0.0,
        },
        timeout=settings.web_search_timeout_s,
    )
    response.raise_for_status()
    return _result(response.json())


def _plugin(settings: Settings) -> dict[str, Any]:
    plugin: dict[str, Any] = {
        "id": "web",
        "engine": settings.web_search_engine,
        "max_results": settings.web_search_results,
    }
    if settings.web_search_engine == "parallel":
        # Turbo is the tier that was measured. Parallel's other modes cost five
        # times as much, for depth this does not need: the agent asks a
        # specific question and reads the answer, it does not research.
        plugin["mode"] = "turbo"
    return plugin


def _result(body: Any) -> Result:
    """What came back, reduced to what the agent can use. Never raises."""
    message = _message(body)
    sources: list[Source] = []
    seen: set[str] = set()

    for annotation in message.get("annotations") or []:
        if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
            continue
        citation = annotation.get("url_citation")
        if not isinstance(citation, dict):
            continue
        # The same rule as a document link: this url ends up in window.open,
        # where a `javascript:` one would run in a document inheriting our
        # origin. A search engine would not send one; a trust boundary is not
        # the place to rely on that.
        url = retriever.openable(citation.get("url"))
        if url is None or url in seen:
            continue
        seen.add(url)
        sources.append(
            Source(
                url=url,
                title=str(citation.get("title") or "").strip() or host(url),
                text=" ".join(str(citation.get("content") or "").split())[:SOURCE_CHARS],
            )
        )

    return Result(summary=_plain(str(message.get("content") or "")), sources=sources)


def _message(body: Any) -> dict[str, Any]:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        logger.warning("a web search came back in a shape this build does not know")
        return {}
    return message if isinstance(message, dict) else {}


def _plain(text: str) -> str:
    return _MD_LINK.sub(r"\1", text).strip()


def host(url: str) -> str:
    """The site, for a citation chip too narrow for a headline."""
    without_scheme = url.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0].removeprefix("www.") or url
