"""Asking the index the same question in more than one way.

The first search a turn makes is the user's sentence, verbatim. The agent is
told to try again with different wording when the fragments do not answer —
and measured on the deployment, that instruction is what turns one search into
a loop worth having. But it only ever fires *after* a search has come back
useless, which costs a round trip and a judging pass to find out.

This is the other half. One cheap call turns the question into a couple of
alternative phrasings, all of them go to the index, and the results are merged
before anything is judged. The point is not that a paraphrase is better than
the question: it is that BM25 and the embedder fail on different things.
"Сколько дней отпуска" and "продолжительность ежегодного отпуска" retrieve
different fragments, and the union is what the judge should be choosing from.

Merged with reciprocal rank fusion — the same 1/(60+rank) the hybrid index
itself uses to combine vectors with BM25, and for the same reason: the scores
that come back are positions in a list, not measures of anything, so they
cannot be compared across two searches but their ranks can.

Never fatal, twice over: a failed rewrite leaves the question alone, and a
failed search among several leaves the others.
"""

import asyncio
import json
import logging

from app import http, retriever
from app.config import Settings

logger = logging.getLogger(__name__)

#: The constant in reciprocal rank fusion. 60 because that is what Pathway's
#: HybridIndexFactory uses on the other side of this call, and two different
#: constants fusing the same ranks would be an arbitrary disagreement.
RRF_K = 60

_INSTRUCTION = (
    "Ты переформулируешь поисковый запрос для поиска по документам. "
    'Верни JSON вида {"queries": ["...", "..."]} — две короткие '
    "альтернативные формулировки того же вопроса: синонимы, официальные "
    "термины, без вопросительных слов. Не отвечай на вопрос и не добавляй "
    "ничего, чего в нём нет. На языке вопроса. Ничего кроме JSON."
)


async def variants(settings: Settings, question: str) -> list[str]:
    """The question, plus a couple of other ways of asking it.

    The original is always first and always present: a rewrite that came back
    empty, wrong or not at all must not be able to replace it.
    """
    if not settings.expand_queries:
        return [question]
    try:
        response = await http.client().post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.rerank_model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _INSTRUCTION},
                    {"role": "user", "content": question},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=settings.rerank_timeout_s,
        )
        response.raise_for_status()
    except Exception:
        logger.exception("could not rephrase a query; searching for it as asked")
        return [question]
    return [question, *_rephrased(response.json(), question, settings.expand_to)]


def _rephrased(body: object, question: str, limit: int) -> list[str]:
    """The usable rewrites: strings, new, and not the question again."""
    try:
        content = body["choices"][0]["message"]["content"]  # type: ignore[index]
        listed = json.loads(content)["queries"]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("a query rewrite came back in a shape this build does not know")
        return []
    if not isinstance(listed, list):
        return []

    seen = {" ".join(question.lower().split())}
    kept: list[str] = []
    for value in listed:
        if not isinstance(value, str):
            continue
        # Bounded like the question itself is: a rewrite is a query, and a
        # model that decided to write an essay must not send one to the index.
        query = " ".join(value.split())[:500]
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        kept.append(query)
    return kept[:limit]


async def search(
    settings: Settings,
    user_id,
    question: str,
    k: int,
    **narrowing,
) -> list[retriever.Chunk]:
    """The question asked several ways, merged into one ranked list.

    Falls back to exactly the old behaviour — one search for the question as
    asked — when expansion is off or the rewrite failed, so nothing here can
    make a search worse than not having it.
    """
    queries = await variants(settings, question)
    if len(queries) == 1:
        return await retriever.retrieve(settings, user_id, question, k, **narrowing)

    # Concurrently: they are independent requests to a local index, and doing
    # them in sequence would add its latency to every search.
    found = await asyncio.gather(
        *(retriever.retrieve(settings, user_id, query, k, **narrowing) for query in queries),
        return_exceptions=True,
    )
    usable: list[list[retriever.Chunk]] = []
    for query, result in zip(queries, found, strict=True):
        if isinstance(result, BaseException):
            # One phrasing failing is not the search failing — unless they all
            # do, which the caller finds out as an empty list from a raise
            # below.
            logger.warning("a search for %r failed inside an expanded query", query[:60])
            continue
        usable.append(result)
    if not usable:
        # Every one of them failed, which is the index being unreachable. The
        # caller's own error handling exists for exactly this, so let it see it.
        return await retriever.retrieve(settings, user_id, question, k, **narrowing)
    return fuse(usable, k)


def fuse(rankings: list[list[retriever.Chunk]], k: int) -> list[retriever.Chunk]:
    """Several ranked lists as one, by reciprocal rank fusion.

    A chunk found by two phrasings outranks one found by either alone, which is
    the whole reason to ask twice. Identity is (document, page) rather than the
    text: the same fragment retrieved twice must not become two.
    """
    scores: dict[tuple[str | None, int | None], float] = {}
    best: dict[tuple[str | None, int | None], retriever.Chunk] = {}
    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            key = (chunk.document_id, chunk.page)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + position)
            # Keep the first copy seen: they are the same fragment, and the
            # earlier ranking is the one that ranked it higher.
            best.setdefault(key, chunk)
    order = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [best[key] for key in order[:k]]
