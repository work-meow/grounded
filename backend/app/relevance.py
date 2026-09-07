"""What of the retrieved set actually reaches the model.

The index is a ranked retriever: it answers any query with its k best fragments,
whether or not any of them is about the question. Measured on this deployment,
across thirteen questions, that came to nineteen useful fragments out of eighty
— three quarters of what the model read was about something else, and for a
question the documents cannot answer at all it still received eight confident
paragraphs and had to work out on its own that none of them helped.

So a second pass reads the candidates and says which ones bear on the question.
A small model, on the same provider as everything else: measured at 0.3–0.7 s
and about $0.00017 a call, a sixth of what a web search costs.

The part that matters most is not the ordering, it is that the answer can be
none. "В базе этого нет" stops being the model's opinion about eight paragraphs
and becomes a fact about the retrieval — and, with web search enabled, the
honest signal to go and look outside.

Never fatal: a reranker that fails returns the candidates untouched, which is
exactly the behaviour this replaces.
"""

import json
import logging
from typing import Any

from app import http, retriever
from app.config import Settings
from app.spend import RERANK, Spend

logger = logging.getLogger(__name__)

#: How much of each candidate the judge reads. Enough to tell what a fragment is
#: about — which is all it is being asked — and short enough that twenty of them
#: stay one cheap call.
EXCERPT_CHARS = 600

_INSTRUCTION = (
    "Ты отбираешь фрагменты, которые действительно относятся к вопросу. "
    'Верни JSON вида {"relevant": [номера]} — только номера тех фрагментов, '
    "по которым можно ответить на вопрос, от самого полезного к менее полезному. "
    "Если ни один не относится к вопросу, верни пустой список. Ничего кроме JSON."
)


async def keep_relevant(
    settings: Settings,
    question: str,
    chunks: list[retriever.Chunk],
    spend: Spend | None = None,
) -> list[retriever.Chunk]:
    """The candidates that bear on the question, best first. May be empty.

    ``spend`` collects what this call cost when the caller is reporting a bill.
    Optional because the search page does not: it shows a page of results, not
    an invoice, and nothing there would read the number.
    """
    if not settings.rerank_enabled or not chunks:
        return chunks[: settings.rerank_keep]

    try:
        chosen = await _ask(settings, question, chunks, spend)
    except Exception:
        # Degrading to the ranked order is exactly what this replaces, so a
        # failure here costs precision and never an answer.
        logger.exception("could not rerank %d candidates", len(chunks))
        return chunks[: settings.rerank_keep]

    if chosen is None:
        return chunks[: settings.rerank_keep]
    return [chunks[index] for index in chosen][: settings.rerank_keep]


async def _ask(
    settings: Settings,
    question: str,
    chunks: list[retriever.Chunk],
    spend: Spend | None = None,
) -> list[int] | None:
    """The indices the judge picked, or None if it did not answer usefully.

    None and [] are different answers: none means "no verdict, keep what the
    index gave you", empty means "the index gave you nothing about this".
    """
    numbered = "\n\n".join(
        f"[{index}] {chunk.filename or 'документ'}\n{chunk.text[:EXCERPT_CHARS]}"
        for index, chunk in enumerate(chunks)
    )
    response = await http.client().post(
        f"{settings.openrouter_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={
            "model": settings.rerank_model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": f"Вопрос: {question}\n\n{numbered}"},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=settings.rerank_timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    if spend is not None:
        spend.add_completion(RERANK, settings.rerank_model, body)
    return _indices(body, len(chunks))


def _indices(body: Any, total: int) -> list[int] | None:
    """The judged order, cleaned. None when the shape is not one we can use."""
    try:
        content = body["choices"][0]["message"]["content"]
        relevant = json.loads(content)["relevant"]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("the reranker answered in a shape this build does not know")
        return None
    if not isinstance(relevant, list):
        return None

    seen: list[int] = []
    for value in relevant:
        # A judge that invents an index would otherwise take an unrelated
        # fragment with it, and one that repeats an index would show the same
        # fragment twice.
        if isinstance(value, int) and 0 <= value < total and value not in seen:
            seen.append(value)
    return seen


def cap_per_document(chunks: list[retriever.Chunk], limit: int) -> list[retriever.Chunk]:
    """At most ``limit`` fragments from any one document, order preserved.

    For browsing, not for answering. A long document legitimately fills the
    whole answer to a question about it — measured, six of eight fragments —
    and trimming that would take away the parts that answer it. A list somebody
    is scanning for *where* something is written wants the opposite: five
    documents beat the same document five times.
    """
    kept: list[retriever.Chunk] = []
    seen: dict[str | None, int] = {}
    for chunk in chunks:
        count = seen.get(chunk.document_id, 0)
        if count >= limit:
            continue
        seen[chunk.document_id] = count + 1
        kept.append(chunk)
    return kept
