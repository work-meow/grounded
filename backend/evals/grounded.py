"""Whether the answer said anything the fragments do not.

The measurement this set needs most is not recall. Recall is 10/10 and has
been for a while; what actually went wrong in production was different. Asked
about the central bank's rate with web search on, the agent answered 8,50% —
a number that appeared in none of the sources it had been given. It looked
like a good answer, it had a citation next to it, and nothing in the harness
would have caught it.

So this asks the one question the other numbers cannot: of the facts in this
answer — the numbers, the dates, the names — which ones are not in the text
the answer was written from?

A judge, on the same cheap model as the relevance pass, with the same rule
about failure: no usable verdict means no verdict, never a failed run. A
metric that breaks the thing it measures is worse than no metric.
"""

import json
import logging
from typing import Any

from app import http, retriever
from app.config import Settings

logger = logging.getLogger(__name__)

#: How much of each fragment the judge reads. Larger than the relevance pass
#: allows itself (600) because the questions differ: that one asks what a
#: fragment is about, this one has to find a particular number inside it, and
#: a number cut off by the excerpt reads as a number that was never there.
EXCERPT_CHARS = 2000

#: A ceiling on the whole prompt's worth of fragments, so one long answer with
#: twenty fragments behind it cannot turn into a call that costs real money.
MAX_FRAGMENTS = 8

_INSTRUCTION = (
    "Ты проверяешь, опирается ли ответ на приведённые фрагменты. "
    "Найди в ответе утверждения, которых во фрагментах нет: числа, даты, суммы, "
    "сроки, имена, названия. "
    'Верни JSON вида {"unsupported": ["точная цитата из ответа", ...]} — только те '
    "куски ответа, которые фрагменты не подтверждают. "
    "Если всё подтверждается, верни пустой список. "
    "Не придирайся к формулировкам и связкам: считается только проверяемый факт. "
    "Фраза «в базе этого нет» — это не утверждение о мире, её проверять не нужно. "
    "Сегодняшнюю дату и день недели модель получает отдельно, во фрагментах их и "
    "не должно быть — это не выдумка. "
    "Ничего кроме JSON."
)


async def unsupported(
    settings: Settings,
    question: str,
    answer: str,
    chunks: list[retriever.Chunk],
) -> tuple[str, ...] | None:
    """Claims in the answer that the fragments do not support.

    Empty tuple means everything checked out. ``None`` means there was no
    verdict — no answer to check, nothing to check it against, or a judge that
    did not answer usefully. The caller must treat that as "unknown" and not
    as "fine".
    """
    if not answer.strip() or not chunks:
        return None

    numbered = "\n\n".join(
        f"[{index}] {chunk.filename or 'документ'}\n{chunk.text[:EXCERPT_CHARS]}"
        for index, chunk in enumerate(chunks[:MAX_FRAGMENTS])
    )
    asked = f"Вопрос: {question}\n\nФрагменты:\n{numbered}\n\nОтвет:\n{answer}"
    try:
        response = await http.client().post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.rerank_model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _INSTRUCTION},
                    {"role": "user", "content": asked},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=settings.rerank_timeout_s,
        )
        response.raise_for_status()
    except Exception:
        logger.exception("could not check whether an answer was grounded")
        return None
    return _claims(response.json())


def _claims(body: Any) -> tuple[str, ...] | None:
    """The judged list, cleaned. None when the shape is not one we can use."""
    try:
        content = body["choices"][0]["message"]["content"]
        listed = json.loads(content)["unsupported"]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("the groundedness judge answered in a shape this build does not know")
        return None
    if not isinstance(listed, list):
        return None
    # Strings only, trimmed, deduplicated, and bounded: a judge that decided to
    # quote the whole answer back would otherwise fill the report with it.
    seen: list[str] = []
    for value in listed:
        if not isinstance(value, str):
            continue
        claim = " ".join(value.split())[:120]
        if claim and claim not in seen:
            seen.append(claim)
    return tuple(seen[:10])
