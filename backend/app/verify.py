"""Checking a piece of text against the documents.

The chat answers questions. This answers a different one: here is a paragraph
somebody wrote — a draft announcement, a summary of a policy, a claim in a
meeting — which parts of it do our own documents actually support?

It is the same promise as the chat's, turned around. Instead of "answer only
from the documents", it is "tell me which of this the documents agree with,
disagree with, or have never heard of". Nothing new had to be built for it:
splitting text into claims is one cheap call, finding the fragments for a claim
is the same retrieval every search uses, and judging support against fragments
is the pattern the relevance pass and the groundedness check already share.

The three verdicts are deliberately not two. "Absent" and "contradicted" are
different situations for the person reading — one means go and write the
document, the other means somebody is about to say something that is on record
as false — and collapsing them into "unsupported" would hide the more
expensive one.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from app import expansion, http, relevance, retriever
from app.config import Settings
from app.spend import RERANK, Spend

logger = logging.getLogger(__name__)

Verdict = Literal["supported", "contradicted", "absent", "unknown"]

#: How many claims one request may be broken into. A page of prose is about
#: ten; the ceiling is here because each one costs a retrieval and a judging
#: call, so a pasted book would otherwise be a bill nobody agreed to.
MAX_CLAIMS = 20

#: How much of each fragment the judge reads for one claim. Larger than the
#: relevance pass allows itself, for the same reason the groundedness check is:
#: this has to find a particular figure inside a fragment, not decide what the
#: fragment is about.
EXCERPT_CHARS = 1500

_SPLIT = (
    "Раздели текст на отдельные проверяемые утверждения. "
    'Верни JSON вида {"claims": ["...", "..."]} — каждое утверждение отдельной '
    "строкой, своими словами автора, без нумерации. Пропускай вводные слова, "
    "вопросы и то, что нельзя проверить по документам. Ничего кроме JSON."
)

_JUDGE = (
    "Ты проверяешь одно утверждение по фрагментам документов. "
    'Верни JSON вида {"verdict": "supported|contradicted|absent", "why": "одна фраза"}. '
    "supported — фрагменты подтверждают утверждение; "
    "contradicted — во фрагментах сказано иное; "
    "absent — во фрагментах об этом ничего нет. "
    "В «why» цитируй то место фрагмента, на котором основан вывод. "
    "Не додумывай и не опирайся на общие знания. Ничего кроме JSON."
)


@dataclass(frozen=True, slots=True)
class Checked:
    """One claim and what the documents said about it."""

    claim: str
    verdict: Verdict
    why: str
    #: The fragments the verdict was based on, numbered as citations are.
    citations: list[dict[str, Any]]


async def claims(settings: Settings, text: str, spend: Spend | None = None) -> list[str]:
    """The text broken into things that can be checked.

    Falls back to the whole text as a single claim: a paragraph checked as one
    lump is a worse answer than the same paragraph checked in pieces, and a far
    better one than an error.
    """
    try:
        response = await http.client().post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.rerank_model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _SPLIT},
                    {"role": "user", "content": text},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=settings.rerank_timeout_s,
        )
        response.raise_for_status()
    except Exception:
        logger.exception("could not split a text into claims; checking it whole")
        return [text]
    body = response.json()
    if spend is not None:
        spend.add_completion(RERANK, settings.rerank_model, body)
    return _listed(body) or [text]


def _listed(body: Any) -> list[str]:
    try:
        content = body["choices"][0]["message"]["content"]
        listed = json.loads(content)["claims"]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("a claim split came back in a shape this build does not know")
        return []
    if not isinstance(listed, list):
        return []
    kept: list[str] = []
    for value in listed:
        if not isinstance(value, str):
            continue
        claim = " ".join(value.split())[:1000]
        if claim and claim not in kept:
            kept.append(claim)
    return kept[:MAX_CLAIMS]


async def check(
    settings: Settings,
    user_id: UUID,
    claim: str,
    spend: Spend | None = None,
) -> Checked:
    """One claim, against whatever the documents have to say about it."""
    found = await expansion.search(settings, user_id, claim, settings.rerank_candidates)
    chunks = await relevance.keep_relevant(settings, claim, found, spend)
    if not chunks:
        # Nothing in the base bears on it — which is the "absent" verdict
        # arrived at without spending a call to be told so.
        return Checked(claim=claim, verdict="absent", why="", citations=[])

    verdict, why = await _judged(settings, claim, chunks, spend)
    return Checked(
        claim=claim,
        verdict=verdict,
        why=why,
        citations=[
            {
                "n": number,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "snippet": chunk.text[:300],
            }
            for number, chunk in enumerate(chunks, start=1)
        ],
    )


async def _judged(
    settings: Settings, claim: str, chunks: list[retriever.Chunk], spend: Spend | None
) -> tuple[Verdict, str]:
    """The verdict, or "unknown" when there is no usable one.

    Not "absent": a judge that failed has told us nothing about the documents,
    and reporting that as "the documents do not mention this" would be the one
    mistake that matters here — somebody would go and write a document that
    already exists.
    """
    numbered = "\n\n".join(
        f"[{index}] {chunk.filename or 'документ'}\n{chunk.text[:EXCERPT_CHARS]}"
        for index, chunk in enumerate(chunks, start=1)
    )
    try:
        response = await http.client().post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.rerank_model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _JUDGE},
                    {"role": "user", "content": f"Утверждение: {claim}\n\nФрагменты:\n{numbered}"},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=settings.rerank_timeout_s,
        )
        response.raise_for_status()
    except Exception:
        logger.exception("could not judge a claim")
        return "unknown", ""
    body = response.json()
    if spend is not None:
        spend.add_completion(RERANK, settings.rerank_model, body)
    return _verdict(body)


def _verdict(body: Any) -> tuple[Verdict, str]:
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        verdict = parsed["verdict"]
        why = parsed.get("why", "")
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("a claim verdict came back in a shape this build does not know")
        return "unknown", ""
    if verdict not in ("supported", "contradicted", "absent"):
        return "unknown", ""
    return verdict, " ".join(str(why).split())[:300]
