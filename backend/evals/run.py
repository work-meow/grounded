"""Whether a change to retrieval or to the prompt made things better.

Every improvement in this system so far was measured by hand on six or nine
questions and then thrown away, so the next one started from nothing. This keeps
the questions, runs them against the real index and the real agent, and prints
two numbers: how often the right document is retrieved at all, and how often the
answer says the right thing.

Run it where the index is reachable, which is inside the API container:

    docker compose exec -T backend python evals/run.py
    docker compose exec -T backend python evals/run.py --retrieval

Deliberately not a pytest. It needs a live index with real documents in it,
it costs a fraction of a cent per question, and its job is to be run before and
after a change — not on every commit.
"""

import argparse
import asyncio
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
import sqlalchemy as sa

from app import agent, http, relevance, retriever
from app.config import Settings, get_settings
from app.db import Session

SET = Path(__file__).with_name("questions.toml")


@dataclass(frozen=True, slots=True)
class Question:
    ask: str
    #: A fragment of the filename that has to come back. Empty for a question
    #: the documents genuinely cannot answer.
    finds: str = ""
    answers: tuple[str, ...] = ()
    empty: bool = False


@dataclass
class Outcome:
    question: Question
    found: bool | None = None
    rank: int | None = None
    documents: int = 0
    #: Fragments that reached the model, and how many of them belong to the
    #: document the question is about. The second number is what relevance
    #: moves: recall was already whole, the noise around it was not.
    chunks: int = 0
    #: What the index put forward before anything judged it.
    candidates: int = 0
    on_target: int = 0
    correct: bool | None = None
    detail: str = ""
    seconds: float = 0.0


def load(path: Path) -> list[Question]:
    raw = tomllib.loads(path.read_text("utf-8"))
    return [
        Question(
            ask=entry["ask"],
            finds=entry.get("finds", ""),
            answers=tuple(entry.get("answers", ())),
            empty=bool(entry.get("empty", False)),
        )
        for entry in raw["question"]
    ]


async def only_user() -> UUID:
    """The one person this deployment belongs to.

    A personal install has exactly one, and typing a UUID on every run is a
    step that adds nothing. More than one and the caller has to say which.
    """
    async with Session() as session:
        found = (await session.execute(sa.text("select id from users order by created_at"))).all()
    if len(found) == 1:
        return found[0][0]
    raise SystemExit(
        f"в базе {len(found)} пользователей — укажите --user"
        + "".join(f"\n  {row[0]}" for row in found)
    )


async def search(
    settings: Settings, user: UUID, question: Question, k: int, *, raw: bool
) -> tuple[Outcome, list]:
    """What the model actually receives, which is the thing worth measuring.

    ``raw`` skips the judging pass, for comparing against the ranked order the
    way it was before there was one.
    """
    started = time.perf_counter()
    candidates = await retriever.retrieve(settings, user, question.ask, k)
    chunks = (
        candidates if raw else await relevance.keep_relevant(settings, question.ask, candidates)
    )
    outcome = Outcome(question=question, seconds=time.perf_counter() - started)
    outcome.candidates = len(candidates)
    outcome.documents = len({chunk.document_id for chunk in chunks})
    outcome.chunks = len(chunks)

    names = [chunk.filename or "" for chunk in chunks]
    if not question.finds:
        return outcome, chunks
    outcome.on_target = sum(question.finds.lower() in name.lower() for name in names)
    for position, name in enumerate(names, start=1):
        if question.finds.lower() in name.lower():
            outcome.found, outcome.rank = True, position
            return outcome, chunks
    outcome.found = False
    return outcome, chunks


async def answer(settings: Settings, user: UUID, question: Question, outcome: Outcome) -> None:
    started = time.perf_counter()
    text, citations = "", []
    async for kind, payload in agent.answer(settings, user, question.ask, []):
        if kind == "token":
            text += payload
        elif kind == "citations":
            citations = payload
    outcome.seconds = time.perf_counter() - started

    if question.empty:
        # Objective, and the property that actually matters: with nothing to
        # cite, an answer that cites anything is pointing the reader at a
        # fragment that does not support it.
        outcome.correct = not citations
        outcome.detail = "нет сносок" if outcome.correct else f"сослался на {len(citations)}"
        return

    lowered = text.lower()
    missing = [want for want in question.answers if want.lower() not in lowered]
    # Any one of the alternatives is enough: several ways of saying the same
    # thing should not be several requirements.
    outcome.correct = len(missing) < len(question.answers) if question.answers else bool(text)
    outcome.detail = "" if outcome.correct else f"нет ни одного из {list(question.answers)}"


def report(title: str, outcomes: list[Outcome], *, key: str) -> tuple[int, int]:
    print(f"\n=== {title}")
    good = total = 0
    for outcome in outcomes:
        value = getattr(outcome, key)
        if value is None:
            continue
        total += 1
        good += bool(value)
        mark = "✓" if value else "✗"
        note = outcome.detail
        if key == "found" and value:
            note = (
                f"на {outcome.rank}-м, своих {outcome.on_target} из {outcome.chunks}, "
                f"документов {outcome.documents}"
            )
        elif key == "found":
            note = f"ожидался «{outcome.question.finds}», документов {outcome.documents}"
        print(f"  {mark} {outcome.question.ask[:52]:<54} {outcome.seconds:4.1f}s  {note}")
    print(f"  — {good} из {total}")
    return good, total


def noise(outcomes: list[Outcome]) -> None:
    """How much of what came back was beside the point.

    Recall says whether the answer was reachable; this says how much else the
    model had to read past to reach it — and, on the questions the documents
    cannot answer, how much came back regardless. A ranked retriever always
    returns k, so that second number is k until something cuts it.
    """
    targeted = [o for o in outcomes if o.question.finds and o.found]
    if targeted:
        on_target = sum(o.on_target for o in targeted)
        total = sum(o.chunks for o in targeted)
        documents = sum(o.documents for o in targeted) / len(targeted)
        print(
            f"  — по делу {on_target} из {total} фрагментов ({100 * on_target / total:.0f}%), "
            f"документов в выдаче в среднем {documents:.1f}"
        )
    blank = [o for o in outcomes if o.question.empty]
    if blank:
        print(
            "  — на вопросах, ответа на которые нет, возвращается в среднем "
            f"{sum(o.chunks for o in blank) / len(blank):.1f} фрагментов"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Мера качества поиска и ответов")
    parser.add_argument("--user", type=UUID, default=None)
    parser.add_argument("--k", type=int, default=None, help="сколько фрагментов запрашивать")
    parser.add_argument("--retrieval", action="store_true", help="только поиск, без ответов")
    parser.add_argument("--raw", action="store_true", help="без отбора релевантных")
    parser.add_argument("--set", type=Path, default=SET)
    args = parser.parse_args()

    settings = get_settings()
    user = args.user or await only_user()
    questions = load(args.set)
    k = args.k or (settings.rerank_candidates if settings.rerank_enabled else settings.retrieve_k)

    async with httpx.AsyncClient() as client:
        http.set_client(client)
        outcomes = []
        for question in questions:
            outcome, _ = await search(settings, user, question, k, raw=args.raw)
            outcomes.append(outcome)
        title = f"поиск, кандидатов {k}" + (", без отбора" if args.raw else "")
        found, of_found = report(title, outcomes, key="found")
        noise(outcomes)

        answered = of_answered = 0
        if not args.retrieval:
            for outcome in outcomes:
                await answer(settings, user, outcome.question, outcome)
            answered, of_answered = report("ответы", outcomes, key="correct")

    print(f"\nитого: найдено {found}/{of_found}", end="")
    if of_answered:
        print(f", верных ответов {answered}/{of_answered}", end="")
    print()
    return 0 if found == of_found and answered == of_answered else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
