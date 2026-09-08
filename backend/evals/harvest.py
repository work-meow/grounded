"""Turn real traffic into a set the measuring script can run.

The shipped set is thirteen questions over five invented documents, and it
passes 10/10 and 13/13 — which means it can no longer tell a better retriever
from a worse one. Meanwhile every question anybody has actually asked is
sitting in `messages`, with the answer it got and the documents that answer
cited, and nothing has ever read it.

This writes those questions out in the same shape `run.py` reads, so they go
through the same harness:

    docker compose exec -T backend python evals/harvest.py
    docker compose exec -T backend python evals/run.py --set evals/local/harvested.toml

What `finds` means here is worth being exact about. It is not ground truth —
nobody labelled these. It is the document the answer *did* cite, which makes
the set a **baseline**: a change that stops finding what used to be found
shows up, and an answer that was wrong all along does not. That is still far
more than thirteen questions everything passes.

`answers` is deliberately left out. Guessing at the right words from an answer
that may itself be wrong would bake a mistake into the test; `run.py` then
counts a non-empty answer as correct, and whether it is *grounded* is measured
separately by `grounded.py`.

The output goes under `evals/local/`, which is gitignored. These are somebody's
own questions about their own documents — this set has already had to be thrown
away once for being built out of private notes, and it is not happening twice.
"""

import argparse
import asyncio
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import sqlalchemy as sa

from app.db import Session
from app.models import Chat, Message

OUT = Path(__file__).parent / "local" / "harvested.toml"

#: Questions shorter than this are "ку", "спасибо", "?" — real messages, but
#: nothing a retriever can be judged on.
MIN_QUESTION_CHARS = 12


@dataclass(frozen=True, slots=True)
class Turn:
    """One question and what became of it."""

    ask: str
    #: The document the answer actually cited, or empty when it cited nothing.
    finds: str
    #: True when the answer cited nothing at all — the base had no answer.
    empty: bool
    #: -1, 1 or None, as somebody left it.
    rated: int | None
    when: datetime


async def collected(user: UUID | None) -> list[Turn]:
    """Every question with the answer that followed it, oldest first."""
    async with Session() as session:
        query = (
            sa.select(
                Message.chat_id,
                Message.role,
                Message.content,
                Message.citations,
                Message.rating,
                Message.created_at,
                Message.trace,
            )
            .join(Chat, Chat.id == Message.chat_id)
            # chat_id first: the pairing below walks one conversation at a time,
            # and interleaved chats would pair a question with somebody else's
            # answer. The id breaks a tie inside one microsecond.
            .order_by(Message.chat_id, Message.created_at, Message.id)
        )
        if user is not None:
            query = query.where(Chat.user_id == user)
        rows = (await session.execute(query)).all()
    return pair(rows)


#: Tools that mean the question was actually put to the documents. A turn that
#: called neither was not a retrieval question — "какое сегодня число" and
#: "какие у нас документы" cite nothing and never could, and counting them as
#: gaps would send somebody looking for a document that does not exist.
_SEARCHED = ("search_knowledge", "read_document")


#: One row as the query above returns it: chat, role, text, citations, rating,
#: when, trace.
Row = tuple[Any, str, str, list, int | None, datetime, dict | None]


def pair(rows: Sequence[Row]) -> list[Turn]:
    """Questions matched with the answers that followed them.

    Its own function, and not part of the query above, because this is the part
    that can be wrong in a way nothing would notice: a question paired with
    somebody else's answer puts an unrelated document in `finds`, and the set
    then measures noise while looking perfectly healthy. Pure input, pure
    output, tested without a database.

    Expects the rows ordered by chat and then by time.
    """
    turns: list[Turn] = []
    pending: tuple[str, datetime] | None = None
    chat: Any = None
    for chat_id, role, content, citations, rating, created_at, trace in rows:
        if chat_id != chat:
            chat, pending = chat_id, None
        if role == "user":
            pending = (content.strip(), created_at)
            continue
        if pending is None:
            # An answer with no question above it — a chat whose beginning was
            # deleted. Nothing to ask again.
            continue
        ask, asked_at = pending
        pending = None
        if len(ask) < MIN_QUESTION_CHARS:
            continue
        if _asked_nothing_of_the_documents(trace):
            continue
        names = _documents(citations)
        if _only_from_the_web(citations):
            # The answer came off the open web. Its "source" is a page title,
            # and asking the document index to return that is a question with
            # no right answer — measured once as 10/27 before this was here,
            # with several of the misses being questions about today's bitcoin
            # price. A web question is a fine question; it is not a retrieval
            # case.
            continue
        turns.append(
            Turn(
                ask=ask,
                finds=names[0] if names else "",
                empty=not names,
                rated=rating,
                when=asked_at,
            )
        )
    return turns


def _documents(citations: list | None) -> list[str]:
    """Filenames of the cited fragments that came from the knowledge base.

    A citation with a url and no document_id is a page from the open web. It
    has a title, the title looks like a filename, and nothing in the index
    will ever match it.
    """
    return [
        str(item.get("filename") or "")
        for item in citations or []
        if isinstance(item, dict) and item.get("document_id")
    ]


def _only_from_the_web(citations: list | None) -> bool:
    """Whether every source under this answer was a web page."""
    listed = [item for item in citations or [] if isinstance(item, dict)]
    return bool(listed) and not any(item.get("document_id") for item in listed)


def _asked_nothing_of_the_documents(trace: dict | None) -> bool:
    """Whether this turn never put the question to the base at all.

    Answerable only for turns recorded since the trace existed. Before that
    there is no way to tell a question the documents could not answer from one
    that was never about them, so those are kept and the summary says so.
    """
    if not isinstance(trace, dict):
        return False
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        return True
    return not any(step.get("tool") in _SEARCHED for step in steps if isinstance(step, dict))


def chosen(turns: list[Turn]) -> list[Turn]:
    """One entry per question, the most informative version of it.

    The same thing gets asked repeatedly — the base changes, the wording
    changes, the answer changes. Keeping every copy would weight the set
    towards whatever somebody was stuck on that week.

    A rated turn wins over an unrated one, because somebody looked at it; among
    equals the most recent wins, since it was asked against the base as it is
    now.
    """
    best: dict[str, Turn] = {}
    for turn in turns:
        key = " ".join(turn.ask.lower().split())
        current = best.get(key)
        if current is None or _rank(turn) > _rank(current):
            best[key] = turn
    return sorted(best.values(), key=_rank, reverse=True)


def _rank(turn: Turn) -> tuple[int, datetime]:
    # A thumb down first of all: it is the one signal a person left on purpose,
    # and a question somebody marked wrong is the question worth re-running.
    opinion = 2 if turn.rated == -1 else 1 if turn.rated else 0
    return opinion, turn.when


def quote(value: str) -> str:
    """A TOML basic string.

    Hand-rolled because the standard library reads TOML and does not write it,
    and one function is cheaper than a dependency. Control characters are
    dropped rather than escaped: they are not in a question anybody typed, and
    a stray one would make the whole file unparseable.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return '"' + "".join(c for c in escaped if c >= " " or c == "\\") + '"'


def render(turns: list[Turn]) -> str:
    """The set, in the shape `run.py` loads."""
    lines = [
        "# Собрано из живых вопросов: evals/harvest.py",
        "#",
        "# `finds` — документ, на который ответ СОСЛАЛСЯ, а не эталон: это база",
        "# для сравнения («раньше находилось это»), а не проверка правильности.",
        "# `empty = true` — ответ не сослался ни на что, то есть в базе не нашлось.",
        "#",
        "# Это чьи-то настоящие вопросы. Файл не коммитится (см. .gitignore).",
        "",
    ]
    for turn in turns:
        lines.append("[[question]]")
        lines.append(f"ask = {quote(turn.ask)}")
        if turn.empty:
            lines.append("empty = true")
        else:
            lines.append(f"finds = {quote(turn.finds)}")
        if turn.rated:
            lines.append(f"# оценка человека: {turn.rated:+d}")
        lines.append("")
    return "\n".join(lines)


def summary(turns: list[Turn], kept: list[Turn]) -> None:
    print(f"вопросов в базе: {len(turns)}, в набор попало: {len(kept)}")
    rated = [turn for turn in kept if turn.rated]
    if rated:
        print(f"с оценкой человека: {len(rated)} (эти идут первыми)")

    # The gaps report: what the base could not answer. Free, and it is the
    # direct answer to "какой документ добавить следующим".
    gaps = [turn for turn in kept if turn.empty]
    if gaps:
        print(f"\nне нашлось в базе — {len(gaps)}:")
        print("  (у ответов, записанных до появления трассы, среди них могут")
        print("   оказаться вопросы, которые к документам и не обращались)")
        for turn in gaps[:20]:
            print(f"  · {turn.ask[:76]}")
        if len(gaps) > 20:
            print(f"  … и ещё {len(gaps) - 20}")

    cited = Counter(turn.finds for turn in kept if turn.finds)
    if cited:
        print("\nчаще всего отвечали по:")
        for name, count in cited.most_common(5):
            print(f"  {count:3d}  {name[:60]}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать набор оценки из живых вопросов")
    parser.add_argument("--user", type=UUID, default=None, help="чьи вопросы (по умолчанию все)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    turns = await collected(args.user)
    if not turns:
        print("вопросов с ответами не нашлось")
        return 1
    kept = chosen(turns)

    body = render(kept)
    # Parsed back before it is written: a file the harness cannot read is worse
    # than no file, and it would fail much later and less clearly.
    tomllib.loads(body)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, "utf-8")
    summary(turns, kept)
    print(f"\nзаписано: {args.out}")
    print(f"прогон:   python evals/run.py --set {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
