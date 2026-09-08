"""Turning real traffic into a set the harness can run.

Two things here can go wrong quietly. A question paired with the wrong answer
would put a document in `finds` that has nothing to do with it, and the set
would measure noise while looking healthy — so the pairing is checked across
chat boundaries and around a deleted question. And a file that does not parse
would fail much later and much less clearly than at the moment it is written,
so what the writer produces is read back with `tomllib`.

`evals/` is not a package (it is run as a script inside the container), so the
module is loaded by path.
"""

import importlib.util
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

_spec = importlib.util.spec_from_file_location(
    "harvest", Path(__file__).parent.parent / "evals" / "harvest.py"
)
assert _spec and _spec.loader
harvest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harvest)

WHEN = datetime(2026, 9, 1, tzinfo=UTC)

SEARCHED = {"steps": [{"tool": "search_knowledge", "query": "отпуск"}]}


def row(chat, role, content, citations=(), *, rating=None, trace=None, when=WHEN):
    """One row in the shape the query returns."""
    return (chat, role, content, list(citations), rating, when, trace)


def turn(ask, finds="Политика.pdf", *, empty=False, rated=None, minutes=0):
    return harvest.Turn(
        ask=ask,
        finds="" if empty else finds,
        empty=empty,
        rated=rated,
        when=WHEN + timedelta(minutes=minutes),
    )


def test_a_question_is_paired_with_the_answer_that_followed_it():
    chat = uuid4()
    rows = [
        row(chat, "user", "сколько дней отпуска?"),
        row(chat, "assistant", "28 дней", [{"filename": "Политика.pdf"}], trace=SEARCHED),
    ]

    [found] = harvest.pair(rows)

    assert found.ask == "сколько дней отпуска?"
    assert found.finds == "Политика.pdf"
    assert found.empty is False


def test_a_question_is_never_paired_across_a_chat_boundary():
    """The pairing walks one conversation at a time. Two chats' rows arriving
    together would otherwise answer one person's question with another's."""
    first, second = uuid4(), uuid4()
    rows = [
        row(first, "user", "вопрос без ответа в первом чате"),
        row(second, "assistant", "ответ из другого чата", [{"filename": "Чужое.md"}]),
    ]

    assert harvest.pair(rows) == []


def test_an_answer_whose_question_was_deleted_is_skipped():
    chat = uuid4()
    rows = [row(chat, "assistant", "ответ", [{"filename": "Х.md"}])]

    assert harvest.pair(rows) == []


def test_an_answer_that_cited_nothing_becomes_a_negative_case():
    """That is what "в базе этого нет" looks like in stored data, and it is
    exactly the case worth keeping: an answer that starts citing something
    later is a change, and a silent one today."""
    chat = uuid4()
    rows = [
        row(chat, "user", "что там про налоговый вычет?"),
        row(chat, "assistant", "в базе этого нет", trace=SEARCHED),
    ]

    [found] = harvest.pair(rows)

    assert found.empty is True and found.finds == ""


def test_a_question_that_never_reached_the_documents_is_not_a_gap():
    """«Какое сегодня число» cites nothing and never could. Counting it as a
    gap would send somebody looking for a document that does not exist."""
    chat = uuid4()
    rows = [
        row(chat, "user", "Какое сегодня число и день недели?"),
        row(chat, "assistant", "7 сентября 2026", trace={"steps": []}),
    ]

    assert harvest.pair(rows) == []


def test_before_the_trace_existed_the_turn_is_kept():
    """There is no way to tell those two apart for an old answer, so it stays
    in the set and the summary says the list may contain both."""
    chat = uuid4()
    rows = [
        row(chat, "user", "что там про налоговый вычет?"),
        row(chat, "assistant", "в базе этого нет", trace=None),
    ]

    [found] = harvest.pair(rows)
    assert found.empty is True


def test_greetings_are_not_questions():
    chat = uuid4()
    rows = [
        row(chat, "user", "ку"),
        row(chat, "assistant", "здравствуйте"),
    ]

    assert harvest.pair(rows) == []


def test_the_same_question_asked_five_times_is_one_entry():
    kept = harvest.chosen([turn("что у меня в вишлисте?", minutes=n) for n in range(5)])

    assert len(kept) == 1


def test_the_version_somebody_judged_wins_over_the_newer_one():
    """A thumb is the only signal a person left on purpose; the question they
    marked wrong is the one worth re-running."""
    kept = harvest.chosen(
        [
            turn("сколько суточные?", rated=-1, minutes=0),
            turn("Сколько  суточные?  ", minutes=99),
        ]
    )

    assert len(kept) == 1
    assert kept[0].rated == -1


def test_rated_questions_come_first():
    kept = harvest.chosen([turn("а", minutes=50), turn("б", rated=-1), turn("в", minutes=90)])

    assert kept[0].ask == "б"


def test_what_is_written_can_be_read_back():
    body = harvest.render(
        [
            turn('вопрос с "кавычками" и \\ обратным слэшем'),
            turn("вопрос\nв две строки"),
            turn("чего в базе нет", empty=True),
            turn("оценённый", rated=1),
        ]
    )

    parsed = tomllib.loads(body)["question"]

    assert [entry["ask"] for entry in parsed][:2] == [
        'вопрос с "кавычками" и \\ обратным слэшем',
        "вопрос\nв две строки",
    ]
    assert parsed[2] == {"ask": "чего в базе нет", "empty": True}
    # No `answers`: guessing the right words out of an answer that may itself
    # be wrong would bake the mistake into the test.
    assert not any("answers" in entry for entry in parsed)


def test_a_control_character_cannot_break_the_file():
    """It is not in anything anybody typed, and one stray byte would make the
    whole set unparseable rather than one question odd."""
    body = harvest.render([turn("вопрос\x07со звонком")])

    assert tomllib.loads(body)["question"][0]["ask"] == "вопроссо звонком"
