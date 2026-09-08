"""A run's numbers, and comparing two of them.

Every improvement in this system so far was measured by hand and then
remembered, so the next one started from nothing. What is worth testing about
keeping them is the direction: fewer unsupported claims is better, fewer
documents found is worse, and a comparison that gets that backwards would
send somebody the wrong way with a straight face.

`evals/` is not a package — the module is loaded by path.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from app.config import Settings

# The script is run as `python evals/run.py`, which puts its own directory on
# the path — that is how its `import grounded` resolves. Loading it by path
# here has to say the same thing, or the import fails at collection.
_EVALS = Path(__file__).parent.parent / "evals"
sys.path.insert(0, str(_EVALS))

_spec = importlib.util.spec_from_file_location("evalrun", _EVALS / "run.py")
assert _spec and _spec.loader
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

SET = Path("questions.toml")


def settings() -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k")


def outcome(**overrides) -> "run.Outcome":
    question = run.Question(
        ask=overrides.pop("ask", "вопрос"),
        finds=overrides.pop("finds", "Политика.pdf"),
        empty=overrides.pop("empty", False),
    )
    return run.Outcome(question=question, **overrides)


def test_the_numbers_a_run_is_judged_by():
    measured = run.numbers(
        [
            outcome(
                found=True,
                rank=1,
                chunks=4,
                on_target=3,
                candidates=20,
                documents=2,
                correct=True,
                unsupported=(),
                seconds=3.0,
            ),
            outcome(
                found=True,
                rank=2,
                chunks=6,
                on_target=3,
                candidates=20,
                documents=3,
                correct=True,
                unsupported=("премия 15 000",),
                seconds=5.5,
            ),
            outcome(
                ask="чего нет",
                finds="",
                empty=True,
                chunks=0,
                candidates=20,
                correct=True,
                seconds=1.0,
            ),
        ],
        settings(),
        k=20,
        question_set=SET,
    )

    assert (measured["found"], measured["of_found"]) == (2, 2)
    assert (measured["on_target"], measured["fragments"]) == (6, 10)
    assert measured["candidates"] == 60
    # The unanswerable question is excluded from precision but counted where it
    # belongs: how much came back for a question with no answer.
    assert measured["blank_fragments_avg"] == 0.0
    assert (measured["grounded"], measured["of_grounded"]) == (1, 2)
    assert measured["unsupported"] == 1
    assert measured["seconds_max"] == 5.5


def test_a_question_nobody_checked_is_not_counted_as_grounded():
    """None means no verdict. Counting it as clean is exactly how a metric
    starts lying about the thing it exists for."""
    measured = run.numbers(
        [outcome(found=True, correct=True, unsupported=None)], settings(), k=8, question_set=SET
    )

    assert measured["of_grounded"] == 0
    assert measured["grounded"] == 0


def test_the_settings_travel_with_the_numbers():
    """«по делу 82%» means nothing without knowing what k was and whether the
    judge was on."""
    measured = run.numbers([outcome(found=True)], settings(), k=20, question_set=SET)

    assert measured["settings"]["k"] == 20
    assert "rerank_enabled" in measured["settings"]
    assert measured["set"] == "questions.toml"


@pytest.mark.parametrize(
    ("key", "was", "now", "verdict"),
    [
        ("found", 8, 10, "лучше"),
        ("found", 10, 8, "ХУЖЕ"),
        ("on_target", 19, 65, "лучше"),
        ("grounded", 5, 4, "ХУЖЕ"),
        ("unsupported", 3, 0, "лучше"),
        ("unsupported", 0, 3, "ХУЖЕ"),
        ("documents_avg", 4.9, 1.3, "лучше"),
        ("blank_fragments_avg", 0.0, 8.0, "ХУЖЕ"),
        ("seconds_max", 16.4, 4.4, "лучше"),
    ],
)
def test_the_direction_is_named_and_not_left_to_the_reader(capsys, key, was, now, verdict):
    run.compare({key: was, "when": "раньше"}, {key: now, "when": "теперь"})

    printed = capsys.readouterr().out
    assert verdict in printed, printed


def test_a_number_that_did_not_move_is_not_called_progress(capsys):
    run.compare({"found": 10}, {"found": 10})

    printed = capsys.readouterr().out
    assert "лучше" not in printed and "ХУЖЕ" not in printed


def test_comparing_two_different_sets_says_so(capsys):
    """The harvested set and the shipped one have different questions; their
    numbers put side by side would look like a change and be an accident."""
    run.compare({"set": "questions.toml"}, {"set": "harvested.toml"})

    assert "несравнимы" in capsys.readouterr().out
