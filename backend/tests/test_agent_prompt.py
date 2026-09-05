"""What the agent is told before it answers.

A model that does not know the date answers "что на этой неделе" from whatever
year its training stopped, and the answer looks like an ordinary wrong answer
rather than a missing fact — which is why this is worth a test rather than a
line of trust.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app import agent
from app.config import Settings


def settings(**overrides) -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k", **overrides)


# --- the date ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (datetime(2026, 9, 5, 12, 0, tzinfo=UTC), "Сегодня суббота, 5 сентября 2026 года."),
        (datetime(2026, 1, 1, 0, 1, tzinfo=UTC), "Сегодня четверг, 1 января 2026 года."),
        (datetime(2026, 12, 31, 23, 59, tzinfo=UTC), "Сегодня четверг, 31 декабря 2026 года."),
    ],
)
def test_the_date_is_written_the_way_a_person_writes_it(when, expected):
    """Genitive month, no leading zero — «5 сентября», not «сентябрь» or «05»."""
    assert agent._today(when) == expected


def test_the_prompt_carries_today_in_the_configured_zone():
    zone = "Pacific/Kiritimati"  # UTC+14, so it is a different day from UTC half the time

    prompt = agent.system_prompt(settings(timezone=zone))

    assert str(datetime.now(ZoneInfo(zone)).day) in prompt
    assert "Считай «сегодня»" in prompt


def test_the_rules_are_still_there():
    """The date is added to the prompt, not instead of it."""
    assert agent.SYSTEM_PROMPT in agent.system_prompt(settings())


def test_the_prompt_holds_still_for_a_whole_day():
    """No clock time in it, deliberately: an identical system prompt across a
    day's requests is what keeps it eligible for upstream prompt caching."""
    assert agent.system_prompt(settings()) == agent.system_prompt(settings())


def test_a_timezone_that_does_not_exist_is_refused_at_startup():
    """Rather than raising inside the agent, mid-prompt, on the first question
    of the day — where it would read as the model being broken."""
    with pytest.raises(ValidationError):
        settings(timezone="Europe/Moskva")
