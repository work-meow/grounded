"""What the agent is told, and what it says while it works.

Two small things that both fail quietly. A model that does not know the date
answers "на этой неделе" from whatever year its training stopped, and the
answer looks like an ordinary wrong answer rather than a missing fact. And a
turn that spends four seconds in a tool with nothing on screen is
indistinguishable from a turn that has hung.
"""

import uuid
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app import agent
from app.config import Settings

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


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


# --- saying what is happening ------------------------------------------------


class _Chunk:
    """An AIMessageChunk, as far as answer() is concerned."""

    def __init__(self, content="", tool_call_chunks=None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []


def _streaming(monkeypatch, chunks):
    class _Agent:
        async def astream(self, _inputs, **_kwargs):
            for chunk in chunks:
                yield chunk, {"langgraph_node": "model"}

    monkeypatch.setattr(agent, "create_agent", lambda **_kwargs: _Agent())


async def _events(monkeypatch, chunks):
    _streaming(monkeypatch, chunks)
    return [event async for event in agent.answer(settings(), USER, "вопрос", [])]


async def test_a_tool_call_is_announced_before_anything_is_written(monkeypatch):
    """The name arrives in the first chunk of the call, before its arguments
    have finished streaming and before the tool has run — which is the whole
    point: it is the earliest moment there is anything to show."""
    events = await _events(
        monkeypatch,
        [
            _Chunk(tool_call_chunks=[{"name": "search_knowledge", "args": "", "index": 0}]),
            _Chunk(tool_call_chunks=[{"name": None, "args": '{"query"', "index": 0}]),
            _Chunk(content="сорок"),
        ],
    )

    assert events[0] == ("step", "search_knowledge")
    assert events[1] == ("token", "сорок")


async def test_one_call_is_one_announcement(monkeypatch):
    """Only the first chunk of a call carries a name; the rest are argument
    fragments. Announcing those too would flicker the label on every token."""
    events = await _events(
        monkeypatch,
        [
            _Chunk(tool_call_chunks=[{"name": "search_knowledge", "index": 0}]),
            _Chunk(tool_call_chunks=[{"name": None, "args": "que", "index": 0}]),
            _Chunk(tool_call_chunks=[{"name": None, "args": "ry", "index": 0}]),
        ],
    )

    assert [event for event in events if event[0] == "step"] == [("step", "search_knowledge")]


async def test_two_calls_are_two_announcements(monkeypatch):
    events = await _events(
        monkeypatch,
        [
            _Chunk(tool_call_chunks=[{"name": "search_knowledge", "index": 0}]),
            _Chunk(tool_call_chunks=[{"name": "read_document", "index": 1}]),
        ],
    )

    assert [payload for kind, payload in events if kind == "step"] == [
        "search_knowledge",
        "read_document",
    ]


async def test_a_turn_with_no_tools_announces_nothing(monkeypatch):
    events = await _events(monkeypatch, [_Chunk(content="сорок")])

    assert [event for event in events if event[0] == "step"] == []
    assert events[0] == ("token", "сорок")
