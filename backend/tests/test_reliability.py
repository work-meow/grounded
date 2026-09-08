"""What happens when a call inside the turn fails.

A blip at the provider used to cost the whole answer: the reader saw an error
event, the API returned 502, and the question had to be asked again by hand —
for the one class of failure a second attempt usually fixes.

The retry is exercised through the real agent graph rather than by inspecting
the middleware list, because "the middleware is present" and "the turn
survives" are different claims and only the second one matters.
"""

from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter

from app import agent
from app.config import Settings

USER = uuid4()


def settings(**overrides) -> Settings:
    return Settings(
        **{
            "jwt_secret": "x" * 40,
            "openrouter_api_key": "k",
            "agent_timeout_s": 10.0,
            **overrides,
        }
    )


class Flaky(GenericFakeChatModel):
    """A model that fails a given number of times and then answers."""

    failures: int = 1

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("upstream said 503")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def turn(monkeypatch, model, **overrides) -> str:
    """One real turn against the given model, with retrieval left out of it."""
    monkeypatch.setattr(agent, "_build_tools", lambda *_args, **_kwargs: [])
    text = ""
    async for kind, payload in agent.answer(
        settings(**overrides), USER, "сколько дней отпуска?", []
    ):
        if kind == "token":
            text += payload
    return text


async def test_a_provider_that_blipped_does_not_cost_the_answer(monkeypatch):
    flaky = Flaky(messages=iter([AIMessage("28 календарных дней")]), failures=1)
    monkeypatch.setattr(agent, "_model", lambda *_args, **_kwargs: flaky)

    assert "28 календарных дней" in await turn(monkeypatch, flaky)


async def test_a_provider_that_stays_down_still_fails_the_turn(monkeypatch):
    """Retries are bounded. A turn that cannot be answered has to end, not
    keep trying — the caller is holding a stream open."""
    flaky = Flaky(messages=iter([AIMessage("не дойдёт")]), failures=99)
    monkeypatch.setattr(agent, "_model", lambda *_args, **_kwargs: flaky)

    with pytest.raises(Exception, match="503"):
        await turn(monkeypatch, flaky)


async def test_retries_can_be_turned_off(monkeypatch):
    flaky = Flaky(messages=iter([AIMessage("28 дней")]), failures=1)
    monkeypatch.setattr(agent, "_model", lambda *_args, **_kwargs: flaky)

    with pytest.raises(Exception, match="503"):
        await turn(monkeypatch, flaky, agent_retries=0, agent_fallback_model="")


# --- the shape of the middleware list ----------------------------------------


def test_the_fallback_is_a_built_client_and_not_a_name():
    """A string goes through LangChain's `init_chat_model`, which knows nothing
    about OpenRouter and would ask for another provider's key. Checked because
    the failure would only show up when the primary model was already down —
    the one moment nobody wants a second bug."""
    [fallback] = [
        limit for limit in agent._limits(settings(), web=False) if hasattr(limit, "models")
    ]

    assert fallback.models
    assert all(isinstance(model, ChatOpenRouter) for model in fallback.models)


def test_building_the_turn_does_not_open_a_new_connection_pool_each_time():
    """The client cache was sized for one model. When the fallback and the
    summariser arrived it silently began evicting on every call — three fresh
    pools per question, which is exactly the leak the cache exists to stop.
    Measured then: hits=0, misses=4."""
    agent._model.cache_clear()
    settings_used = settings()

    agent._limits(settings_used, web=False)
    agent._limits(settings_used, web=False)

    assert agent._model.cache_info().hits > 0


def test_reading_a_document_is_capped_like_searching_is():
    """One request to the index plus one call to the judge, same as a search —
    but it was bounded only by the overall six, which made "read it again,
    differently" looser than the loop that was actually designed for."""
    capped = {
        limit.tool_name: limit.run_limit
        for limit in agent._limits(settings(), web=False)
        if getattr(limit, "tool_name", None)
    }

    assert capped["read_document"] == settings().max_document_reads_per_run
