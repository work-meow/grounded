"""What one request cost.

Every paid call this service makes goes through OpenRouter, and OpenRouter puts
the price of a completion in its ``usage`` object — measured, and worth
recording here because it is not what the documentation implies: ``cost`` comes
back on an ordinary request, with no ``usage: {"include": true}`` asked for, and
it survives LangChain's wrapper into ``response_metadata["cost"]``. So the bill
for a turn is not estimated from a price table that would go stale; it is the
sum of what the provider charged, call by call.

A turn spends in up to three places — the agent's own model calls, the
relevance judge behind every search, and a web search when it is on — and which
of them dominates is the useful part. A web search is six times the judge and,
on a short question, more than the answer itself. So the report keeps the
breakdown rather than one number.

One thing this deliberately does not do is guess. A call whose cost the
provider did not report contributes its tokens and nothing to the total, and
``cost_complete`` goes false — a floor that says so beats a total that quietly
understates the bill.
"""

from dataclasses import dataclass, field
from typing import Any

#: The three stages that can spend money in one request. Strings rather than an
#: enum because they are part of the public API's response shape.
ANSWER = "answer"
RERANK = "rerank"
WEB_SEARCH = "web_search"


@dataclass(frozen=True, slots=True)
class Call:
    """One paid call to a model."""

    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    #: US dollars, as the provider reported them. None when it did not say.
    cost_usd: float | None


@dataclass
class Spend:
    """What one request paid for, in the order it paid.

    Mutated from several coroutines — the agent streams while its tools run —
    which is safe here without a lock: ``list.append`` is one bytecode and
    asyncio does not preempt inside it. Nothing reads the list until the turn
    is over.
    """

    calls: list[Call] = field(default_factory=list)

    def add(
        self,
        stage: str,
        model: str,
        *,
        prompt_tokens: Any = 0,
        completion_tokens: Any = 0,
        cost_usd: Any = None,
    ) -> None:
        # Sanitised here rather than at the two call sites, because both read
        # numbers off a provider's JSON: a null cost, a string where a count
        # belongs, or a negative anything must not reach the report.
        self.calls.append(
            Call(
                stage=stage,
                model=model,
                prompt_tokens=_count(prompt_tokens),
                completion_tokens=_count(completion_tokens),
                cost_usd=_money(cost_usd),
            )
        )

    def add_completion(self, stage: str, requested_model: str, body: Any) -> None:
        """One call, read out of an OpenRouter chat-completion response.

        Never raises: a body in a shape this build does not know costs the
        accounting one call, and must not cost the caller its answer. The model
        is taken from the response rather than the request — OpenRouter is free
        to route ``:floor`` or a dated variant, and the bill belongs to the one
        that actually ran.
        """
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            return
        model = body.get("model") if isinstance(body, dict) else None
        self.add(
            stage,
            str(model or requested_model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"),
        )

    def report(self) -> dict[str, Any]:
        """The totals, and where they came from."""
        groups: dict[tuple[str, str], list[Call]] = {}
        for call in self.calls:
            groups.setdefault((call.stage, call.model), []).append(call)

        prompt = sum(call.prompt_tokens for call in self.calls)
        completion = sum(call.completion_tokens for call in self.calls)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cost_usd": _total(self.calls),
            # False when some call did not report a price, which makes the
            # total above a floor rather than the bill.
            "cost_complete": all(call.cost_usd is not None for call in self.calls),
            "calls": len(self.calls),
            "stages": [
                {
                    "stage": stage,
                    "model": model,
                    "calls": len(calls),
                    "prompt_tokens": sum(call.prompt_tokens for call in calls),
                    "completion_tokens": sum(call.completion_tokens for call in calls),
                    "cost_usd": _total(calls),
                }
                for (stage, model), calls in groups.items()
            ],
        }


def _total(calls: list[Call]) -> float:
    # Rounded, and to eight places rather than the four a currency would want:
    # the judge behind one search costs $0.00017, and two decimals fewer would
    # report every cheap request as free.
    return round(sum(call.cost_usd or 0.0 for call in calls), 8)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _money(value: Any) -> float | None:
    """A price, or None if the provider sent something that is not one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value >= 0 else None
