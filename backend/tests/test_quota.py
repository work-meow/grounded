"""A ceiling on what one token can spend in a day.

One turn was bounded from the start — searches, model calls, a wall clock —
but the stream of turns was not, and a token never expires early. A leaked one
could spend real money in a loop with the bill as the first notification.

Two properties are worth pinning. The unit is dollars, because two questions
differ twenty times in cost and money is what is being protected. And the
check is not in the authentication dependency: verifying a signature
deliberately never touches the database, which is what keeps a document
listing cheap — so a request that spends nothing is not charged for one.
"""

from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app import quota
from app.config import Settings


def settings(**overrides) -> Settings:
    return Settings(**{"jwt_secret": "x" * 40, "openrouter_api_key": "k", **overrides})


class Recorded:
    """A session standing in for PostgreSQL, remembering what it was told."""

    def __init__(self, spent: float | None = None):
        self.spent = spent
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)

        class Result:
            def scalar_one_or_none(_self):
                return self.spent

        return Result()

    async def commit(self):
        pass


async def test_nothing_is_refused_when_there_is_no_ceiling():
    """Off by default: this is a personal install whose owner is the only
    caller, and a limit nobody asked for is a limit that surprises them."""
    session = Recorded(spent=999.0)

    await quota.refuse_if_spent(session, settings(daily_cost_limit_usd=0), uuid4())

    assert session.statements == [], "и в базу за этим не ходили"


async def test_a_turn_under_the_ceiling_goes_ahead():
    await quota.refuse_if_spent(
        Recorded(spent=0.4), settings(daily_cost_limit_usd=1.0), uuid4()
    )


async def test_a_spent_day_is_refused_with_how_long_to_wait():
    with pytest.raises(HTTPException) as refused:
        await quota.refuse_if_spent(
            Recorded(spent=1.0), settings(daily_cost_limit_usd=1.0), uuid4()
        )

    assert refused.value.status_code == 429
    # A number, not a guess: a client that retries at the wrong second is
    # refused again for no reason it can see.
    assert int(refused.value.headers["Retry-After"]) > 0
    assert "1.00" in refused.value.detail


async def test_a_day_with_nothing_on_it_is_zero_and_not_an_error():
    """The common case — the first question of the day has no row yet."""
    assert await quota.spent_today(Recorded(spent=None), uuid4()) == 0.0


@pytest.mark.parametrize(
    ("now", "least", "most"),
    [
        (datetime(2026, 9, 8, 0, 0, tzinfo=UTC), 86_000, 86_400),
        (datetime(2026, 9, 8, 23, 59, tzinfo=UTC), 1, 120),
        (datetime(2026, 9, 8, 12, 0, tzinfo=UTC), 43_000, 43_400),
    ],
)
def test_the_wait_is_until_midnight(now, least, most):
    """The window is a UTC day, so the honest answer is how long is left of
    it — not a fixed hour that would refuse the client twice."""
    wait = quota._until_midnight(now)

    assert least <= wait <= most


def test_recording_adds_rather_than_overwrites():
    """Two turns finishing at once would each read the same total and write it
    back, and the cheaper of them would vanish. Postgres does the addition, so
    what goes out has to be an upsert with an increment in it."""
    session = Recorded()

    import asyncio

    asyncio.run(quota.record(session, uuid4(), 0.0013, date(2026, 9, 8)))

    [statement] = session.statements
    rendered = str(statement).lower()
    assert "on conflict" in rendered
    assert "spending.cost_usd +" in rendered or "cost_usd + " in rendered


def test_yesterdays_spending_is_not_todays():
    """The window has to move, or a busy Tuesday would lock out every
    Wednesday after it."""
    session = Recorded(spent=5.0)

    import asyncio

    asyncio.run(quota.spent_today(session, uuid4()))

    [statement] = session.statements
    today = datetime.now(UTC).date()
    assert str(today) in str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert str(today - timedelta(days=1)) not in str(
        statement.compile(compile_kwargs={"literal_binds": True})
    )
