"""A ceiling on what one token may spend in a day.

One turn has been bounded from the start — so many searches, so many model
calls, a wall clock — but the *stream* of turns has not. A token is a personal
credential that never expires early, so a leaked one can spend real money in a
loop, and the first anybody would know is the bill.

Two choices worth explaining.

**Dollars, not requests.** A request is a poor unit here: a question answered
from two fragments and one answered after three searches and a web lookup
differ by twenty times in cost. What is being protected is money, so the
number counted is money — the same provider-reported figure the trace and the
API already report.

**Not in the authentication dependency.** Checking the signature deliberately
never touches the database; that is why a token cannot be revoked early, and
it is what keeps a document listing cheap. So the ceiling is enforced where
money is actually spent — the endpoints that answer questions — and a request
that costs nothing is not charged for one.

The consequence is that a turn is checked before it runs and recorded after,
so the last turn of a day may overshoot the limit by its own cost. Refusing
mid-answer would be worse, and reserving an estimate up front would mean
inventing a number for something whose price is only known afterwards.
"""

import logging
import uuid
from datetime import UTC, date, datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Spending

logger = logging.getLogger(__name__)


#: How long to tell a client to wait. The window is a calendar day in UTC, so
#: the honest answer is "until midnight" — computed, not guessed, because a
#: client that retries at the wrong second gets refused again.
def _until_midnight(now: datetime) -> int:
    tomorrow = datetime.combine(now.date(), datetime.min.time(), tzinfo=UTC)
    return max(1, int((tomorrow.timestamp() + 86_400) - now.timestamp()))


async def spent_today(session: AsyncSession, user_id: uuid.UUID) -> float:
    """What this user has been charged for today, in dollars."""
    found = await session.execute(
        select(Spending.cost_usd).where(
            Spending.user_id == user_id, Spending.day == datetime.now(UTC).date()
        )
    )
    return float(found.scalar_one_or_none() or 0.0)


async def refuse_if_spent(session: AsyncSession, settings: Settings, user_id: uuid.UUID) -> None:
    """429 when today's budget is gone. Does nothing when there is no budget."""
    if settings.daily_cost_limit_usd <= 0:
        return
    spent = await spent_today(session, user_id)
    if spent < settings.daily_cost_limit_usd:
        return
    logger.warning("user %s has spent $%.4f today; refusing", user_id, spent)
    raise HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        f"Дневной лимит ${settings.daily_cost_limit_usd:.2f} израсходован",
        headers={"Retry-After": str(_until_midnight(datetime.now(UTC)))},
    )


async def record(session: AsyncSession, user_id: uuid.UUID, cost_usd: float, when: date) -> None:
    """Add a turn's cost to the day's total.

    An upsert rather than select-then-update: two turns finishing at once would
    otherwise each read the same total and write it back, and the cheaper of
    them would vanish. Postgres does the addition.
    """
    await session.execute(
        insert(Spending)
        .values(user_id=user_id, day=when, turns=1, cost_usd=cost_usd)
        .on_conflict_do_update(
            index_elements=[Spending.user_id, Spending.day],
            set_={
                "turns": Spending.turns + 1,
                "cost_usd": Spending.cost_usd + cost_usd,
            },
        )
    )
    await session.commit()
