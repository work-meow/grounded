"""Mint a login token: ``uv run rag-token [--days 30] [--email you@example.com]``

Creates the user row if it does not exist and prints a JWT to paste into the
login screen. This is the whole user-management story for a personal install.
"""

import argparse
import asyncio
from datetime import timedelta
from uuid import UUID

from sqlalchemy import select

from app.config import get_settings
from app.db import Session, engine
from app.models import User
from app.security import issue_token


async def _run(email: str | None, user_id: UUID | None, days: int) -> str:
    try:
        return await _issue(email, user_id, days)
    finally:
        # Inside this loop, not after it: asyncpg's connections belong to the
        # loop that opened them, and disposing from a second asyncio.run()
        # fails with "Event loop is closed".
        await engine.dispose()


async def _issue(email: str | None, user_id: UUID | None, days: int) -> str:
    settings = get_settings()
    async with Session() as session:
        user = None
        if user_id is not None:
            user = await session.get(User, user_id)
            if user is None:
                raise SystemExit(f"no user with id {user_id}")
        elif email is not None:
            user = (
                await session.execute(select(User).where(User.email == email))
            ).scalar_one_or_none()

        if user is None:
            user = User(email=email)
            session.add(user)
            await session.commit()

        token = issue_token(settings, user.id, timedelta(days=days))
        print(f"user_id: {user.id}")
        print(f"email:   {user.email or '-'}")
        print(f"expires: {days} d\n")
        print(token)
        return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue a login JWT.")
    parser.add_argument("--email", help="find or create the user by email")
    parser.add_argument("--user-id", type=UUID, help="issue for an existing user id")
    parser.add_argument("--days", type=int, default=30, help="token lifetime (default: 30)")
    args = parser.parse_args()

    asyncio.run(_run(args.email, args.user_id, args.days))


if __name__ == "__main__":
    main()
