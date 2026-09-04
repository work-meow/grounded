from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    # The PostgreSQL instance is shared with other services on the host, so the
    # pool stays small: a single-worker API answering one person does not need
    # twenty connections, and holding them idle is pure cost to everyone else.
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    # Queries are cheap; an LLM turn is not. A connection must never be held
    # across one — see routers/chats.py.
    pool_pre_ping=True,
    pool_recycle=1800,
)

Session = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with Session() as session:
        yield session
