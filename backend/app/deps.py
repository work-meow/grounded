from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.security import read_token


# Both dependencies are async even though neither awaits anything: FastAPI
# hands a synchronous dependency to the threadpool, and these run on every
# authenticated request for work measured in microseconds. A threadpool slot is
# a shared, limited resource — it should be spent on real blocking I/O.
async def _settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]

_UNAUTHORIZED = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Требуется вход")


async def current_user(
    settings: SettingsDep,
    rag_session: Annotated[str | None, Cookie(alias="rag_session")] = None,
) -> UUID:
    if not rag_session:
        raise _UNAUTHORIZED
    try:
        return read_token(settings, rag_session)
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise _UNAUTHORIZED from exc


UserDep = Annotated[UUID, Depends(current_user)]
