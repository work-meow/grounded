from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.security import SESSION_COOKIE, read_token


# Both dependencies are async even though neither awaits anything: FastAPI
# hands a synchronous dependency to the threadpool, and these run on every
# authenticated request for work measured in microseconds. A threadpool slot is
# a shared, limited resource — it should be spent on real blocking I/O.
async def _settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]

#: The same token, presented the other way. The browser sends it as an HttpOnly
#: cookie so that no script can read it; a program integrating this service has
#: no cookie jar and no reason to want one, and sends the header instead.
#:
#: `auto_error=False` because this is one of two ways in, not the only one: the
#: scheme's own 403 would fire on every request the browser makes. Declaring it
#: at all is what puts `bearerAuth` in the OpenAPI document, and with it the
#: Authorize button in the interactive docs.
_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="bearerAuth",
    description="JWT, выпущенный `uv run rag-token`.",
)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Требуется вход",
    # What the standard asks for on a 401, and useful rather than ceremonial:
    # it is how a client library knows the token goes in this header. Bearer,
    # not Basic — a Basic challenge makes browsers pop up a login dialog.
    headers={"WWW-Authenticate": "Bearer"},
)


async def current_user(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> UUID:
    """Whose request this is, from the header or the cookie.

    The header wins when both are present. A caller that took the trouble to
    set Authorization means that token — most likely a service acting for one
    user from a machine where somebody happens to have logged in too, and
    silently answering as the cookie's owner would be the worst possible way to
    resolve that.
    """
    token = credentials.credentials if credentials else session_cookie
    if not token:
        raise _UNAUTHORIZED
    try:
        return read_token(settings, token)
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise _UNAUTHORIZED from exc


UserDep = Annotated[UUID, Depends(current_user)]
