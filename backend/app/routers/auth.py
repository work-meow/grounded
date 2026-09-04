import jwt
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel

from app.deps import SettingsDep, UserDep
from app.security import read_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Tokens are long-lived and minted offline; the cookie just mirrors that
# lifetime and is capped at 30 days so a stolen browser session ages out.
_COOKIE_MAX_AGE = 30 * 24 * 3600


class LoginRequest(BaseModel):
    token: str


@router.post("/login")
async def login(body: LoginRequest, response: Response, settings: SettingsDep) -> dict[str, str]:
    """Exchange a pasted JWT for an HttpOnly cookie.

    The token never reaches localStorage — script-readable storage is exactly
    what we are avoiding here.
    """
    try:
        user_id = read_token(settings, body.token.strip())
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный или истёкший токен"
        ) from exc

    response.set_cookie(
        settings.session_cookie,
        body.token.strip(),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
    return {"user_id": str(user_id)}


@router.post("/logout")
async def logout(response: Response, settings: SettingsDep) -> dict[str, bool]:
    response.delete_cookie(settings.session_cookie, path="/")
    return {"ok": True}


@router.get("/me")
async def me(user_id: UserDep) -> dict[str, str]:
    return {"user_id": str(user_id)}
