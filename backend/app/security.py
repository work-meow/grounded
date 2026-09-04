from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from app.config import Settings

# One name, defined once. It used to be a setting that only auth.py honoured —
# deps.py read a hard-coded alias, because FastAPI's Cookie(alias=...) has to be
# a constant. Changing the setting therefore issued a cookie nothing read, and
# locked every user out until it was changed back. A knob whose only reachable
# setting is its default is not a knob.
SESSION_COOKIE = "rag_session"

# Likewise a constant, and for a stronger reason than the cookie name. As a
# setting it was a way to write "none" into .env and turn signature
# verification off — an auth bypass reachable by editing a config file. There is
# also nothing to choose between: the key is a shared secret, so the algorithm
# has to be HMAC, and SHA-256 is the one everything implements.
JWT_ALGORITHM = "HS256"

_LEEWAY_S = 30


def issue_token(settings: Settings, user_id: UUID, ttl: timedelta) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": str(user_id),
            "iss": settings.jwt_issuer,
            "iat": now,
            "exp": now + ttl,
        },
        settings.jwt_secret,
        algorithm=JWT_ALGORITHM,
    )


def read_token(settings: Settings, token: str) -> UUID:
    """Verify signature, issuer and expiry, and return the subject.

    Raises ``jwt.InvalidTokenError`` (or ``ValueError`` for a malformed
    subject) — callers turn that into a 401.
    """
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[JWT_ALGORITHM],
        issuer=settings.jwt_issuer,
        leeway=_LEEWAY_S,
        options={"require": ["exp", "iat", "sub", "iss"]},
    )
    return UUID(payload["sub"])
