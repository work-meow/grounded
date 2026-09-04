from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from app.config import Settings

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
        algorithm=settings.jwt_algorithm,
    )


def read_token(settings: Settings, token: str) -> UUID:
    """Verify signature, issuer and expiry, and return the subject.

    Raises ``jwt.InvalidTokenError`` (or ``ValueError`` for a malformed
    subject) — callers turn that into a 401.
    """
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        issuer=settings.jwt_issuer,
        leeway=_LEEWAY_S,
        options={"require": ["exp", "iat", "sub", "iss"]},
    )
    return UUID(payload["sub"])
