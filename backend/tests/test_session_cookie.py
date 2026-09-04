"""Login and authentication must agree on the cookie's name.

They once did not: the name was a setting, but the dependency that reads it
needs a compile-time constant for FastAPI's Cookie(alias=...) and carried its
own copy. Setting it to anything else issued a cookie nothing would read.
"""

from datetime import timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.security import SESSION_COOKIE, issue_token


def test_the_cookie_login_sets_is_the_cookie_auth_reads():
    settings = get_settings()
    user_id = uuid4()

    # https, because the cookie is issued Secure by default and a client would
    # rightly refuse to send it back over plain http — which is itself worth
    # knowing when the login "works" locally but every later call is a 401.
    with TestClient(app, base_url="https://testserver") as client:
        token = issue_token(settings, user_id, timedelta(minutes=5))

        login = client.post("/api/auth/login", json={"token": token})
        assert login.status_code == 200
        assert SESSION_COOKIE in login.cookies

        # The client replays exactly what the server set — nothing is passed by
        # hand, so a name mismatch shows up as a 401.
        assert client.get("/api/auth/me").json() == {"user_id": str(user_id)}

        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").status_code == 401


def test_a_token_this_server_did_not_sign_is_refused():
    with TestClient(app, base_url="https://testserver") as client:
        assert client.post("/api/auth/login", json={"token": "not.a.token"}).status_code == 401
