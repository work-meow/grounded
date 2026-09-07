"""The same token, two ways in — and what happens when both arrive.

A program integrating this service sends ``Authorization: Bearer``; the browser
sends an HttpOnly cookie. The case worth a test is both at once, on a developer
machine where somebody is logged in and a script is running: answering as the
cookie's owner would be a wrong-tenant answer that looks entirely successful.
"""

from datetime import timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.security import SESSION_COOKIE, issue_token


def token(user_id) -> str:
    return issue_token(get_settings(), user_id, timedelta(minutes=5))


def test_a_bearer_token_is_a_way_in():
    user_id = uuid4()
    with TestClient(app, base_url="https://testserver") as client:
        response = client.get("/api/v1/me", headers={"Authorization": f"Bearer {token(user_id)}"})

    assert response.status_code == 200
    assert response.json() == {"user_id": str(user_id)}


def test_the_header_wins_over_a_cookie_from_somebody_else():
    caller, logged_in = uuid4(), uuid4()
    with TestClient(app, base_url="https://testserver") as client:
        client.cookies.set(SESSION_COOKIE, token(logged_in))
        response = client.get("/api/v1/me", headers={"Authorization": f"Bearer {token(caller)}"})

    assert response.json() == {"user_id": str(caller)}


def test_a_token_this_server_did_not_sign_is_refused_with_the_challenge():
    with TestClient(app, base_url="https://testserver") as client:
        response = client.get("/api/v1/me", headers={"Authorization": "Bearer not.a.token"})

    assert response.status_code == 401
    # How a client library learns where the token goes. Bearer and not Basic:
    # a Basic challenge makes a browser open a login dialog of its own.
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_a_scheme_that_is_not_bearer_falls_through_to_the_cookie():
    """Basic auth is not something this service offers, so it is not a token.

    It must not shadow the cookie either — a proxy that adds its own
    Authorization header would otherwise log everybody out.
    """
    user_id = uuid4()
    with TestClient(app, base_url="https://testserver") as client:
        client.cookies.set(SESSION_COOKIE, token(user_id))
        response = client.get("/api/v1/me", headers={"Authorization": "Basic dXNlcjpwYXNz"})

    assert response.json() == {"user_id": str(user_id)}


def test_nothing_at_all_is_a_401():
    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/api/v1/me").status_code == 401
