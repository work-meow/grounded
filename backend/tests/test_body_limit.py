"""The ceiling on a request body — the one thing no framework layer provides.

Measured before this existed: 40 MB of JSON with no token took resident memory
from 290 MB to 520 MB, because FastAPI reads and parses a body before it solves
dependencies. Authentication happened after the memory was spent, and
`max_length=8000` was checked against a string already built. On a 4 GB box
that is an out-of-memory kill available to anyone who can reach the port.
"""

from datetime import timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import get_settings
from app.limits import JSON_LIMIT
from app.main import app
from app.security import issue_token


def test_an_oversized_body_is_refused_before_authentication():
    """413 and not 401, deliberately: this is decided before anything about the
    request is parsed, which is the entire point of deciding it here."""
    payload = '{"question": "' + "а" * (JSON_LIMIT + 1000) + '"}'

    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/api/v1/answer", content=payload, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 413
    assert "МБ" in response.json()["detail"]


def test_a_body_that_hides_its_length_is_cut_off():
    """Chunked encoding declares no length, so the guard has to count. The
    request ends as a disconnect rather than a 413: by then the application is
    running, and a client that went away is what every layer above understands."""

    def chunks():
        for _ in range(5):
            yield b"x" * (JSON_LIMIT // 2)

    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/api/v1/answer", content=chunks(), headers={"Content-Type": "application/json"}
        )

    assert response.status_code >= 400
    assert response.status_code != 200


def test_a_body_of_a_reasonable_size_is_not_touched():
    token = issue_token(get_settings(), uuid4(), timedelta(minutes=5))

    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/api/v1/answer",
            json={"question": "к" * 900},
            headers={"Authorization": f"Bearer {token}"},
        )

    # 502 because there is no indexer here to answer — the point is that the
    # request reached the endpoint at all.
    assert response.status_code != 413


def test_an_upload_gets_the_upload_ceiling_and_not_the_json_one():
    """The two file endpoints legitimately carry sixty times the JSON limit;
    a shared ceiling would mean either broken uploads or a useless guard."""
    body = b"%PDF-1.4\n" + b"0" * (JSON_LIMIT + 1000)

    with TestClient(app, base_url="https://testserver") as client:
        response = client.post("/api/v1/documents", files={"file": ("big.pdf", body)})

    # 401, having passed the guard: no token was sent.
    assert response.status_code == 401


def test_a_get_is_not_measured():
    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/api/health").status_code == 200
