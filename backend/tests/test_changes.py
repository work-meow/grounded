"""What was added or changed lately, by source.

A base fed by Notion, Drive and a shared folder changes without anybody
watching, and the answer to "что я пропустил" used to be to scroll everything
by date and remember where you stopped.

A page and not a schedule, deliberately: there is no job runner in this
deployment, and adding one so that a summary could arrive by itself would be a
large piece of machinery for a question somebody asks on a Monday.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routers import sources as sources_router
from app.security import issue_token

NOW = datetime.now(UTC)


def document(name: str, source: str, days_ago: float, source_id=None) -> dict:
    return {
        "id": str(uuid4()),
        "filename": name,
        "mime_type": "text/markdown",
        "size_bytes": 100,
        "created_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "status": "ready",
        "source_id": str(source_id or uuid4()),
        "source_name": source,
        "removable": False,
    }


@pytest.fixture
def client():
    token = issue_token(get_settings(), uuid4(), timedelta(minutes=5))
    with TestClient(app, base_url="https://testserver") as opened:
        opened.headers["Authorization"] = f"Bearer {token}"
        yield opened


@pytest.fixture
def listed(monkeypatch):
    def serve(documents):
        async def list_documents(**_kwargs):
            return [sources_router.DocumentOut(**one) for one in documents]

        monkeypatch.setattr(sources_router, "list_documents", list_documents)

    return serve


def test_only_what_moved_inside_the_window(client, listed):
    listed(
        [
            document("вчерашний.md", "Notion", days_ago=1),
            document("прошлогодний.md", "Notion", days_ago=300),
        ]
    )

    body = client.get("/api/v1/changes", params={"days": 7}).json()

    assert body["total"] == 1
    assert body["sources"][0]["documents"][0]["filename"] == "вчерашний.md"


def test_documents_are_grouped_by_their_source(client, listed):
    notion, drive = uuid4(), uuid4()
    listed(
        [
            document("а.md", "Notion", 1, notion),
            document("б.md", "Notion", 2, notion),
            document("в.md", "Google Drive", 1, drive),
        ]
    )

    body = client.get("/api/v1/changes", params={"days": 7}).json()

    counted = {group["source_name"]: len(group["documents"]) for group in body["sources"]}
    assert counted == {"Notion": 2, "Google Drive": 1}


def test_the_source_that_moved_most_recently_comes_first(client, listed):
    """Whichever changed today, not whichever is first alphabetically."""
    old, fresh = uuid4(), uuid4()
    listed(
        [
            document("давно.md", "Архив", 6, old),
            document("сегодня.md", "Яндекс.Диск", 0.1, fresh),
        ]
    )

    body = client.get("/api/v1/changes", params={"days": 7}).json()

    assert [group["source_name"] for group in body["sources"]] == ["Яндекс.Диск", "Архив"]


def test_a_quiet_week_is_an_empty_answer_and_not_an_error(client, listed):
    listed([document("прошлогодний.md", "Notion", 300)])

    body = client.get("/api/v1/changes", params={"days": 7}).json()

    assert body == {"days": 7, "sources": [], "total": 0}


def test_a_source_with_nothing_new_is_not_listed(client, listed):
    """A page of sources that all say "ничего" is a page that answers nothing."""
    quiet, busy = uuid4(), uuid4()
    listed([document("старое.md", "Тихий", 100, quiet), document("новое.md", "Живой", 1, busy)])

    body = client.get("/api/v1/changes", params={"days": 7}).json()

    assert [group["source_name"] for group in body["sources"]] == ["Живой"]


@pytest.mark.parametrize("days", [0, 366, -1])
def test_a_window_that_is_not_a_window_is_refused(client, listed, days):
    listed([])

    assert client.get("/api/v1/changes", params={"days": days}).status_code == 422
