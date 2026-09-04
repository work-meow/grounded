"""OneDrive folders as documents, over Microsoft Graph.

Graph walks a drive one folder at a time, so this is a breadth-first traversal
of the connected folder with both the number of folders and the pages per folder
bounded — a personal drive is a tree of unknown shape and an unbounded walk of
it is an unbounded number of requests every ten minutes.

Downloading an item is a redirect to a storage host. The ``Authorization``
header must not follow it there, and httpx drops it on a cross-origin redirect,
which is the behaviour this relies on.
"""

from collections.abc import Iterable, Iterator
from typing import Any

from rag_shared.connectors import ConnectorSpec

from rag_indexer.connectors.http import Http, OAuthToken, as_timestamp
from rag_indexer.connectors.remote import RemoteFile, folder_path

_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
_GRAPH = "https://graph.microsoft.com/v1.0"
_PAGE_SIZE = 200
_MAX_FOLDERS = 500
_MAX_PAGES_PER_FOLDER = 25


class OneDriveSource:
    def __init__(self, spec: ConnectorSpec) -> None:
        label = f"onedrive source {spec.source_id}"
        self._http = Http(label)
        self._token = OAuthToken(
            label,
            url=_TOKEN_URL,
            data={
                "client_id": spec.config["client_id"],
                "client_secret": spec.config["client_secret"],
                "refresh_token": spec.config["refresh_token"],
                "grant_type": "refresh_token",
                # offline_access is what keeps the refresh token alive; without
                # it Microsoft returns an access token and no successor.
                "scope": "Files.Read.All offline_access",
            },
        )
        self._path = folder_path(spec.config["path"])

    def close(self) -> None:
        self._http.close()
        self._token.close()

    def list(self) -> Iterable[RemoteFile]:
        self._authorize()
        files: list[RemoteFile] = []
        queue = [self._root_url()]
        visited = 0
        while queue and visited < _MAX_FOLDERS:
            visited += 1
            for item in self._children(queue.pop(0)):
                if "folder" in item:
                    queue.append(f"{_GRAPH}/me/drive/items/{item['id']}/children")
                elif "file" in item:
                    files.append(_as_file(item))
        return files

    def fetch(self, file: RemoteFile, limit: int) -> bytes | None:
        self._authorize()
        return self._http.download(
            "GET", f"{_GRAPH}/me/drive/items/{file.external_id}/content", limit=limit
        )

    def _root_url(self) -> str:
        if not self._path:
            return f"{_GRAPH}/me/drive/root/children"
        # Graph's colon syntax: everything between the colons is a path, so it
        # needs no escaping of its own separators.
        return f"{_GRAPH}/me/drive/root:/{self._path}:/children"

    def _children(self, url: str | None) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] | None = {"$top": _PAGE_SIZE}
        for _ in range(_MAX_PAGES_PER_FOLDER):
            if not url:
                return
            payload = self._http.json("GET", url, params=params)
            yield from payload.get("value", [])
            # The continuation link already carries the paging parameters.
            url, params = payload.get("@odata.nextLink"), None

    def _authorize(self) -> None:
        self._http.set_header("Authorization", self._token.header())


def _as_file(item: dict[str, Any]) -> RemoteFile:
    return RemoteFile(
        external_id=item["id"],
        filename=item.get("name", ""),
        modified_at=as_timestamp(item.get("lastModifiedDateTime")),
        web_url=item.get("webUrl"),
        size=item.get("size"),
    )
