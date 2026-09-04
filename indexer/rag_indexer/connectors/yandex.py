"""Яндекс.Диск folders as documents.

The simplest of the four: the OAuth token a user issues at oauth.yandex.ru is
long-lived, so there is no refresh to run, and the REST API lists a folder
directly. Downloading is two requests — one for a short-lived href, one for the
bytes — which is why the listing is not also used as a download plan.

Folders are walked breadth-first with the same bounds as OneDrive, and for the
same reason.
"""

from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import quote

from rag_shared.connectors import ConnectorSpec

from rag_indexer.connectors.http import Http, as_timestamp
from rag_indexer.connectors.remote import RemoteFile, folder_path

_API = "https://cloud-api.yandex.net/v1/disk"
_PAGE_SIZE = 200
_MAX_FOLDERS = 500
_MAX_PAGES_PER_FOLDER = 25
#: The API prefixes every path with this; the web client's URLs do not.
_DISK = "disk:"


class YandexSource:
    def __init__(self, spec: ConnectorSpec) -> None:
        label = f"yandex source {spec.source_id}"
        self._http = Http(label, headers={"Authorization": f"OAuth {spec.config['token']}"})
        # A second client, without the token. Downloads go to a pre-signed href
        # on a storage host that does not want the credential and has no use for
        # it; there is no reason for it to travel further than the API.
        self._downloads = Http(label)
        # No rstrip: strip("/") has already removed a trailing slash, and
        # stripping again would turn the disk root "disk:/" into "disk:", which
        # the API does not recognise.
        self._path = f"{_DISK}/{folder_path(spec.config['path'])}"

    def close(self) -> None:
        self._http.close()
        self._downloads.close()

    def list(self) -> Iterable[RemoteFile]:
        files: list[RemoteFile] = []
        queue = [self._path]
        visited = 0
        while queue and visited < _MAX_FOLDERS:
            visited += 1
            for item in self._items(queue.pop(0)):
                if item.get("type") == "dir":
                    queue.append(item["path"])
                elif item.get("type") == "file":
                    files.append(_as_file(item))
        return files

    def fetch(self, file: RemoteFile, limit: int) -> bytes | None:
        # The href is signed and expires in minutes, so it is fetched per
        # download rather than kept from the listing.
        href = self._http.json(
            "GET", f"{_API}/resources/download", params={"path": file.external_id}
        )["href"]
        return self._downloads.download("GET", href, limit=limit)

    def _items(self, path: str) -> Iterator[dict[str, Any]]:
        for page in range(_MAX_PAGES_PER_FOLDER):
            payload = self._http.json(
                "GET",
                f"{_API}/resources",
                params={"path": path, "limit": _PAGE_SIZE, "offset": page * _PAGE_SIZE},
            )
            items = (payload.get("_embedded") or {}).get("items") or []
            yield from items
            if len(items) < _PAGE_SIZE:
                return


def _as_file(item: dict[str, Any]) -> RemoteFile:
    path = item.get("path", "")
    folder, _, _ = path.rpartition("/")
    return RemoteFile(
        # The path, not resource_id: it is what the download endpoint takes, and
        # a moved file on Яндекс.Диск is a new document either way because the
        # API gives no rename-stable handle the download call accepts.
        external_id=path,
        filename=item.get("name", ""),
        modified_at=as_timestamp(item.get("modified")),
        web_url=f"https://disk.yandex.ru/client/disk{quote(folder.removeprefix(_DISK))}",
        size=item.get("size"),
    )
