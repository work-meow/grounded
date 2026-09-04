"""Dropbox folders as documents.

The user connects one folder, not the account: ``path`` scopes the listing, and
Dropbox itself scopes the token to whatever the app was granted. A recursive
listing of that folder is one request per thousand entries.

Files are addressed by ``id:`` rather than by path, so renaming or moving a file
inside the connected folder keeps it the same document and its citations keep
resolving.
"""

import json
from typing import Any
from urllib.parse import quote

from rag_shared.connectors import ConnectorSpec

from rag_indexer.connectors.http import Http, OAuthToken, as_timestamp
from rag_indexer.connectors.remote import Listing, RemoteFile, folder_path

_TOKEN_URL = "https://api.dropbox.com/oauth2/token"
_API = "https://api.dropboxapi.com/2"
_CONTENT = "https://content.dropboxapi.com/2"
_PAGE_SIZE = 1000
#: A folder that pages past this is not a document source any more.
_MAX_PAGES = 50


class DropboxSource:
    def __init__(self, spec: ConnectorSpec) -> None:
        label = f"dropbox source {spec.source_id}"
        self._http = Http(label)
        self._token = OAuthToken(
            label,
            url=_TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": spec.config["refresh_token"]},
            auth=(spec.config["app_key"], spec.config["app_secret"]),
        )
        # Dropbox spells the account root as the empty string, not "/".
        folder = folder_path(spec.config["path"])
        self._path = f"/{folder}" if folder else ""

    def close(self) -> None:
        self._http.close()
        self._token.close()

    def contents(self) -> Listing:
        entries, complete = self._entries()
        return Listing(
            files=[_as_file(entry) for entry in entries if entry.get(".tag") == "file"],
            complete=complete,
        )

    def fetch(self, file: RemoteFile, limit: int) -> bytes | None:
        self._authorize()
        return self._http.download(
            "POST",
            f"{_CONTENT}/files/download",
            limit=limit,
            # A header, so it has to be ASCII whatever the filename is. The path
            # is the file id, which is ASCII anyway; ensure_ascii keeps that
            # true if this ever becomes a real path.
            headers={"Dropbox-API-Arg": json.dumps({"path": file.external_id}, ensure_ascii=True)},
        )

    def _entries(self) -> tuple[list[dict[str, Any]], bool]:
        self._authorize()
        payload = self._http.json(
            "POST",
            f"{_API}/files/list_folder",
            json={"path": self._path, "recursive": True, "limit": _PAGE_SIZE},
        )
        entries: list[dict[str, Any]] = []
        for _ in range(_MAX_PAGES):
            entries.extend(payload.get("entries", []))
            if not payload.get("has_more"):
                return entries, True
            payload = self._http.json(
                "POST", f"{_API}/files/list_folder/continue", json={"cursor": payload["cursor"]}
            )
        return entries, False

    def _authorize(self) -> None:
        self._http.set_header("Authorization", self._token.header())


def _as_file(entry: dict[str, Any]) -> RemoteFile:
    path = entry.get("path_display") or entry.get("path_lower") or ""
    folder, _, _ = path.rpartition("/")
    name = entry.get("name", "")
    return RemoteFile(
        external_id=entry["id"],
        filename=name,
        # server_modified, not client_modified: the client's clock is whatever
        # the uploading device said it was, and a clock that runs backwards
        # would hide an edit from the change detector.
        modified_at=as_timestamp(entry.get("server_modified")),
        web_url=f"https://www.dropbox.com/home{quote(folder)}?preview={quote(name)}",
        size=entry.get("size"),
    )
