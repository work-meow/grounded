"""Google Drive folders as documents.

Access is arranged the opposite way round from the usual OAuth flow. The
deployment holds one service account; the user shares a folder with its address.
That is strictly less access than a consent screen would grant — one folder,
read-only, revoked from Drive's own sharing dialog — and it means this system
never holds a credential that could reach the rest of somebody's Drive.

Not ``pw.io.gdrive``, and the reason is that same sharing model. Pathway decides
how to list a folder by whether its root reports a ``parents`` field; a folder
*shared with* an account reports none, because its parent sits in somebody
else's Drive. Pathway reads that as "this is a drive root" and switches to
scanning everything the account owns — which, for an account that owns nothing
and was merely given a folder, is nothing at all. Measured against the real API:
``tree()`` returned zero files for a folder whose contents ``'<id>' in parents``
lists perfectly well. Its other strategy does work, but falls back to the broken
one above 32 directories, so forcing it would only move the silence further off.

Google's own formats are exported on the way out: a Doc becomes a .docx, which
is a format the parser already reads.
"""

from collections.abc import Iterator
from typing import Any

from rag_shared.connectors import ConnectorSpec

from rag_indexer.connectors.http import as_timestamp
from rag_indexer.connectors.remote import Listing, RemoteFile

_FOLDER = "application/vnd.google-apps.folder"

#: Google's own formats have no bytes until they are exported, and no file
#: extension at all. Naming them by what they export to is what lets the rest of
#: the system — the format check here, the parser later — treat them as files.
_EXPORTS = {
    "application/vnd.google-apps.document": (
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "application/vnd.google-apps.spreadsheet": (
        ".xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "application/vnd.google-apps.presentation": (
        ".pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
}

_FIELDS = "id, name, mimeType, modifiedTime, size, webViewLink"
#: One request can ask about several parents at once, which is how a wide
#: folder costs one round trip rather than one per subfolder.
_PARENTS_PER_REQUEST = 25
_PAGE_SIZE = 1000
#: A bound on the walk, like every other connector has. Unlike Pathway's, going
#: over it is reported rather than silently swapped for a strategy that returns
#: nothing.
_MAX_FOLDERS = 500


class GoogleDriveSource:
    def __init__(self, spec: ConnectorSpec, credentials: dict[str, Any]) -> None:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        self._root = spec.config["folder_id"]
        # No scopes, which is what Pathway does too: google-api-python-client
        # then signs its own JWT for the API audience, and what the account may
        # read is decided by what has been shared with it rather than by an
        # OAuth scope. Verified against the live API.
        self._drive = build(
            "drive",
            "v3",
            credentials=Credentials.from_service_account_info(credentials),
            cache_discovery=False,
        )

    def close(self) -> None:
        self._drive.close()

    def contents(self) -> Listing:
        files: list[RemoteFile] = []
        folders = [self._root]
        visited = 0
        complete = True

        while folders:
            if visited >= _MAX_FOLDERS:
                complete = False
                break
            batch, folders = folders[:_PARENTS_PER_REQUEST], folders[_PARENTS_PER_REQUEST:]
            visited += len(batch)
            for item in self._children(batch):
                if item.get("mimeType") == _FOLDER:
                    folders.append(item["id"])
                else:
                    files.append(_as_file(item))

        return Listing(files=files, complete=complete)

    def fetch(self, file: RemoteFile, limit: int) -> bytes | None:
        export = _EXPORTS.get(file.mime_type or "")
        if export is not None:
            payload = self._drive.files().export_media(fileId=file.external_id, mimeType=export[1])
        else:
            payload = self._drive.files().get_media(fileId=file.external_id)
        data = payload.execute(num_retries=3)
        return data if len(data) <= limit else None

    def _children(self, parents: list[str]) -> Iterator[dict[str, Any]]:
        query = " or ".join(f"'{parent}' in parents" for parent in parents)
        page_token = None
        while True:
            response = (
                self._drive.files()
                .list(
                    q=f"({query}) and trashed=false",
                    fields=f"nextPageToken, files({_FIELDS})",
                    pageSize=_PAGE_SIZE,
                    pageToken=page_token,
                    # A folder can be shared from a personal Drive or live on a
                    # shared drive; both have to be reachable.
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute(num_retries=3)
            )
            yield from response.get("files", [])
            page_token = response.get("nextPageToken")
            if not page_token:
                return


def _as_file(item: dict[str, Any]) -> RemoteFile:
    """One Drive entry as a document.

    Everything is returned, readable or not: the polling loop is what decides
    what it can parse and says so once per file. A folder with a hundred holiday
    photos in it should be able to say why none of them are in the index.
    """
    name = str(item.get("name") or "")
    mime = str(item.get("mimeType") or "")
    if (export := _EXPORTS.get(mime)) is not None and not name.endswith(export[0]):
        name += export[0]
    return RemoteFile(
        external_id=item["id"],
        filename=name,
        modified_at=as_timestamp(item.get("modifiedTime")),
        # Drive's own page for the file, which for a Google Doc opens the editor
        # rather than a download.
        web_url=item.get("webViewLink"),
        # Reported for Docs Editors files too, so an oversized document is
        # skipped before it is exported rather than after.
        size=_as_int(item.get("size")),
        mime_type=mime,
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
