"""Any S3-compatible object store, as a source of documents.

One connector rather than one per vendor. MinIO, Backblaze B2, Cloudflare R2,
Wasabi, Selectel, VK Cloud and AWS itself all speak the same two calls this
needs — list the objects under a prefix, fetch one of them — and differ only in
the endpoint, which is a field.

It is also the first connector where the user names the machine we connect to.
Every other one talks to an address compiled into this file; this one takes an
address from a form, and the indexer sits on a private network with the object
store, the API and the database on it. So the endpoint is checked against
:mod:`rag_shared.net` before the client is built and again before every pass —
see the note there about what that does and does not cover.
"""

import logging
from typing import Any

import botocore.session
from botocore.config import Config
from rag_shared.connectors import ConnectorSpec
from rag_shared.net import verify_public
from rag_shared.s3 import addressing_style

from rag_indexer.connectors.remote import Listing, RemoteFile, folder_path

logger = logging.getLogger(__name__)

#: A bound on one listing, like every other connector has. Ten thousand objects
#: is far more than a personal document store and still one page-through that
#: finishes; past it the listing is reported as incomplete so that the polling
#: loop does not read the missing tail as deletions.
MAX_OBJECTS = 10_000

#: What botocore falls back to when the store does not care. MinIO, R2 and most
#: self-hosted gateways ignore the region entirely; AWS signs with it, which is
#: why it is a field the user can set.
DEFAULT_REGION = "us-east-1"

#: Reading the object is a request against somebody else's box, on a timer.
#: Long enough for a 64 MB document over a slow link, short enough that a wedged
#: store costs one pass rather than the process.
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 120


class S3Source:
    def __init__(self, spec: ConnectorSpec) -> None:
        config = spec.config
        self._endpoint = str(config["endpoint_url"]).strip().rstrip("/")
        verify_public(self._endpoint)

        self._bucket = str(config["bucket"]).strip()
        # A prefix, not a path: S3 has no folders, only keys that happen to
        # contain slashes. The trailing one is added back because "notes" must
        # not also match "notes-2025".
        prefix = folder_path(str(config.get("prefix") or ""))
        self._prefix = f"{prefix}/" if prefix else ""
        self._client = botocore.session.get_session().create_client(
            "s3",
            endpoint_url=self._endpoint,
            region_name=str(config.get("region") or "").strip() or DEFAULT_REGION,
            aws_access_key_id=str(config["access_key_id"]).strip(),
            aws_secret_access_key=str(config["secret_access_key"]).strip(),
            config=Config(
                s3={"addressing_style": addressing_style(self._endpoint)},
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=_CONNECT_TIMEOUT,
                read_timeout=_READ_TIMEOUT,
            ),
        )

    def close(self) -> None:
        self._client.close()

    def contents(self) -> Listing:
        # Again, not only at construction: this process runs for days, and a
        # name that answers with a public address today can answer with a
        # private one tomorrow. The cost is a resolver lookup per pass.
        verify_public(self._endpoint)

        files: list[RemoteFile] = []
        complete = True
        for page in self._client.get_paginator("list_objects_v2").paginate(
            Bucket=self._bucket, Prefix=self._prefix
        ):
            for entry in page.get("Contents") or []:
                if len(files) >= MAX_OBJECTS:
                    complete = False
                    break
                if (file := _as_file(entry)) is not None:
                    files.append(file)
            if not complete:
                break
        return Listing(files=files, complete=complete)

    def fetch(self, file: RemoteFile, limit: int) -> bytes | None:
        """The object's bytes, or None if it turned out to be larger than the limit.

        The size in a listing is the store's word about itself and the same
        store serves the body, so it is rarely wrong here — but the read is
        bounded anyway, because "rarely" is not a memory limit.
        """
        body = self._client.get_object(Bucket=self._bucket, Key=file.external_id)["Body"]
        try:
            payload = body.read(limit + 1)
        finally:
            body.close()
        return None if len(payload) > limit else payload


def _as_file(entry: dict[str, Any]) -> RemoteFile | None:
    """One object as a document, or None for something that is not one.

    A key ending in a slash is how the consoles write a folder: a zero-byte
    object that exists so the interface has something to draw.
    """
    key = str(entry.get("Key") or "")
    if not key or key.endswith("/"):
        return None
    modified = entry.get("LastModified")
    return RemoteFile(
        external_id=key,
        # The whole key: clean_name reduces it to the last segment, so
        # "docs/2026/Договор.pdf" is shown as its file name and still keeps a
        # key of its own.
        filename=key,
        modified_at=int(modified.timestamp()) if modified is not None else 0,
        size=entry.get("Size"),
        # No page to open: an object store has no browser interface. The API
        # signs a link on demand instead — see document_link.
        web_url=None,
    )
