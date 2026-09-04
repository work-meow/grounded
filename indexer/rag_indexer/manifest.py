"""Which sources to connect, and when to rebuild the index.

The API owns the source list and the indexer owns the index, and the two cannot
simply talk: a Pathway dataflow graph is fixed once built, so there is no
message that adds a connector to a running process. The rebuild *is* the
restart.

So the API writes the list into the bucket both processes already share, and
this module does two things with it: reads it at startup, and watches its ETag
while the process runs. When it changes, the process exits and the container
policy brings it back — a few seconds, during which search is unavailable and
after which nothing has to be re-embedded, because the embedding cache is keyed
on content and survives on disk.

Reading it costs one HEAD every 30 seconds against our own MinIO, which is the
cheapest reliable way to notice a write by another process.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass

import botocore.session
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from rag_shared.connectors import MANIFEST_KEY, ConnectorSpec, load_manifest
from rag_shared.crypto import Sealer

from rag_indexer.config import IndexerSettings

logger = logging.getLogger(__name__)

#: Distinct from a crash so that a restart loop is legible in `docker logs`.
RESTART_CODE = 75


@dataclass(frozen=True, slots=True)
class Manifest:
    specs: list[ConnectorSpec]
    #: What the bucket held when these specs were read. Empty when there is no
    #: manifest yet, which is the normal state until the first source is added.
    etag: str


def load(settings: IndexerSettings) -> Manifest:
    """The current source list. Never raises: no manifest means uploads only."""
    if not settings.secrets_key:
        logger.warning("SECRETS_KEY is not set; connected sources are disabled")
        return Manifest(specs=[], etag="")

    try:
        sealer = Sealer(settings.secrets_key)
    except ValueError:
        logger.exception("SECRETS_KEY is unusable; connected sources are disabled")
        return Manifest(specs=[], etag="")

    raw, etag = _fetch(settings)
    if raw is None:
        return Manifest(specs=[], etag="")
    return Manifest(specs=load_manifest(raw, sealer), etag=etag)


def watch(settings: IndexerSettings, etag: str) -> None:
    """Restart this process when the source list changes underneath it."""
    thread = threading.Thread(
        target=_watch, args=(settings, etag), name="manifest-watch", daemon=True
    )
    thread.start()


def _watch(settings: IndexerSettings, etag: str) -> None:
    # One client for the life of the thread. Building a botocore client parses
    # the service model from JSON, which is far more work than the HEAD it is
    # for, and this runs every thirty seconds forever.
    client = _client(settings)
    while True:
        time.sleep(settings.manifest_poll_s)
        current = _etag(client, settings)
        if current is None or current == etag:
            continue
        logger.warning("the connected sources changed; restarting to rebuild the index")
        # os._exit, not sys.exit: Pathway's engine runs on threads that would
        # not see the exception, and there is nothing here worth unwinding — the
        # index is rebuilt from the bucket either way.
        logging.shutdown()
        os._exit(RESTART_CODE)


def _client(settings: IndexerSettings):
    return botocore.session.get_session().create_client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=Config(
            s3={"addressing_style": "path" if settings.s3_path_style else "auto"},
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
        ),
    )


def _fetch(settings: IndexerSettings) -> tuple[bytes | None, str]:
    try:
        response = _client(settings).get_object(Bucket=settings.s3_bucket, Key=MANIFEST_KEY)
        return response["Body"].read(), str(response.get("ETag", ""))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("no connector manifest yet; indexing uploads only")
        else:
            logger.exception("could not read the connector manifest")
        return None, ""
    except BotoCoreError:
        logger.exception("could not read the connector manifest")
        return None, ""


def _etag(client, settings: IndexerSettings) -> str | None:
    """The manifest's ETag, "" if there is none, or None if we could not ask.

    The distinction matters: MinIO being briefly unreachable must not read as
    "the manifest was deleted" and restart the process.
    """
    try:
        head = client.head_object(Bucket=settings.s3_bucket, Key=MANIFEST_KEY)
        return str(head.get("ETag", ""))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NoSuchBucket"):
            return ""
        logger.warning("could not check the connector manifest; will try again")
        return None
    except BotoCoreError:
        logger.warning("could not check the connector manifest; will try again")
        return None
