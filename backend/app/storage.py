"""S3 access for the original files, and for the two documents the API and the
indexer use to talk.

Uploads are write-only from here: reading and indexing them is the indexer's
job, through Pathway's S3 connector on the same bucket. The exception is the
pair of small control documents — the connector manifest this process writes,
and the source-health document the indexer writes back (rag_shared.health) —
which is why there is a reader at all.
"""

import logging
from contextlib import asynccontextmanager

import aioboto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings

logger = logging.getLogger(__name__)

_session = aioboto3.Session()


def _config(settings: Settings) -> Config:
    # Virtual-host addressing would turn the bucket into a subdomain, which no
    # self-hosted gateway routes. Path style is only forced when asked for, so
    # AWS keeps its own default.
    style = "path" if settings.s3_path_style else "auto"
    return Config(
        s3={"addressing_style": style},
        signature_version="s3v4",
        # Explicit, because botocore's defaults are made for a batch job: five
        # attempts against sixty-second timeouts is four minutes of a user
        # waiting on an upload while the object store is wedged. Three attempts
        # in standard mode, and a connect that fails fast — the store is one hop
        # away on the same docker network, so a slow connect is a dead store.
        connect_timeout=5,
        read_timeout=settings.s3_timeout_s,
        retries={"max_attempts": 3, "mode": "standard"},
    )


@asynccontextmanager
async def _client(settings: Settings, endpoint: str | None = None):
    async with _session.client(
        "s3",
        region_name=settings.s3_region,
        endpoint_url=endpoint or settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=_config(settings),
    ) as client:
        yield client


async def put(settings: Settings, key: str, body: bytes, content_type: str) -> None:
    async with _client(settings) as client:
        await client.put_object(
            Bucket=settings.s3_bucket, Key=key, Body=body, ContentType=content_type
        )


async def get(settings: Settings, key: str) -> bytes | None:
    """One small object, or None if it is not there or could not be read.

    None rather than an exception, because the only caller is asking about a
    document written by another process that may not have started yet. A source
    list that failed because the indexer has never published its health would be
    a worse answer than a source list with nothing known about it.
    """
    try:
        async with _client(settings) as client:
            response = await client.get_object(Bucket=settings.s3_bucket, Key=key)
            return await response["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "NoSuchBucket"):
            logger.warning("could not read %s from the bucket", key)
        return None
    except BotoCoreError:
        logger.warning("could not read %s from the bucket", key)
        return None


async def delete(settings: Settings, key: str) -> None:
    async with _client(settings) as client:
        await client.delete_object(Bucket=settings.s3_bucket, Key=key)


async def presigned_url(settings: Settings, key: str, expires_in: int = 900) -> str:
    """Short-lived link so the UI can open a cited document.

    Signed against the public endpoint when there is one: the signature covers
    the host, so a link signed for the internal name would not verify from a
    browser even if the browser could reach it.
    """
    endpoint = settings.s3_public_url or settings.s3_endpoint_url
    async with _client(settings, endpoint) as client:
        return await client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=expires_in,
        )
