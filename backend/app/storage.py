"""S3 access for the original files.

The API only ever writes here. Reading and indexing is the indexer's job — it
watches the same bucket through Pathway's S3 connector.
"""

from contextlib import asynccontextmanager

import aioboto3
from botocore.config import Config

from app.config import Settings

_session = aioboto3.Session()


def _config(settings: Settings) -> Config:
    # Virtual-host addressing would turn the bucket into a subdomain, which no
    # self-hosted gateway routes. Path style is only forced when asked for, so
    # AWS keeps its own default.
    style = "path" if settings.s3_path_style else "auto"
    return Config(s3={"addressing_style": style}, signature_version="s3v4")


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
