"""S3 access for the original files.

The API only ever writes here. Reading and indexing is the indexer's job — it
watches the same bucket through Pathway's S3 connector.
"""

from contextlib import asynccontextmanager

import aioboto3

from app.config import Settings

_session = aioboto3.Session()


@asynccontextmanager
async def _client(settings: Settings):
    async with _session.client(
        "s3",
        region_name=settings.s3_region,
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
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
    """Short-lived link so the UI can open a cited document."""
    async with _client(settings) as client:
        return await client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=expires_in,
        )
