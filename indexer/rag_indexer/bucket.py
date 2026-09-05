"""The object-store client, shared by both directions of the bucket channel.

The indexer talks to the bucket for three things: Pathway reads the uploads
through its own S3 connector, :mod:`rag_indexer.manifest` reads the source list
the API writes, and :mod:`rag_indexer.health` writes back how those sources are
doing. The last two are ours, and they need the same client built the same way.
"""

from typing import Any

import botocore.session
from botocore.config import Config

from rag_indexer.config import IndexerSettings


def client(settings: IndexerSettings) -> Any:
    """A botocore S3 client.

    Worth holding on to. Building one parses the service model out of JSON,
    which is far more work than any single call made through it — and these run
    on a timer, forever.
    """
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
