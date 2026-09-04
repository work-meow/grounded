"""The files a user uploads, from our own bucket.

Always present and never configured: uploads are the one source that needs no
credential, so it is wired in unconditionally rather than described by a
manifest entry.

This is also the only connector that needs no metadata rewriting. The API wrote
these keys, so they already are the key layout — which is exactly why that
layout is what every other connector is made to imitate.
"""

import pathway as pw
from rag_shared.doc_key import PREFIX

from rag_indexer.config import IndexerSettings


def build(settings: IndexerSettings) -> pw.Table:
    return pw.io.s3.read(
        # Only our own prefix; anything else in the bucket is ignored — which
        # now includes the connector manifest, written alongside it.
        path=f"{PREFIX}/",
        format="binary",
        mode="streaming",
        with_metadata=True,
        aws_s3_settings=pw.io.s3.AwsS3Settings(
            bucket_name=settings.s3_bucket,
            access_key=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            region=settings.s3_region,
            endpoint=settings.s3_endpoint_url,
            with_path_style=settings.s3_path_style,
        ),
    )
