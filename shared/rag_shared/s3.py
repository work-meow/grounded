"""The one rule about S3 addressing that two processes have to agree on.

The indexer reads objects with its own client; the API signs links for the same
objects with another. A signature covers the host, so if one of them puts the
bucket in the path and the other puts it in front of the host, the link is
formed correctly, signed correctly, and refused — from a mismatch neither side
can see on its own.

So the rule lives here, once.
"""

#: Where the bucket name goes. Measured against botocore rather than read off a
#: page: with an explicit endpoint — which every connected store has, it is a
#: required field — "auto" resolves to path style even for AWS, so the choice
#: has to be made outright.
PATH = "path"
VIRTUAL = "virtual"


def addressing_style(endpoint: str) -> str:
    """Path style, unless the endpoint is AWS's own.

    MinIO, Ceph and most self-hosted gateways answer only path-style requests:
    ``bucket.minio:9000`` is not a name that resolves. AWS still serves both but
    has deprecated path style for buckets made since 2020, so it gets the form
    it prefers. Every other provider measured — Backblaze, R2, Wasabi — takes
    path style.
    """
    return VIRTUAL if ".amazonaws.com" in endpoint else PATH
