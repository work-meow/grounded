"""Connecting an S3-compatible store, and the one thing that is new about it.

Every connector until now talked to an address compiled into the source. This
is the first where the user names the machine, and the indexer sits on a private
network with the object store, the API and the database on it — so most of what
is worth testing here is not S3 at all, it is which addresses this server can be
talked into fetching.
"""

import pytest
from rag_shared.connectors import (
    IDENTITY_FIELDS,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    Kind,
    fingerprint,
)
from rag_shared.net import verify_public

from app import connectors, retriever, storage
from app.config import Settings

WHOLE = {
    "endpoint_url": "https://s3.us-west-004.backblazeb2.com",
    "bucket": "documents",
    "access_key_id": "key",
    "secret_access_key": "secret",
}


def settings(**overrides) -> Settings:
    return Settings(jwt_secret="x" * 40, openrouter_api_key="k", **overrides)


# --- which addresses this server may be sent to ------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9000",
        "http://localhost:9000",
        "http://10.0.0.5",
        "http://192.168.1.10:9000",
        "http://172.17.0.2:9000",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:9000",
        # The same address, spelled to get past a list of strings. Resolution is
        # what makes these come out as what they are.
        "http://2130706433/",
        "http://0x7f000001/",
    ],
)
def test_an_address_inside_the_network_is_refused(url):
    with pytest.raises(ValueError):
        verify_public(url)


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com", "file:///etc/passwd", "gopher://example.com", "https://", "not a url"],
)
def test_something_that_is_not_a_fetchable_address_is_refused(url):
    with pytest.raises(ValueError):
        verify_public(url)


def test_a_name_that_does_not_resolve_is_refused():
    """Including the docker-network names that are the whole reason for this:
    from outside they do not resolve, from inside they resolve to a private
    address, and both are a refusal."""
    with pytest.raises(ValueError):
        verify_public("http://minio.invalid:9000")


def test_a_public_address_is_allowed():
    verify_public("https://s3.amazonaws.com")


async def test_the_api_refuses_the_endpoint_before_it_is_sealed():
    """The check that matters is the indexer's, which runs where the connecting
    happens. This one exists so somebody typing an internal address is told at
    once instead of getting a source that never indexes."""
    with pytest.raises(ValueError):
        await connectors.verify_endpoint({**WHOLE, "endpoint_url": "http://127.0.0.1:9000"})


async def test_a_config_without_an_endpoint_passes_through():
    """Every other kind has no such field, and must not be made to have one."""
    await connectors.verify_endpoint({"token": "ntn_x"})


# --- the shape of the configuration ------------------------------------------


def test_the_optional_fields_are_kept_when_given_and_dropped_when_blank():
    """An absent value and an empty one have to reach the indexer as the same
    thing, or "" ends up being signed as a region."""
    assert connectors.clean_config(
        Kind.S3, {**WHOLE, "region": " eu-central-1 ", "prefix": ""}
    ) == {
        **WHOLE,
        "region": "eu-central-1",
    }


@pytest.mark.parametrize("field", REQUIRED_FIELDS[Kind.S3])
def test_a_missing_required_field_is_named(field):
    with pytest.raises(ValueError, match=field):
        connectors.clean_config(Kind.S3, {**WHOLE, field: "  "})


def test_the_place_is_what_identifies_the_source_not_the_keys():
    """The only kind where it can be: an endpoint, a bucket and a prefix name
    the object exactly, so a second set of keys to the same bucket is the same
    source rather than a second copy of it."""
    assert "access_key_id" not in IDENTITY_FIELDS[Kind.S3]
    assert fingerprint(Kind.S3, WHOLE) == fingerprint(
        Kind.S3, {**WHOLE, "access_key_id": "other", "secret_access_key": "other"}
    )
    assert fingerprint(Kind.S3, WHOLE) != fingerprint(Kind.S3, {**WHOLE, "bucket": "elsewhere"})


def test_every_kind_answers_the_optional_lookup():
    """The response model asks for one list per kind; a kind added without an
    entry would render a form missing its optional half."""
    for kind in Kind:
        assert isinstance(OPTIONAL_FIELDS.get(kind, ()), tuple)


# --- the link to an object ----------------------------------------------------


async def test_a_signed_link_points_at_the_object_and_carries_no_secret():
    """An object store has no page to open, so this link is the only way to hand
    somebody the file. It is signed locally: no request leaves this process,
    which is why a user-supplied endpoint is not a fetch the server performs."""
    url = await storage.foreign_presigned_url(
        {**WHOLE, "region": "us-west-004"}, "договоры/аренда.pdf"
    )

    assert url.startswith("https://s3.us-west-004.backblazeb2.com/documents/")
    assert "X-Amz-Signature=" in url
    assert WHOLE["secret_access_key"] not in url


async def test_a_link_expires():
    url = await storage.foreign_presigned_url(WHOLE, "a.pdf", expires_in=60)

    assert "X-Amz-Expires=60" in url


async def test_aws_is_signed_the_way_aws_expects():
    """Path style for self-hosted gateways, the bucket as a subdomain for AWS.
    Measured against botocore: with an explicit endpoint — which every connected
    store has — "auto" resolves to path style even for AWS, so the choice has to
    be made outright rather than left to it."""
    url = await storage.foreign_presigned_url(
        {**WHOLE, "endpoint_url": "https://s3.eu-central-1.amazonaws.com"}, "a.pdf"
    )

    assert url.startswith("https://documents.s3.eu-central-1.amazonaws.com/a.pdf")


def test_a_link_that_is_not_one_never_reaches_the_browser():
    assert retriever.openable("javascript:alert(1)") is None
    assert retriever.openable("https://s3.test/a.pdf?X-Amz-Signature=x") is not None
