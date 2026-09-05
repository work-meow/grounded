"""What the S3 connector has to get right without a network.

Half of it is the listing: an object store answers with keys, not files, and
the difference shows up in two places — the folder markers a console writes,
and a bucket bigger than one pass can honestly report on.

The other half is the endpoint. This is the only connector whose address comes
from a form, and it runs in a process that sits on a private network with the
object store, the API and the database on it.
"""

import datetime
import io
import uuid

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from rag_shared.connectors import ConnectorSpec, Kind

from rag_indexer.connectors.remote import RemoteFile
from rag_indexer.connectors.s3 import MAX_OBJECTS, S3Source

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
SOURCE = uuid.UUID("22222222-2222-2222-2222-222222222222")

#: A literal rather than a name: it is globally routable, so the address check
#: passes, and it resolves without asking anybody — a unit test that needs DNS
#: is a unit test that fails on a train.
ENDPOINT = "https://93.184.216.34:9000"

WHEN = datetime.datetime(2026, 3, 1, 10, 0, tzinfo=datetime.UTC)


def source(**overrides) -> S3Source:
    config = {
        "endpoint_url": ENDPOINT,
        "bucket": "documents",
        "access_key_id": "key",
        "secret_access_key": "secret",
        **overrides,
    }
    return S3Source(
        ConnectorSpec(source_id=SOURCE, user_id=USER, kind=Kind.S3, name="store", config=config)
    )


def _object(key: str, size: int = 1024) -> dict:
    return {"Key": key, "LastModified": WHEN, "Size": size, "ETag": '"x"'}


def _listing(built: S3Source, contents: list[dict], *, truncated: bool = False):
    stub = Stubber(built._client)
    stub.add_response(
        "list_objects_v2",
        {"Contents": contents, "IsTruncated": truncated, "KeyCount": len(contents)},
    )
    return stub


# --- reading a bucket --------------------------------------------------------


def test_an_object_becomes_a_document():
    built = source()
    with _listing(built, [_object("договоры/аренда.pdf", 2048)]):
        (file,) = built.contents().files

    # The key, whole: it is what the object is called at the far end, and what
    # a signed link has to name again.
    assert file.external_id == "договоры/аренда.pdf"
    assert file.modified_at == int(WHEN.timestamp())
    assert file.size == 2048


def test_a_folder_marker_is_not_a_document():
    """A zero-byte key ending in a slash is how a console draws a folder."""
    built = source()
    with _listing(built, [_object("договоры/", 0), _object("договоры/аренда.pdf")]):
        listing = built.contents()

    assert [f.external_id for f in listing.files] == ["договоры/аренда.pdf"]


def test_a_bucket_bigger_than_one_pass_is_reported_as_incomplete():
    """The polling loop removes documents that have gone missing from a listing.
    A listing that stopped early has not lost them, and saying so is what keeps
    the tail of a large bucket from flickering out of search every pass."""
    built = source()
    with _listing(built, [_object(f"{n}.pdf") for n in range(MAX_OBJECTS + 5)]):
        listing = built.contents()

    assert listing.complete is False
    assert len(listing.files) == MAX_OBJECTS


def test_an_empty_bucket_is_an_empty_listing_not_a_failure():
    built = source()
    stub = Stubber(built._client)
    stub.add_response("list_objects_v2", {"IsTruncated": False, "KeyCount": 0})
    with stub:
        listing = built.contents()

    assert listing.files == []
    assert listing.complete is True


@pytest.mark.parametrize(
    ("given", "expected"), [("договоры", "договоры/"), ("/договоры/", "договоры/"), ("", "")]
)
def test_a_prefix_is_asked_for_the_way_the_store_expects(given, expected):
    """With the trailing slash put back: "notes" would otherwise also match
    "notes-2025", which is a different folder to everyone but the API."""
    assert source(prefix=given)._prefix == expected


# --- reading one object ------------------------------------------------------


def test_an_object_over_the_limit_is_refused_rather_than_returned():
    """The size in a listing is the store's word about itself. The read is
    bounded anyway, because "usually right" is not a memory limit."""
    built = source()
    stub = Stubber(built._client)
    stub.add_response("get_object", {"Body": _body(b"x" * 5000)})
    file = _remote("big.pdf")
    with stub:
        assert built.fetch(file, 1024) is None


def test_an_object_within_the_limit_comes_back_whole():
    built = source()
    stub = Stubber(built._client)
    stub.add_response("get_object", {"Body": _body(b"x" * 500)})
    with stub:
        assert built.fetch(_remote("small.pdf"), 1024) == b"x" * 500


def _body(payload: bytes) -> StreamingBody:
    return StreamingBody(io.BytesIO(payload), len(payload))


def _remote(key: str) -> RemoteFile:
    return RemoteFile(external_id=key, filename=key, modified_at=1, size=None)


# --- the address -------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    ["http://minio:9000", "http://127.0.0.1:9000", "http://169.254.169.254", "http://10.1.2.3"],
)
def test_a_store_inside_the_network_is_refused_before_a_client_exists(endpoint):
    """The check the API also makes, repeated here because this is the process
    that does the connecting — and because a manifest is not obliged to have
    come from the current API."""
    with pytest.raises(ValueError):
        source(endpoint_url=endpoint)


def test_the_address_is_checked_again_on_every_pass(monkeypatch):
    """The process runs for days. A name that answers with a public address
    today can answer with a private one tomorrow, and the client would follow
    it without being asked again."""
    built = source()
    checked: list[str] = []
    monkeypatch.setattr("rag_indexer.connectors.s3.verify_public", lambda url: checked.append(url))
    with _listing(built, []):
        built.contents()

    assert checked == [ENDPOINT]
