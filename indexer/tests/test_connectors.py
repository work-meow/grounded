"""What a connector promises, without a network.

Every connector talks to a live service, so what can be tested here is the half
that is ours: the metadata each one has to produce, and the diff that decides
what the index is told. Those are also the two places where a mistake is silent
— a document with the wrong key is not visible to its owner and not an error
anywhere, and a diff that misses a change leaves a stale answer in the index.
"""

import json
import uuid

import pytest
from rag_shared.connectors import ConnectorSpec, Kind
from rag_shared.doc_key import document_id_for, parse_key

from rag_indexer.connectors.contract import FALLBACK_NAME, clean_name, describe
from rag_indexer.connectors.remote import RemoteFile, _PollingSubject

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
SOURCE = uuid.UUID("22222222-2222-2222-2222-222222222222")


# --- the metadata contract ---------------------------------------------------


def test_description_parses_back_into_the_key_layout():
    metadata = describe(
        user_id=USER,
        source_id=SOURCE,
        external_id="drive-file-1",
        filename="Отчёт.pdf",
        modified_at=1_700_000_000,
        web_url="https://drive.google.com/file/d/x/view",
    )

    parsed = parse_key(metadata["path"])
    assert parsed is not None, "a connector document must satisfy the tenant filter"
    assert parsed["user_id"] == str(USER)
    assert parsed["source_id"] == str(SOURCE)
    assert parsed["document_id"] == str(document_id_for(SOURCE, "drive-file-1"))
    assert parsed["filename"] == "Отчёт.pdf"
    assert metadata["modified_at"] == 1_700_000_000
    assert metadata["seen_at"] >= metadata["modified_at"]


def test_the_same_remote_file_keeps_its_document_id():
    """Citations written today must still resolve after tomorrow's restart."""
    first = describe(
        user_id=USER, source_id=SOURCE, external_id="page-7", filename="a.md", modified_at=1
    )
    later = describe(
        user_id=USER, source_id=SOURCE, external_id="page-7", filename="a.md", modified_at=999
    )
    assert parse_key(first["path"])["document_id"] == parse_key(later["path"])["document_id"]


def test_the_same_file_reached_twice_is_two_documents():
    other_source = uuid.uuid4()
    assert document_id_for(SOURCE, "page-7") != document_id_for(other_source, "page-7")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("folder/sub/report.pdf", "report.pdf"),
        ("windows\\path\\report.pdf", "report.pdf"),
        ("bad\x07name.txt", "badname.txt"),
        ("line\nbreak.md", "linebreak.md"),
        ("   ", FALLBACK_NAME),
        ("", FALLBACK_NAME),
    ],
)
def test_remote_names_are_reduced_to_something_safe(given, expected):
    assert clean_name(given) == expected


def test_a_very_long_name_still_parses():
    metadata = describe(
        user_id=USER,
        source_id=SOURCE,
        external_id="x",
        filename="я" * 5000 + ".txt",
        modified_at=1,
    )
    assert parse_key(metadata["path"]) is not None


# --- the polling diff --------------------------------------------------------


class _Service:
    """A remote service that holds whatever the test puts in it."""

    def __init__(self, files: list[RemoteFile], *, body: bytes = b"hello"):
        self.files = files
        self.body = body
        self.fetched: list[str] = []
        self.fail_on: set[str] = set()
        self.listing_fails = False

    def list(self):
        if self.listing_fails:
            raise ConnectionError("the service is down")
        return list(self.files)

    def fetch(self, file: RemoteFile) -> bytes:
        self.fetched.append(file.external_id)
        if file.external_id in self.fail_on:
            raise ConnectionError("that one download failed")
        return self.body

    def close(self) -> None:
        pass


class _Subject(_PollingSubject):
    """The real diff, with the two calls into Pathway recorded instead."""

    def __init__(self, service: _Service, *, size_limit: int = 1024):
        super().__init__(
            service,
            spec=ConnectorSpec(
                source_id=SOURCE, user_id=USER, kind=Kind.NOTION, name="test", config={}
            ),
            refresh_interval=0,
            size_limit=size_limit,
        )
        self.added: list[dict] = []
        self.removed: list[str] = []

    def _add(self, key, message, metadata=None):
        self.added.append(json.loads(metadata))

    def _remove(self, key, message, metadata=None):
        self.removed.append(str(key))


def _file(external_id="f1", *, name="doc.md", modified_at=1, size=None):
    return RemoteFile(external_id=external_id, filename=name, modified_at=modified_at, size=size)


def test_a_new_file_is_indexed_once_and_not_again():
    service = _Service([_file()])
    subject = _Subject(service)
    seen: dict[str, int] = {}

    subject._poll(seen)
    subject._poll(seen)

    assert len(subject.added) == 1, "an unchanged file must not be re-embedded every poll"
    assert service.fetched == ["f1"]


def test_an_edited_file_is_re_indexed():
    service = _Service([_file(modified_at=1)])
    subject = _Subject(service)
    seen: dict[str, int] = {}
    subject._poll(seen)

    service.files = [_file(modified_at=2)]
    subject._poll(seen)

    assert len(subject.added) == 2
    assert subject.removed == [], "an upsert replaces the document, it does not delete it"


def test_a_deleted_file_leaves_the_index():
    service = _Service([_file("f1"), _file("f2")])
    subject = _Subject(service)
    seen: dict[str, int] = {}
    subject._poll(seen)

    service.files = [_file("f1")]
    subject._poll(seen)

    assert len(subject.removed) == 1
    assert set(seen) == {"f1"}


def test_a_service_that_is_down_changes_nothing():
    """The last good snapshot is worth more than an empty one."""
    service = _Service([_file()])
    subject = _Subject(service)
    seen: dict[str, int] = {}
    subject._poll(seen)

    service.listing_fails = True
    subject._poll(seen)

    assert subject.removed == [], "an outage must not look like every file being deleted"
    assert set(seen) == {"f1"}


def test_a_failed_download_is_retried_on_the_next_pass():
    service = _Service([_file()])
    service.fail_on = {"f1"}
    subject = _Subject(service)
    seen: dict[str, int] = {}

    subject._poll(seen)
    assert seen == {}, "a file that could not be read is not a file that is indexed"

    service.fail_on = set()
    subject._poll(seen)
    assert len(subject.added) == 1


def test_a_format_we_cannot_read_is_skipped_without_downloading_it():
    service = _Service([_file(name="holiday.mp4")])
    subject = _Subject(service)
    seen: dict[str, int] = {}

    subject._poll(seen)
    subject._poll(seen)

    assert service.fetched == [], "an unreadable file must not be downloaded at all"
    assert subject.added == []
    assert set(seen) == {"f1"}, "and must not be reconsidered on every poll"


def test_an_oversized_file_is_skipped_from_the_listing():
    service = _Service([_file(size=99_999)])
    subject = _Subject(service, size_limit=1024)

    subject._poll({})

    assert service.fetched == [], "the size in the listing is what saves the download"
    assert subject.added == []


def test_a_file_the_listing_lied_about_is_skipped_after_download():
    service = _Service([_file(size=10)], body=b"x" * 5000)
    subject = _Subject(service, size_limit=1024)

    subject._poll({})

    assert service.fetched == ["f1"]
    assert subject.added == [], "the real length is checked too"


def test_the_metadata_handed_to_pathway_is_ours():
    service = _Service([_file()])
    subject = _Subject(service)

    subject._poll({})

    (metadata,) = subject.added
    assert parse_key(metadata["path"])["user_id"] == str(USER)
    assert metadata.keys() == {"path", "modified_at", "seen_at", "web_url"}
