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
    assert metadata.keys() == {"path", "modified_at", "seen_at", "web_url", "size"}


# --- through a real graph ----------------------------------------------------


def test_a_polled_source_produces_rows_pathway_can_read():
    """The one check that the metadata survives the trip into the engine.

    Everything above tests the diff in isolation. This runs the connector as
    Pathway actually runs it — subject, JSON metadata, Json column — because
    that is the boundary where a shape mistake would show up only in
    production, as documents that index but belong to nobody.
    """
    import pathway as pw
    from rag_shared.doc_key import tenant_metadata

    from rag_indexer.connectors.remote import polling_table

    body = "Срок уведомления — 45 дней".encode()
    table = polling_table(
        _Service([_file(name="Договор.md")], body=body),
        spec=ConnectorSpec(
            source_id=SOURCE, user_id=USER, kind=Kind.NOTION, name="test", config={}
        ),
        refresh_interval=0,
        size_limit=1 << 20,
        mode="static",
    )

    rows: list[dict] = []
    pw.io.subscribe(table, on_change=lambda key, row, time, is_addition: rows.append(row))
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)

    (row,) = rows
    assert row["data"] == body

    metadata = row["_metadata"]
    metadata = metadata.as_dict() if hasattr(metadata, "as_dict") else dict(metadata)
    # The post-processor the DocumentStore applies is what turns the path into
    # the fields the tenant filter matches on.
    _, tagged = tenant_metadata("", metadata)
    assert tagged["user_id"] == str(USER)
    assert tagged["source_id"] == str(SOURCE)
    assert tagged["filename"] == "Договор.md"


def test_every_kind_builds_and_the_tables_concatenate():
    """DocumentStore concatenates its inputs, which requires one schema.

    Each connector reaches its service differently and produces its metadata
    differently, so "they all still line up" is a claim that has to be checked
    rather than assumed — and a mismatch would only appear at startup, in
    production, as an index that refuses to build at all.
    """
    import json

    import pathway as pw

    from rag_indexer.config import IndexerSettings
    from rag_indexer.connectors import build_tables

    configs = {
        Kind.GDRIVE: {"folder_id": "1AbCdEf"},
        Kind.NOTION: {"token": "x"},
        Kind.YANDEX: {"token": "y", "path": "/Docs"},
        Kind.DROPBOX: {"app_key": "a", "app_secret": "b", "refresh_token": "c", "path": "/D"},
        Kind.ONEDRIVE: {"client_id": "a", "client_secret": "b", "refresh_token": "c", "path": "D"},
    }
    specs = [
        ConnectorSpec(source_id=uuid.uuid4(), user_id=USER, kind=kind, name=kind.value, config=c)
        for kind, c in configs.items()
    ]
    settings = IndexerSettings(
        openrouter_api_key="unused",
        gdrive_credentials_json=json.dumps({"type": "service_account"}),
    )

    tables = build_tables(settings, specs)

    # One per connected source, plus the uploads table that is always there.
    assert len(tables) == len(configs) + 1
    merged = pw.Table.concat_reindex(*(t.select(pw.this.data, pw.this._metadata) for t in tables))
    assert sorted(merged.column_names()) == ["_metadata", "data"]


def test_a_source_that_cannot_be_built_costs_only_itself():
    """One user's misconfiguration must not take the index down for everyone."""
    from rag_indexer.config import IndexerSettings
    from rag_indexer.connectors import build_tables

    broken = ConnectorSpec(
        source_id=uuid.uuid4(),
        user_id=USER,
        kind=Kind.GDRIVE,
        name="no credentials on this deployment",
        config={"folder_id": "1AbCdEf"},
    )

    tables = build_tables(IndexerSettings(openrouter_api_key="unused"), [broken])

    assert len(tables) == 1, "the uploads table survives a Drive source that cannot be built"
