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
from rag_indexer.connectors.remote import Listing, RemoteFile, _PollingSubject

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
        self.listing_truncated = False

    def contents(self) -> Listing:
        if self.listing_fails:
            raise ConnectionError("the service is down")
        return Listing(files=list(self.files), complete=not self.listing_truncated)

    def fetch(self, file: RemoteFile, limit: int) -> bytes | None:
        self.fetched.append(file.external_id)
        if file.external_id in self.fail_on:
            raise ConnectionError("that one download failed")
        # What Http.download does: stop at the limit rather than return more.
        return self.body if len(self.body) <= limit else None

    def close(self) -> None:
        pass


class _Recorder:
    """The health channel, kept in memory."""

    def __init__(self):
        self.reports: list[tuple[bool, str, int | None]] = []

    def report(self, source_id, *, ok, problem="", documents=None):
        assert source_id == SOURCE
        self.reports.append((ok, problem, documents))


class _Subject(_PollingSubject):
    """The real diff, with the two calls into Pathway recorded instead."""

    def __init__(self, service: _Service, *, size_limit: int = 1024, reporter=None):
        super().__init__(
            service,
            spec=ConnectorSpec(
                source_id=SOURCE, user_id=USER, kind=Kind.NOTION, name="test", config={}
            ),
            refresh_interval=0,
            size_limit=size_limit,
            reporter=reporter or _Recorder(),
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


def test_a_listing_cut_short_removes_nothing():
    """Everything past the bound is missing from the listing, not deleted.

    Every connector walks a tree of unknown shape under a bound. Treating a
    truncated listing as deletions would drop the tail of a large source out of
    search and pull it back on the next pass — for ever, at the price of
    re-embedding it each time.
    """
    service = _Service([_file("f1"), _file("f2")])
    subject = _Subject(service)
    seen: dict[str, int] = {}
    subject._poll(seen)

    service.files = [_file("f1")]
    service.listing_truncated = True
    subject._poll(seen)

    assert subject.removed == []
    assert set(seen) == {"f1", "f2"}, "the one past the bound is still considered indexed"


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


def test_a_file_the_listing_lied_about_is_skipped_mid_download():
    """The size in a listing is the service's word, not a fact."""
    service = _Service([_file(size=10)], body=b"x" * 5000)
    subject = _Subject(service, size_limit=1024)

    subject._poll({})

    assert service.fetched == ["f1"]
    assert subject.added == [], "the download stops at the limit rather than being measured after"


def test_the_metadata_handed_to_pathway_is_ours():
    service = _Service([_file()])
    subject = _Subject(service)

    subject._poll({})

    (metadata,) = subject.added
    assert parse_key(metadata["path"])["user_id"] == str(USER)
    assert metadata.keys() == {"path", "modified_at", "seen_at", "web_url", "size"}


# --- what the UI is told -----------------------------------------------------


def test_a_good_pass_reports_what_the_source_holds():
    reporter = _Recorder()
    subject = _Subject(_Service([_file("f1"), _file("f2")]), reporter=reporter)

    subject._poll({})

    assert reporter.reports == [(True, "", 2)]


def test_an_empty_source_reports_zero_rather_than_a_problem():
    """The wrong folder id, a folder with nothing in it, and Notion pages that
    were never shared with the integration all look like this — and all of them
    are things the person who connected it can fix, once they can see it."""
    reporter = _Recorder()
    subject = _Subject(_Service([]), reporter=reporter)

    subject._poll({})

    assert reporter.reports == [(True, "", 0)]


def test_a_source_that_cannot_be_listed_says_so_instead_of_looking_healthy():
    """The failure this whole channel exists for. Before it, a revoked token
    was a log line, and the UI went on showing a source that was fine."""
    service = _Service([_file()])
    reporter = _Recorder()
    subject = _Subject(service, reporter=reporter)
    subject._poll({})

    service.listing_fails = True
    subject._poll({})

    ok, problem, _ = reporter.reports[-1]
    assert ok is False
    assert problem, "a failure with nothing to show the user is the bug being fixed"


def test_a_file_that_will_not_download_does_not_condemn_the_source():
    """One unreadable file is not an outage: the other documents indexed, and
    saying otherwise would send someone to re-issue a working token."""
    service = _Service([_file("f1"), _file("f2")])
    service.fail_on = {"f1"}
    reporter = _Recorder()

    _Subject(service, reporter=reporter)._poll({})

    assert reporter.reports == [(True, "", 2)]


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


def test_every_kind_builds_and_the_tables_concatenate(monkeypatch):
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

    monkeypatch.setenv("GDRIVE_CREDENTIALS_JSON", json.dumps(_service_account_key()))
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
    settings = IndexerSettings(openrouter_api_key="unused")

    tables = build_tables(settings, specs)

    # One per connected source, plus the uploads table that is always there.
    assert len(tables) == len(configs) + 1
    merged = pw.Table.concat_reindex(*(t.select(pw.this.data, pw.this._metadata) for t in tables))
    assert sorted(merged.column_names()) == ["_metadata", "data"]


def _service_account_key() -> dict:
    """A throwaway service account key, generated fresh.

    Google's client parses the private key when the credentials are built, so a
    placeholder string will not do — and the point of the test above is that
    every kind reaches a working connector, which for Drive includes this step.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return {
        "type": "service_account",
        "project_id": "test",
        "private_key_id": "test",
        "private_key": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
        "client_email": "test@test.iam.gserviceaccount.com",
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


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


def test_a_name_with_trailing_space_is_still_a_pdf():
    """The suffix check must see the name the document is actually indexed under."""
    service = _Service([_file(name="report.pdf ")])
    subject = _Subject(service)

    subject._poll({})

    assert service.fetched == ["f1"]
    assert parse_key(subject.added[0]["path"])["filename"] == "report.pdf"


@pytest.mark.parametrize(
    ("typed", "yandex", "dropbox", "onedrive"),
    [
        ("/", "disk:/", "", ""),
        ("  /  ", "disk:/", "", ""),
        ("/Документы/", "disk:/Документы", "/Документы", "Документы"),
        ("Docs", "disk:/Docs", "/Docs", "Docs"),
    ],
)
def test_a_typed_folder_reaches_each_service_the_way_it_spells_it(typed, yandex, dropbox, onedrive):
    """Three services, three spellings of the root, one thing a person types."""
    from rag_indexer.connectors.dropbox import DropboxSource
    from rag_indexer.connectors.onedrive import OneDriveSource
    from rag_indexer.connectors.yandex import YandexSource

    def build(cls, kind, config):
        source = cls(
            ConnectorSpec(
                source_id=SOURCE, user_id=USER, kind=kind, name="s", config=config | {"path": typed}
            )
        )
        path = source._path
        source.close()
        return path

    assert build(YandexSource, Kind.YANDEX, {"token": "t"}) == yandex
    assert (
        build(
            DropboxSource,
            Kind.DROPBOX,
            {"app_key": "a", "app_secret": "b", "refresh_token": "c"},
        )
        == dropbox
    )
    assert (
        build(
            OneDriveSource,
            Kind.ONEDRIVE,
            {"client_id": "a", "client_secret": "b", "refresh_token": "c"},
        )
        == onedrive
    )


# --- pacing ------------------------------------------------------------------


def test_a_failed_pass_is_retried_sooner_than_the_refresh_interval():
    """Until a listing succeeds this source has no documents in the index.

    "Keep the last snapshot" is the right answer to a blip, but on a fresh start
    the last snapshot is empty — and this process restarts whenever anybody adds
    a source. Waiting a full interval after one bad response would leave the
    source dark for that whole interval.
    """
    subject = _Subject(_Service([]))
    subject._refresh_interval = 600

    assert subject._delay(0) == 600, "a pass that worked waits the full interval"
    assert [subject._delay(n) for n in (1, 2, 3)] == [30, 60, 120]
    assert subject._delay(10) == 600, "and backs off no further than the interval"
    # Bounded in the exponent too: a source left connected with a dead token
    # fails every half minute for as long as it stays connected.
    assert subject._delay(1_000_000) == 600


def test_a_poll_reports_whether_it_worked():
    subject = _Subject(_Service([_file()]))

    assert subject._poll({}) is True

    subject._source.listing_fails = True
    assert subject._poll({}) is False


# --- Google Drive ------------------------------------------------------------


class _Drive:
    """Google's client, as far as this connector can tell."""

    def __init__(self, tree: dict[str, list[dict]]):
        self.tree = tree
        self.queries: list[str] = []

    def files(self):
        return self

    def list(self, *, q, **kwargs):
        self.queries.append(q)
        parents = [part.split("'")[1] for part in q.split(" or ")]
        items = [item for parent in parents for item in self.tree.get(parent, [])]
        return _Executed({"files": items})

    def export_media(self, *, fileId, mimeType):
        return _Executed(b"PK\x03\x04 exported")

    def get_media(self, *, fileId):
        return _Executed(b"raw bytes")

    def close(self):
        pass


class _Executed:
    def __init__(self, value):
        self.value = value

    def execute(self, **kwargs):
        return self.value


@pytest.fixture
def drive(monkeypatch):
    from rag_indexer.connectors import gdrive

    def build(tree):
        fake = _Drive(tree)
        monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **k: fake)
        monkeypatch.setattr(
            "google.oauth2.service_account.Credentials.from_service_account_info",
            classmethod(lambda cls, info, **kw: object()),
        )
        source = gdrive.GoogleDriveSource(
            ConnectorSpec(
                source_id=SOURCE,
                user_id=USER,
                kind=Kind.GDRIVE,
                name="drive",
                config={"folder_id": "root"},
            ),
            {"type": "service_account"},
        )
        return source, fake

    return build


def _entry(name, mime="application/pdf", **extra):
    return {"id": f"id-{name}", "name": name, "mimeType": mime, "size": "100"} | extra


def test_a_google_doc_is_named_by_what_it_exports_to(drive):
    """It has no extension of its own, and the parser dispatches on bytes it
    would never receive unless the file is asked for as a .docx."""
    source, _ = drive(
        {"root": [_entry("Импульсивные покупки", "application/vnd.google-apps.document")]}
    )

    (file,) = source.contents().files

    assert file.filename == "Импульсивные покупки.docx"
    assert source.fetch(file, 1 << 20).startswith(b"PK")


def test_a_plain_file_is_downloaded_rather_than_exported(drive):
    source, _ = drive({"root": [_entry("Договор.pdf")]})

    (file,) = source.contents().files

    assert file.filename == "Договор.pdf"
    assert source.fetch(file, 1 << 20) == b"raw bytes"


def test_subfolders_are_walked_and_batched(drive):
    source, fake = drive(
        {
            "root": [_entry("a.pdf"), _entry("sub", "application/vnd.google-apps.folder")],
            "id-sub": [_entry("b.pdf")],
        }
    )

    files = source.contents().files

    assert {file.filename for file in files} == {"a.pdf", "b.pdf"}
    assert len(fake.queries) == 2, "one request per level, not one per folder"


def test_an_unreadable_file_still_reaches_the_loop_that_reports_it(drive):
    """The polling loop is what decides what it can parse, and it says so."""
    source, _ = drive({"root": [_entry("отпуск.mp4", "video/mp4")]})

    (file,) = source.contents().files
    assert file.filename == "отпуск.mp4"


def test_a_folder_tree_past_the_bound_says_it_is_incomplete(drive):
    """Pathway's connector silently swaps strategies here and returns nothing.
    Ours says so, and the loop stops deleting on the strength of it."""
    from rag_indexer.connectors import gdrive

    deep = {"root": [_entry(f"d{i}", "application/vnd.google-apps.folder") for i in range(40)]}
    for i in range(40):
        deep[f"id-d{i}"] = [
            _entry(f"d{i}-{j}", "application/vnd.google-apps.folder") for j in range(40)
        ]
    source, _ = drive(deep)

    listing = source.contents()

    assert not listing.complete
    assert gdrive._MAX_FOLDERS < 40 * 40
