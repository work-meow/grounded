"""What happens to a credential between the form and the bucket.

The interesting property is negative — that it does not come back out — which is
the kind of thing that stays true only if something checks. The rest is the
validation the indexer depends on: it re-checks the same fields, but by then a
bad source is a log line nobody reads rather than a message to the person who
typed it.
"""

import json
import uuid

import pytest
from pydantic import ValidationError
from rag_shared.connectors import MANIFEST_KEY, REQUIRED_FIELDS, Kind, fingerprint, load_manifest
from rag_shared.crypto import Sealer, generate_key

from app import connectors, storage
from app.config import Settings
from app.models import Source
from app.routers.connectors import ConnectorIn, ConnectorOut, ConnectorsOut

KEY = generate_key()


def settings(**overrides) -> Settings:
    return Settings(
        **{"jwt_secret": "x" * 40, "openrouter_api_key": "k", "secrets_key": KEY, **overrides}
    )


# --- validation --------------------------------------------------------------


def test_only_the_fields_the_kind_needs_are_kept():
    config = connectors.clean_config(
        Kind.YANDEX, {"token": "  y0_x  ", "path": "/Docs", "extra": "junk"}
    )

    assert config == {"token": "y0_x", "path": "/Docs"}


@pytest.mark.parametrize("missing", ["token", "path"])
def test_a_missing_field_is_named_in_the_error(missing):
    config = {"token": "y0_x", "path": "/Docs"} | {missing: "   "}

    with pytest.raises(ValueError, match=missing):
        connectors.clean_config(Kind.YANDEX, config)


def test_a_config_with_too_many_keys_is_refused_before_it_is_read():
    """clean_config keeps only what the kind needs — but by then the body has
    already been parsed into a dict."""
    with pytest.raises(ValidationError):
        ConnectorIn(
            kind=Kind.NOTION,
            name="x",
            config={f"k{i}": "v" for i in range(1000)},
        )


def test_a_field_long_enough_to_be_storage_is_refused():
    with pytest.raises(ValueError, match="слишком длинное"):
        connectors.clean_config(Kind.NOTION, {"token": "x" * (connectors.MAX_FIELD_LENGTH + 1)})


@pytest.mark.parametrize("kind", list(Kind))
def test_every_kind_has_a_config_that_can_be_filled_in(kind):
    """A kind the form can offer but the API can never accept would be a trap."""
    filled = dict.fromkeys(REQUIRED_FIELDS[kind], "value")
    assert connectors.clean_config(kind, filled) == filled


# --- the credential does not come back ---------------------------------------


def test_no_response_model_can_carry_a_credential():
    for model in (ConnectorOut, ConnectorsOut):
        assert "config" not in model.model_fields
        assert "sealed_config" not in model.model_fields


def test_a_deployment_without_a_key_does_not_pretend_to_have_one():
    assert connectors.sealer(settings()) is not None
    assert connectors.sealer(settings(secrets_key="")) is None
    assert connectors.sealer(settings(secrets_key="not-a-key")) is None


# --- publishing --------------------------------------------------------------


class _Rows:
    """Just enough of an AsyncSession for publish() to read a source list."""

    def __init__(self, rows):
        self._rows = rows
        self.committed = False

    async def commit(self):
        self.committed = True

    async def execute(self, _statement):
        return self

    def scalars(self):
        return self

    def all(self):
        return self._rows


def _source(kind: Kind, config: dict) -> Source:
    return Source(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        kind=kind.value,
        name=kind.value,
        sealed_config=Sealer(KEY).seal(config),
    )


@pytest.fixture
def published(monkeypatch):
    """Capture what publish() would write to the bucket."""
    written = {}

    async def put(_settings, key, body, content_type):
        written["key"], written["body"], written["type"] = key, body, content_type

    monkeypatch.setattr(storage, "put", put)
    return written


async def test_the_manifest_is_what_the_indexer_reads_back(published):
    rows = [_source(Kind.NOTION, {"token": "ntn_x"}), _source(Kind.GDRIVE, {"folder_id": "1Ab"})]

    await connectors.publish(settings(), _Rows(rows))

    assert published["key"] == MANIFEST_KEY
    back = {spec.source_id: spec for spec in load_manifest(published["body"], Sealer(KEY))}
    assert back.keys() == {row.id for row in rows}
    assert back[rows[0].id].config == {"token": "ntn_x"}


async def test_publishing_twice_writes_the_same_bytes(published):
    """A rewrite that changed nothing must not restart the indexer."""
    rows = [_source(Kind.NOTION, {"token": "ntn_x"})]

    await connectors.publish(settings(), _Rows(rows))
    first = published["body"]
    await connectors.publish(settings(), _Rows(rows))

    assert published["body"] == first


async def test_no_credential_reaches_the_bucket_in_the_clear(published):
    await connectors.publish(settings(), _Rows([_source(Kind.NOTION, {"token": "ntn_topsecret"})]))

    assert b"topsecret" not in published["body"]


async def test_a_kind_this_build_does_not_know_is_left_out(published):
    """A row from a newer deployment must not break the manifest for the rest."""
    stale = _source(Kind.NOTION, {"token": "ntn_x"})
    stale.kind = "sharepoint"

    await connectors.publish(settings(), _Rows([stale, _source(Kind.NOTION, {"token": "y"})]))

    assert len(json.loads(published["body"])["sources"]) == 1


# --- the same source, twice ---------------------------------------------------


def test_the_same_place_under_a_different_name_is_the_same_source():
    """The name is what the user calls it; the folder is what it reads."""
    config = {"token": "y0_x", "path": "/Docs"}

    assert fingerprint(Kind.YANDEX, config) == fingerprint(Kind.YANDEX, dict(config))


@pytest.mark.parametrize("spelling", ["/Docs", "Docs", "Docs/", "  /Docs/  "])
def test_a_folder_spelled_differently_is_still_that_folder(spelling):
    """Otherwise the second attempt at the same folder gets in on a typo."""
    canonical = fingerprint(Kind.YANDEX, {"token": "y0_x", "path": "/Docs"})

    assert fingerprint(Kind.YANDEX, {"token": "y0_x", "path": spelling}) == canonical


@pytest.mark.parametrize(
    ("kind", "left", "right"),
    [
        (Kind.YANDEX, {"token": "a", "path": "/A"}, {"token": "a", "path": "/B"}),
        (Kind.YANDEX, {"token": "a", "path": "/A"}, {"token": "b", "path": "/A"}),
        (Kind.GDRIVE, {"folder_id": "1Aa"}, {"folder_id": "1Bb"}),
        (Kind.NOTION, {"token": "ntn_a"}, {"token": "ntn_b"}),
    ],
)
def test_a_different_place_is_a_different_source(kind, left, right):
    assert fingerprint(kind, left) != fingerprint(kind, right)


def test_the_credential_is_not_recoverable_from_the_fingerprint():
    """It sits in a plain column next to the sealed blob. Sealing exists so a
    database dump gives up the source list and not the tokens behind it."""
    token = "ntn_" + "K7pQ2mZx9RtLvB4nWcE6jH1sYdF3aU8gN5oX0iT"

    finger = fingerprint(Kind.NOTION, {"token": token})

    assert token not in finger
    assert set(finger) <= set("0123456789abcdef") and len(finger) == 64


# --- sources connected before any of this existed -----------------------------


def _row(config: dict, *, kind=Kind.NOTION, name="источник", finger=None) -> Source:
    return Source(
        id=uuid.uuid4(),
        user_id=USER,
        kind=kind.value,
        name=name,
        sealed_config=Sealer(KEY).seal(config),
        fingerprint=finger,
    )


USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


async def test_a_source_from_before_the_check_gets_a_fingerprint():
    """Without this, the sources most likely to be added again are exactly the
    ones nothing can be compared against."""
    row = _row({"token": "ntn_a"})

    await connectors.backfill_fingerprints(Sealer(KEY), _Rows([row]), USER)

    assert row.fingerprint == fingerprint(Kind.NOTION, {"token": "ntn_a"})


async def test_a_pair_that_is_already_a_duplicate_is_left_as_it_is():
    """There is no honest way to pick which of the two to keep, and filling in
    both would make the unique index refuse a change neither of them asked for.
    Both stay visible in the list, where the user can delete one."""
    first = _row({"token": "ntn_a"}, name="первый")
    second = _row({"token": "ntn_a"}, name="второй")

    await connectors.backfill_fingerprints(Sealer(KEY), _Rows([first, second]), USER)

    assert first.fingerprint is not None
    assert second.fingerprint is None


async def test_a_config_this_key_cannot_open_is_skipped_rather_than_fatal():
    """A source sealed under a previous SECRETS_KEY. The indexer is skipping it
    for the same reason; adding a different source must still work."""
    row = _row({"token": "ntn_a"})
    row.sealed_config = Sealer(generate_key()).seal({"token": "ntn_a"})

    await connectors.backfill_fingerprints(Sealer(KEY), _Rows([row]), USER)

    assert row.fingerprint is None


async def test_the_backfill_is_kept_even_when_the_add_it_ran_for_is_refused():
    """It usually runs *because* a duplicate is about to be refused. Leaving it
    to the caller's commit would roll it back on exactly that path, and the
    unique index would go on having nothing to enforce."""
    row = _row({"token": "ntn_a"})
    session = _Rows([row])

    await connectors.backfill_fingerprints(Sealer(KEY), session, USER)

    assert session.committed, "the fill has to outlive the request that triggered it"
