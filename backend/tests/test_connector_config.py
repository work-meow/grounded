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
from rag_shared.connectors import MANIFEST_KEY, REQUIRED_FIELDS, Kind, load_manifest
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
