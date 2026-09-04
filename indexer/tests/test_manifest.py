"""The file the API and the indexer agree on.

Its job is to survive being written by a process that may be a version ahead,
and to carry credentials without ever writing one down. Both are things you only
find out about in production unless they are pinned here.
"""

import json
import uuid

import pytest
from rag_shared.connectors import ConnectorSpec, Kind, dump_manifest, load_manifest
from rag_shared.crypto import InvalidToken, Sealer, generate_key

KEY = generate_key()
SEALER = Sealer(KEY)
USER = uuid.UUID("11111111-1111-1111-1111-111111111111")

CONFIGS = {
    Kind.GDRIVE: {"folder_id": "1AbCdEfGh"},
    Kind.NOTION: {"token": "ntn_topsecret"},
    Kind.YANDEX: {"token": "y0_topsecret", "path": "/Документы"},
    Kind.DROPBOX: {
        "app_key": "ak",
        "app_secret": "as_topsecret",
        "refresh_token": "rt_topsecret",
        "path": "/Docs",
    },
    Kind.ONEDRIVE: {
        "client_id": "ci",
        "client_secret": "cs_topsecret",
        "refresh_token": "rt_topsecret",
        "path": "Docs",
    },
}


def _spec(kind: Kind, **overrides) -> ConnectorSpec:
    return ConnectorSpec(
        source_id=overrides.pop("source_id", uuid.uuid4()),
        user_id=USER,
        kind=kind,
        name=overrides.pop("name", kind.value),
        config=overrides.pop("config", dict(CONFIGS[kind])),
    )


def test_every_kind_survives_the_round_trip():
    specs = [_spec(kind) for kind in CONFIGS]

    back = {spec.kind: spec for spec in load_manifest(dump_manifest(specs, SEALER), SEALER)}

    assert back.keys() == CONFIGS.keys()
    for kind, config in CONFIGS.items():
        assert back[kind].config == config
        assert back[kind].missing_fields() == ()


def test_no_credential_is_written_down():
    raw = dump_manifest([_spec(kind) for kind in CONFIGS], SEALER)

    assert b"topsecret" not in raw, "a credential reached the bucket in the clear"
    assert b"1AbCdEfGh" not in raw, "even a folder id is somebody's private information"


def test_the_bytes_are_stable_for_an_unchanged_list():
    """The indexer restarts on a changed ETag, so key order must not churn."""
    specs = [_spec(kind) for kind in CONFIGS]

    first = json.loads(dump_manifest(specs, SEALER))
    second = json.loads(dump_manifest(list(reversed(specs)), SEALER))

    assert [entry["source_id"] for entry in first["sources"]] == [
        entry["source_id"] for entry in second["sources"]
    ]


def test_a_wrong_key_opens_nothing():
    raw = dump_manifest([_spec(Kind.NOTION)], SEALER)

    assert load_manifest(raw, Sealer(generate_key())) == []


def test_a_tampered_seal_is_refused():
    sealed = SEALER.seal({"token": "ntn_x"})
    with pytest.raises(InvalidToken):
        SEALER.unseal(sealed[:-4] + "AAAA")


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda e: e | {"kind": "sharepoint"}, id="a kind this build does not have"),
        pytest.param(lambda e: e | {"source_id": "not-a-uuid"}, id="a malformed id"),
        pytest.param(lambda e: {k: v for k, v in e.items() if k != "name"}, id="a missing key"),
        pytest.param(lambda e: e | {"sealed_config": "gAAAAAnonsense"}, id="an unopenable seal"),
    ],
)
def test_one_bad_entry_costs_only_that_source(damage):
    """The alternative is a user's typo taking down everyone else's index."""
    good, bad = _spec(Kind.NOTION), _spec(Kind.YANDEX)
    payload = json.loads(dump_manifest([good, bad], SEALER))
    payload["sources"] = [
        damage(entry) if entry["source_id"] == str(bad.source_id) else entry
        for entry in payload["sources"]
    ]

    kept = load_manifest(json.dumps(payload).encode(), SEALER)

    assert [spec.source_id for spec in kept] == [good.source_id]


def test_a_config_missing_a_required_field_is_refused():
    spec = _spec(Kind.YANDEX, config={"token": "y0_x"})  # no path

    assert spec.missing_fields() == ("path",)
    assert load_manifest(dump_manifest([spec], SEALER), SEALER) == []


@pytest.mark.parametrize(
    "raw",
    [b"", b"not json", b"[]", b'{"version": 1}', b'{"sources": "nope"}'],
)
def test_an_unreadable_manifest_means_uploads_only(raw):
    """Never an exception: uploads have nothing to do with a broken manifest."""
    assert load_manifest(raw, SEALER) == []


def test_the_key_itself_is_checked_before_it_is_used():
    with pytest.raises(ValueError, match="SECRETS_KEY"):
        Sealer("not-a-fernet-key")
