"""The .env.example must describe settings that exist. See the API's twin."""

import re
from pathlib import Path

from rag_indexer.config import IndexerSettings

TEMPLATE = Path(__file__).resolve().parents[1] / ".env.example"


def test_every_key_in_the_template_is_a_real_setting():
    keys = {
        match.group(1).lower()
        for match in re.finditer(r"^#?\s*([A-Z][A-Z0-9_]*)=", TEMPLATE.read_text("utf-8"), re.M)
    }
    fields = set(IndexerSettings.model_fields)

    assert keys, "the template lost its keys"
    assert keys <= fields, sorted(keys - fields)


def test_the_settings_you_cannot_start_without_are_in_it():
    required = {n for n, f in IndexerSettings.model_fields.items() if f.is_required()}
    template = TEMPLATE.read_text("utf-8")

    for name in required:
        assert re.search(rf"^{name.upper()}=", template, re.M), f"{name} missing from the template"
