"""The .env.example must describe settings that exist.

A template outlives the code that gave it meaning: a renamed field leaves a key
nobody reads, and someone sets it, restarts, and wonders why nothing changed.
"""

import re
from pathlib import Path

from app.config import Settings

TEMPLATE = Path(__file__).resolve().parents[1] / ".env.example"


def test_every_key_in_the_template_is_a_real_setting():
    keys = {
        match.group(1).lower()
        for match in re.finditer(r"^#?\s*([A-Z][A-Z0-9_]*)=", TEMPLATE.read_text("utf-8"), re.M)
    }

    assert keys, "the template lost its keys"
    assert keys <= set(Settings.model_fields), sorted(keys - set(Settings.model_fields))


def test_the_settings_you_cannot_start_without_are_in_it():
    required = {name for name, field in Settings.model_fields.items() if field.is_required()}
    template = TEMPLATE.read_text("utf-8")

    for name in required:
        assert re.search(rf"^{name.upper()}=", template, re.M), f"{name} missing from the template"
