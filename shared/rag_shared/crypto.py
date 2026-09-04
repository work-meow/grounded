"""Sealing third-party credentials at rest.

A connected source carries somebody else's credential — a Notion integration
token, a Dropbox refresh token, a Yandex OAuth token. Those are worth more than
anything else this deployment stores, because they reach *outside* it: a leaked
chat history embarrasses, a leaked refresh token hands over the user's Drive.

They come to rest in two places — the ``sources`` table and the connector
manifest in the bucket — and in both they are sealed here rather than written
down. One key, ``SECRETS_KEY``, is shared by the API (which seals) and the
indexer (which opens); nothing else in the system needs it.

Fernet rather than anything hand-rolled: AES-128-CBC with an HMAC-SHA256 tag
over the ciphertext, one reviewed primitive with no knobs left to get wrong.
"""

import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

__all__ = ["InvalidToken", "Sealer", "generate_key"]


def generate_key() -> str:
    """A fresh SECRETS_KEY, ready to paste into .env."""
    return Fernet.generate_key().decode()


class Sealer:
    """Seals and opens the configuration of one connected source.

    Sealing the whole config object, not the secret fields inside it, is
    deliberate: there is then no list of "which keys are sensitive" to forget to
    update when a connector grows a field.
    """

    __slots__ = ("_fernet",)

    def __init__(self, key: str) -> None:
        try:
            self._fernet = Fernet(key.encode())
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "SECRETS_KEY must be a url-safe base64 encoded 32-byte key. "
                "Generate one with: python -c "
                "'from rag_shared.crypto import generate_key; print(generate_key())'"
            ) from exc

    def seal(self, config: dict[str, Any]) -> str:
        return self._fernet.encrypt(json.dumps(config, separators=(",", ":")).encode()).decode()

    def unseal(self, sealed: str) -> dict[str, Any]:
        """The config back. Raises InvalidToken on a wrong key or a tampered blob."""
        try:
            value = json.loads(self._fernet.decrypt(sealed.encode()))
        except (ValueError, TypeError) as exc:  # not JSON, or not text
            raise InvalidToken("sealed config is not JSON") from exc
        if not isinstance(value, dict):
            raise InvalidToken("sealed config is not an object")
        return value
