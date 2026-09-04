"""Settings are required fields, and importing `app.db` builds them at import
time. Supplying them here — rather than in the CI job — is what keeps
`uv run pytest` working on a fresh checkout with no .env at all.

No test opens a socket: the database engine is only constructed, never used.
"""

import os

os.environ.setdefault("JWT_SECRET", "test-secret-that-is-at-least-32-characters")
os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5432/rag")
