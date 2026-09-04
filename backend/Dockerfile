# Build context is the repository root: the API depends on ../shared.
# Two stages so uv's download cache and the intermediate virtualenv never reach
# the final image — see indexer/Dockerfile for what that cost when it did.

FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# rag-shared is an editable path dependency resolved as ../shared, so it has to
# sit next to /app at the same relative position it has in the repository.
COPY shared/ /shared/

WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY backend/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /shared /shared
COPY --from=builder /app /app

RUN useradd --system --uid 10001 rag && chown -R rag:rag /app /shared
USER rag

EXPOSE 8000

# Migrations run here rather than in a separate job: one replica, and a schema
# that is always in step with the image that serves it.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'"]
