# Build context is the repository root: the indexer depends on ../shared.
#
# Two stages on purpose. A single-stage build left uv's download cache and a
# second copy of the virtualenv in the layers — 8.19 GB for an image whose
# virtualenv is 1.8 GB. The cache mount keeps downloads out of the layers, and
# only the finished /app and /shared cross into the final image.

FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY shared/ /shared/

WORKDIR /app
COPY indexer/pyproject.toml indexer/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY indexer/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    NLTK_DATA=/opt/nltk_data \
    PATH="/app/.venv/bin:$PATH"

# libmagic is how unstructured sniffs file types.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# The user exists before the copies so ownership can be set by COPY itself.
# A `chown -R` afterwards rewrites every file's metadata, and Docker stores a
# full copy of everything it touched in a new layer — that alone added 1.85 GB.
RUN useradd --system --uid 10001 rag \
    && mkdir -p /data/cache \
    && chown -R rag:rag /data

COPY --from=builder --chown=rag:rag /shared /shared
COPY --from=builder --chown=rag:rag /app /app

# unstructured fetches these on first use. Baking them in keeps the first
# upload from depending on nltk's servers being up.
RUN python -m nltk.downloader -d $NLTK_DATA punkt_tab averaged_perceptron_tagger_eng

USER rag

EXPOSE 8666

CMD ["python", "-m", "rag_indexer.pipeline"]
