# Build context is the repository root: the indexer depends on ../shared.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/opt/nltk_data \
    PATH="/app/.venv/bin:$PATH"

# libmagic is how unstructured sniffs file types.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /shared/

WORKDIR /app
COPY indexer/pyproject.toml indexer/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY indexer/ ./
RUN uv sync --frozen --no-dev

# unstructured fetches these on first use. Baking them in keeps the first
# upload from depending on nltk's servers being up.
RUN python -m nltk.downloader -d $NLTK_DATA punkt_tab averaged_perceptron_tagger_eng

RUN useradd --system --uid 10001 rag \
    && mkdir -p /data/cache \
    && chown -R rag:rag /app /shared /data
USER rag

EXPOSE 8666

CMD ["python", "-m", "rag_indexer.pipeline"]
