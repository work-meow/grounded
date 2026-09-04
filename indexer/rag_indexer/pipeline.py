"""The live index.

    uploads (S3)  \\
    Google Drive   \\
    Notion          >-  parse  ->  tenant metadata  ->  chunk  ->  embed
    Dropbox        //                                               |
    OneDrive      //                     usearch KNN  +  tantivy BM25
    Яндекс.Диск  //                                \\    /
                                                    RRF        -> /v1/retrieve

Run it with ``uv run python -m rag_indexer.pipeline``. Every input is streaming:
a file that appears, changes or disappears at any of those sources is reflected
in the index without anything being restarted or re-indexed by hand.

The one thing that *does* need a restart is the source list itself, because a
Pathway graph is fixed once built. :mod:`rag_indexer.manifest` watches for that
and restarts the process; :mod:`rag_indexer.connectors` explains the rest.
"""

import logging
import os

import pathway as pw
from pathway.stdlib.indexing import (
    HybridIndexFactory,
    TantivyBM25Factory,
    UsearchKnnFactory,
)
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.servers import DocumentStoreServer
from pathway.xpacks.llm.splitters import RecursiveSplitter
from rag_shared.connectors import ConnectorSpec
from rag_shared.doc_key import tenant_metadata

from rag_indexer import manifest
from rag_indexer.config import IndexerSettings
from rag_indexer.connectors import build_tables
from rag_indexer.parsers import parse_document

logger = logging.getLogger(__name__)


#: Lines Pathway emits on every engine tick, each saying the same thing whether
#: or not anything changed. There are ten ticks a second now (see
#: COMMIT_INTERVAL_MS), so unfiltered they came to a thousand lines a minute —
#: about a hundred megabytes a day into a log capped at thirty, which is a few
#: hours of history and no room for anything worth reading.
#:
#: Note what is *not* here: "N entries have been sent to the engine" for a
#: non-zero N. That line is how you tell a connector re-read one changed
#: document rather than all of them, which is worth the space. Only the zero
#: case goes.
_TICK_NOISE = (
    "pending download tasks",
    "Persisting a chunk of",
    "0 entries (",
    "save metas",
    "Preparing commit",
    "Running garbage collection",
    "Garbage collect",
)


class _DropPollingNoise(logging.Filter):
    """Drop the per-tick chatter, keep everything else.

    Attached to the handler rather than the logger: a filter on a logger is not
    applied to records that propagate up to it from Pathway's own loggers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(noise in message for noise in _TICK_NOISE)


def configure_logging(level: str) -> None:
    """Own the logging config before Pathway does.

    ``logging.basicConfig`` is a no-op once the root logger has handlers, so
    configuring here wins; ``pw.run(default_logging=False)`` keeps Pathway from
    trying at all.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropPollingNoise())


def build_store(settings: IndexerSettings, specs: list[ConnectorSpec]) -> DocumentStore:
    # Must happen before the embedder is constructed: Pathway builds the OpenAI
    # client itself and never forwards a base URL, so this env var is the only
    # way to reach an OpenAI-compatible endpoint. Setting it here rather than in
    # main() means no caller can get the ordering wrong and silently embed
    # against api.openai.com.
    os.environ["OPENAI_BASE_URL"] = settings.openrouter_base_url

    embedder = OpenAIEmbedder(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        # Not the default. Without it, `with_cache=True` and a cache directory
        # buy nothing: that pair enables the UDF cache, and a UDF only uses it
        # if it asks for a strategy — so every restart re-embedded, and paid
        # for, the whole corpus. Measured: two calls per restart for one
        # document, one of them the dimension probe.
        #
        # It matters far more now than it did. A source list that changes
        # restarts this process on purpose, so re-embedding on restart would
        # put the price of the entire index on every connect and disconnect.
        cache_strategy=pw.udfs.DiskCache(),
        # Chunks are capped at 500 tokens by the splitter, decades below the
        # embedder's limit, so the truncation path (and its per-chunk lookup
        # warning for non-OpenAI model names) is dead weight.
        truncation_keep_strategy=None,
    )

    # Vector recall for meaning, BM25 for exact terms — names, numbers, article
    # references — that embeddings routinely miss. HybridIndexFactory fuses the
    # two with reciprocal rank fusion.
    retriever_factory = HybridIndexFactory(
        [
            UsearchKnnFactory(embedder=embedder),
            TantivyBM25Factory(),
        ]
    )

    return DocumentStore(
        # One list, concatenated by DocumentStore: every source shares the same
        # parser, splitter and index, and differs only in where its bytes and
        # its metadata came from.
        docs=build_tables(settings, specs),
        retriever_factory=retriever_factory,
        parser=pw.udf(parse_document),
        # Recursive, not TokenCount: it splits on paragraph and sentence
        # boundaries before falling back to raw length, so a heading is not
        # welded onto the body of the section below it. Measured against
        # TokenCountSplitter, which merged both and left no overlap.
        splitter=RecursiveSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            encoding_name="cl100k_base",
        ),
        doc_post_processors=[tenant_metadata],
    )


def main() -> None:
    settings = IndexerSettings()  # type: ignore[call-arg]
    configure_logging(settings.log_level)

    sources = manifest.load(settings)
    logger.info("indexing uploads and %d connected source(s)", len(sources.specs))

    # Building the store probes the embedder once to learn its dimension, so a
    # bad key or an unreachable endpoint fails loudly here rather than on the
    # first upload.
    store = build_store(settings, sources.specs)
    server = DocumentStoreServer(settings.host, settings.port, store)

    # Started only once the graph is built, so that a source added while the
    # index was still starting restarts a process that is actually running.
    manifest.watch(settings, sources)

    logger.info("indexer listening on http://%s:%s", settings.host, settings.port)
    server.run(
        with_cache=True,
        cache_backend=pw.persistence.Backend.filesystem(settings.cache_dir),
        default_logging=False,
    )


if __name__ == "__main__":
    main()
