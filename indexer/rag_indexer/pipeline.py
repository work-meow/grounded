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


class _DropPollingNoise(logging.Filter):
    """Pathway's S3 connector reports every poll at INFO, roughly twice a second.

    It says the same thing whether or not anything changed, so it buries the
    events that matter. Attached to the handler rather than the logger: a
    filter on a logger is not applied to records that propagate up to it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "pending download tasks" not in record.getMessage()


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
