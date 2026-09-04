"""The live index.

    S3 (streaming)  ->  parse  ->  tenant metadata  ->  chunk  ->  embed
                                                                    |
                                          usearch KNN  +  tantivy BM25
                                                    \\    /
                                                     RRF          -> /v1/retrieve

Run it with ``uv run python -m rag_indexer.pipeline``. Pathway keeps watching the
bucket, so a newly uploaded file becomes searchable without restarting anything,
and a deleted object drops out of the index on its own.
"""

import os
import sys
from pathlib import Path

import pathway as pw
from pathway.stdlib.indexing import (
    HybridIndexFactory,
    TantivyBM25Factory,
    UsearchKnnFactory,
)
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.parsers import UnstructuredParser
from pathway.xpacks.llm.servers import DocumentStoreServer
from pathway.xpacks.llm.splitters import TokenCountSplitter

from rag_indexer.config import IndexerSettings

# ponytail: the S3 key layout is shared with the API, which lives in a separate
# virtualenv (its langchain>=1.0 cannot coexist with Pathway's langchain<0.4).
# A path insert is the cheapest way to keep one definition of the layout.
# Ceiling: breaks if the two are deployed from different trees. Upgrade path:
# publish `shared/` as a tiny package both depend on.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from doc_key import PREFIX, tenant_metadata


def build_store(settings: IndexerSettings) -> DocumentStore:
    # Must happen before the embedder is constructed: Pathway builds the OpenAI
    # client itself and never forwards a base URL, so this env var is the only
    # way to reach an OpenAI-compatible endpoint. Setting it here rather than in
    # main() means no caller can get the ordering wrong and silently embed
    # against api.openai.com.
    os.environ["OPENAI_BASE_URL"] = settings.openrouter_base_url

    files = pw.io.s3.read(
        # Only our own prefix; anything else in the bucket is ignored.
        path=f"{PREFIX}/",
        format="binary",
        mode="streaming",
        with_metadata=True,
        aws_s3_settings=pw.io.s3.AwsS3Settings(
            bucket_name=settings.s3_bucket,
            access_key=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            region=settings.s3_region,
            endpoint=settings.s3_endpoint_url,
            with_path_style=settings.s3_path_style,
        ),
    )

    # "paged" keeps `page_number` on the metadata, which is what lets a citation
    # say "contract.pdf, стр. 14". strategy="fast" parses digital PDFs with
    # pdfminer and never loads a layout model.
    parser = UnstructuredParser(
        chunking_mode="paged",
        partition_kwargs={"strategy": "fast"},
    )

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
        docs=files,
        retriever_factory=retriever_factory,
        parser=parser,
        splitter=TokenCountSplitter(
            min_tokens=settings.chunk_min_tokens,
            max_tokens=settings.chunk_max_tokens,
        ),
        doc_post_processors=[tenant_metadata],
    )


def main() -> None:
    settings = IndexerSettings()  # type: ignore[call-arg]
    # Building the store probes the embedder once to learn its dimension, so a
    # bad key or an unreachable endpoint fails loudly here rather than on the
    # first upload.
    store = build_store(settings)
    server = DocumentStoreServer(settings.host, settings.port, store)
    print(f"indexer listening on http://{settings.host}:{settings.port}", flush=True)
    server.run(
        with_cache=True,
        cache_backend=pw.persistence.Backend.filesystem(settings.cache_dir),
    )


if __name__ == "__main__":
    main()
