"""One metadata shape, whichever service a document came from.

Everything downstream of the index reads exactly four things off a document:

* ``path`` — the key layout, which is what carries ``user_id``/``source_id``/
  ``document_id`` into every chunk and therefore what makes tenant isolation
  work at all;
* ``modified_at`` and ``seen_at`` — unix seconds, required by Pathway's
  ``DocumentStore`` for its statistics table;
* ``web_url`` — where a citation opens the original, for documents that live
  somewhere else. Uploads leave it unset and get a presigned link instead;
* ``size`` — bytes, when the service says. Only so the file list can show one.

Only the S3 connector produces that shape for free, because the API wrote those
keys itself. Google Drive reports ``id``/``name``/``modifiedTime``; the polling
connectors report whatever their service does. So every other connector ends
here. A connector that skipped it would index documents matching no tenant
filter — invisible rather than leaked, but invisible is still broken.
"""

import time
import unicodedata
from collections.abc import Callable
from pathlib import PurePosixPath
from uuid import UUID

import pathway as pw
from rag_shared.doc_key import build_key, document_id_for

#: Shown when a remote file has no usable name. Never empty: the key layout
#: matches a filename with ``.+``, so an empty one would fail to parse and the
#: document would silently satisfy no filter.
FALLBACK_NAME = "документ"

#: How often an input connector commits what it has read.
#:
#: Not a throughput knob — a latency one. Pathway answers a query only once the
#: dataflow has advanced past it, and the frontier advances no faster than the
#: slowest committing input. At Pathway's default of 1500 ms every search and
#: every file listing waited for the next tick: measured on the deployment,
#: /v1/retrieve took 1.1-1.9 s and /v1/inputs a flat 1.48 s, almost all of it
#: waiting. At 100 ms the same calls take 0.09-0.10 s.
#:
#: The cost is that the engine ticks fifteen times more often: CPU on an idle
#: index went from 2.5% to 7%, and the persistence log from 79 lines a minute to
#: 938 — which is why pipeline.py filters that line out. An agent turn makes two
#: or three searches, so this is four seconds off every answer.
COMMIT_INTERVAL_MS = 100


def describe(
    *,
    user_id: UUID,
    source_id: UUID,
    external_id: str,
    filename: str,
    modified_at: float,
    web_url: str | None = None,
    size: int | None = None,
) -> dict:
    """The metadata for one remote document, in our shape.

    ``external_id`` is whatever the service calls the file and only has to be
    stable there; :func:`document_id_for` turns it into an id stable here.
    """
    document_id = document_id_for(source_id, external_id)
    return {
        "path": build_key(user_id, source_id, document_id, clean_name(filename)),
        "modified_at": int(modified_at),
        "seen_at": int(time.time()),
        "web_url": web_url,
        "size": size,
    }


def clean_name(filename: str) -> str:
    """A remote name reduced to something safe to put in a key and show to a user.

    Remote services allow names our own upload path does not: directory
    separators, control characters, leading dots, 4 kB of emoji. The name ends
    up inside an S3-style key and inside the citation chip, so it is taken down
    to its last segment with control characters removed and its length bounded.
    """
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    return name[:200] or FALLBACK_NAME


def conform(table: pw.Table, description: Callable[[dict], dict]) -> pw.Table:
    """Replace a connector's own metadata with ours.

    Not a merge: the connector's fields are dropped. They are the service's
    vocabulary, they collide across services, and none of them is read anywhere
    downstream — keeping them would only make it possible to accidentally depend
    on one.
    """

    @pw.udf
    def rebrand(metadata: pw.Json) -> dict:
        return description(metadata.as_dict())

    return table.select(data=pw.this.data, _metadata=rebrand(pw.this._metadata))
