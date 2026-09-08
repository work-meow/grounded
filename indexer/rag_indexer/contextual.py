"""A sentence in front of each chunk saying what it is part of.

A chunk from the middle of a leave policy can read, in full, "28 календарных
дней". Nothing in it says leave, or policy, or which company — so it is
findable by meaning, and only if the meaning survived being written down that
tersely. The document's name is already prepended by ``title_heading``, but to
the *first* chunk of each part only: putting it on all of them was measured
and rejected, because a name repeated into every chunk dilutes every embedding
and is paid for on every one.

This is the other way round. Instead of the same name everywhere, each chunk
gets its own line — what section it belongs to, what it is about — written by
a cheap model that has just read the part it came from. Both indexes are built
over the same text, so the line lands in the vector embedding and in BM25 at
once, which is where the technique gets most of its effect.

Three things make it affordable:

* **One call per part, not per chunk.** A part here is a PDF page or a short
  file — one or two chunks — so the document's text is sent once instead of
  once per chunk.
* **Cached on disk by the part's text.** A restart pays nothing, which matters
  more than it sounds: adding a source restarts this process on purpose.
* **Skipped when there is nothing to situate.** A part that split into a single
  chunk already carries the title and is its own context.

And one rule holds throughout: a failure here costs the context, never the
document. Every path returns the chunk unchanged.
"""

import asyncio
import json
import logging

import httpx
import pathway as pw
from pathway.xpacks.llm.splitters import BaseSplitter

from rag_indexer.config import IndexerSettings

logger = logging.getLogger(__name__)

#: Bumped when the instruction below changes. It goes into the cache key, so a
#: new instruction means everything is described again — the whole point, since
#: the old lines were written to a different brief. Without it the key is the
#: part's text alone, and editing the prompt would silently change nothing.
PROMPT_VERSION = 1


class _VersionedCache(pw.udfs.DiskCache):
    """The on-disk cache, with the instruction's version in the key.

    Through ``make_key`` rather than through the cache's ``name``: a named
    cache must be unique for the life of the process, so naming it after the
    version made the splitter a class that could not be constructed twice —
    harmless in production, where it is built once, and a landmine everywhere
    else. A test found it immediately.
    """

    def make_key(self, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
        return f"v{PROMPT_VERSION}:{super().make_key(args, kwargs)}"


_INSTRUCTION = (
    "Ты помогаешь поиску. Для каждого фрагмента напиши одну короткую фразу: "
    "частью чего он является и о чём. Называй конкретное — раздел, документ, "
    "предмет, — чтобы фрагмент нашёлся по словам, которых в нём самом нет. "
    "Пиши на языке документа, без вступлений и без пересказа самого фрагмента. "
    'Верни JSON вида {"contexts": ["фраза", ...]} — ровно по одной фразе на '
    "каждый фрагмент, в том же порядке. Ничего кроме JSON."
)

#: One client per event loop. Pathway runs async UDFs on a loop of its own, and
#: an httpx client binds to the loop of its first request — so a module-level
#: one would break if that loop were ever replaced, and a per-call one would
#: open a connection pool per document.
_clients: dict[int, httpx.AsyncClient] = {}


def _client(timeout: float) -> httpx.AsyncClient:
    key = id(asyncio.get_running_loop())
    if key not in _clients:
        _clients[key] = httpx.AsyncClient(timeout=timeout)
    return _clients[key]


class ContextualSplitter(pw.UDF):
    """A splitter that also says what each chunk is part of.

    Wraps the real splitter rather than replacing it: where the chunk
    boundaries fall is a decision that was already made and measured, and this
    only changes what is written in front of them.

    It has to be the splitter, because there is nowhere else. Pathway's
    DocumentStore runs parser → post-processors → splitter → index, and the
    post-processors see whole parts, before there are chunks to describe.
    """

    def __init__(self, inner: BaseSplitter, settings: IndexerSettings) -> None:
        super().__init__(
            executor=pw.udfs.async_executor(
                capacity=settings.context_concurrency,
                retry_strategy=pw.udfs.ExponentialBackoffRetryStrategy(max_retries=2),
            ),
            # Keyed on the part's text and on the instruction's version, so a
            # restart pays nothing and an edited prompt pays for everything.
            cache_strategy=_VersionedCache(),
        )
        self._inner = inner
        self._settings = settings

    async def __wrapped__(self, text: str) -> list[tuple[str, dict]]:
        chunks = self._inner.chunk(text)
        if len(chunks) < 2:
            # One chunk is the whole part: it already opens with the document's
            # name and there is nothing to situate it against.
            return chunks
        if len(chunks) > self._settings.context_max_chunks:
            logger.info(
                "not describing a part of %d chunks, over the %d limit",
                len(chunks),
                self._settings.context_max_chunks,
            )
            return chunks

        described = await self._describe(text, [body for body, _ in chunks])
        return [
            (f"{context}\n\n{body}" if context else body, metadata)
            for (body, metadata), context in zip(chunks, described, strict=True)
        ]

    async def _describe(self, part: str, bodies: list[str]) -> list[str]:
        """One line per chunk, in order. Empty strings where there is none."""
        described: list[str] = []
        batch = self._settings.context_batch
        for start in range(0, len(bodies), batch):
            group = bodies[start : start + batch]
            described.extend(await self._ask(part, group))
        return described

    async def _ask(self, part: str, bodies: list[str]) -> list[str]:
        """Describe one batch. Never raises: a batch without context is fine."""
        listed = "\n\n".join(
            f"[{index}] {body[: self._settings.context_document_chars]}"
            for index, body in enumerate(bodies)
        )
        asked = (
            f"Документ:\n{part[: self._settings.context_document_chars]}\n\nФрагменты:\n{listed}"
        )
        try:
            response = await _client(self._settings.context_timeout_s).post(
                f"{self._settings.openrouter_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._settings.openrouter_api_key}"},
                json={
                    "model": self._settings.context_model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": _INSTRUCTION},
                        {"role": "user", "content": asked},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
        except Exception:
            logger.exception("could not describe %d chunks; indexing them as they are", len(bodies))
            return [""] * len(bodies)
        return _contexts(response.json(), len(bodies))


def _contexts(body: object, wanted: int) -> list[str]:
    """The model's lines, or nothing at all.

    All or nothing per batch on purpose. A short list would otherwise be padded
    and the lines would slide onto the wrong chunks — a fragment described as
    something it is not is worse than one described as nothing, because it
    becomes findable under the wrong words.
    """
    try:
        content = body["choices"][0]["message"]["content"]  # type: ignore[index]
        listed = json.loads(content)["contexts"]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("a chunk description came back in a shape this build does not know")
        return [""] * wanted
    if not isinstance(listed, list) or len(listed) != wanted:
        logger.warning("asked for %d chunk descriptions and got %s", wanted, type(listed).__name__)
        return [""] * wanted
    return [" ".join(str(line).split()) if isinstance(line, str) else "" for line in listed]
