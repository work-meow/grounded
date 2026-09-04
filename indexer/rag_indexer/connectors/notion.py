"""Notion pages as documents.

Notion has no files, only a tree of blocks, so a "document" here is one page
rendered to Markdown — headings, lists, quotes, code and table rows, in the order
they appear. That is what the reader sees, and Markdown is a format the parser
already reads, so nothing further downstream has to know Notion exists.

The integration token decides the scope: a Notion integration sees only the
pages a person has explicitly shared with it, so connecting a workspace cannot
pull in more than was offered. ``/search`` returns exactly that set.
"""

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from rag_shared.connectors import ConnectorSpec

from rag_indexer.connectors.http import Http, as_timestamp
from rag_indexer.connectors.remote import RemoteFile

logger = logging.getLogger(__name__)

_API = "https://api.notion.com/v1"
# Pinned. Notion routes on this header, so an unpinned version would change the
# response shape underneath us without a deploy.
_VERSION = "2022-06-28"
_PAGE_SIZE = 100

#: How deep a page's block tree is followed. Toggles inside toggles inside
#: columns get deep quickly, and each level costs one request per parent.
_MAX_DEPTH = 6
#: And a ceiling on the requests one page may cost, for a tree that is wide
#: rather than deep. A page this large is a database in disguise.
_MAX_REQUESTS_PER_PAGE = 200

#: Block type to the Markdown that introduces it. Types absent from this map
#: still contribute their text, just without a marker.
_PREFIX = {
    "heading_1": "# ",
    "heading_2": "## ",
    "heading_3": "### ",
    "bulleted_list_item": "- ",
    "numbered_list_item": "1. ",
    "to_do": "- [ ] ",
    "quote": "> ",
    "callout": "> ",
}


@dataclass(slots=True)
class _Budget:
    """How many more requests one page is allowed to cost."""

    left: int

    def spend(self) -> bool:
        self.left -= 1
        return self.left >= 0


class NotionSource:
    """Lists and renders the pages one integration token can see."""

    def __init__(self, spec: ConnectorSpec) -> None:
        self._label = f"notion source {spec.source_id}"
        self._http = Http(
            self._label,
            headers={
                "Authorization": f"Bearer {spec.config['token']}",
                "Notion-Version": _VERSION,
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._http.close()

    def list(self) -> Iterable[RemoteFile]:
        files = []
        for page in self._search():
            # Archived pages are in the bin, not deleted; treating them as gone
            # is what a person means by having moved a page to the bin.
            if page.get("archived") or page.get("in_trash"):
                continue
            # Slashes are ordinary in a Notion title and would otherwise be
            # read as path separators when the name is cleaned, leaving
            # "Продажи/2026" showing up in citations as "2026".
            title = (_title(page) or "Без названия").replace("/", "-")
            files.append(
                RemoteFile(
                    external_id=page["id"],
                    # The suffix is what tells the parser to read it as text;
                    # Notion titles have none.
                    filename=f"{title}.md",
                    modified_at=as_timestamp(page.get("last_edited_time")),
                    web_url=page.get("url"),
                )
            )
        return files

    def fetch(self, file: RemoteFile) -> bytes:
        budget = _Budget(_MAX_REQUESTS_PER_PAGE)
        lines = list(self._render(file.external_id, depth=0, budget=budget))
        # Negative, not zero: the counter only goes below zero when a request
        # was actually refused. Exactly spending the budget is not truncation.
        if budget.left < 0:
            logger.warning("%s: %r is too large to read in full", self._label, file.filename)
        # The title is not a block, and it is often the only place the subject
        # of the page is named at all.
        return "\n".join([f"# {file.filename.removesuffix('.md')}", "", *lines]).encode()

    def _search(self) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            body: dict[str, Any] = {
                "filter": {"value": "page", "property": "object"},
                "page_size": _PAGE_SIZE,
            }
            if cursor:
                body["start_cursor"] = cursor
            payload = self._http.json("POST", f"{_API}/search", json=body)
            yield from payload.get("results", [])
            if not payload.get("has_more"):
                return
            cursor = payload.get("next_cursor")
            if not cursor:  # has_more without a cursor would loop forever
                return

    def _render(self, block_id: str, *, depth: int, budget: _Budget) -> Iterator[str]:
        if depth > _MAX_DEPTH:
            return
        for block in self._children(block_id, budget):
            block_type = block.get("type", "")
            body = block.get(block_type) or {}
            if text := _plain_text(body):
                yield _PREFIX.get(block_type, "") + text
            if block.get("has_children"):
                # One level per recursion, which compounds on the way back up:
                # a block three deep comes out with six spaces. Indenting by the
                # parent's own depth instead would leave the first level flush
                # with the text above it.
                for line in self._render(block["id"], depth=depth + 1, budget=budget):
                    yield "  " + line

    def _children(self, block_id: str, budget: _Budget) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while budget.spend():
            params = {"page_size": _PAGE_SIZE} | ({"start_cursor": cursor} if cursor else {})
            payload = self._http.json("GET", f"{_API}/blocks/{block_id}/children", params=params)
            yield from payload.get("results", [])
            if not payload.get("has_more"):
                return
            cursor = payload.get("next_cursor")
            if not cursor:
                return


def _title(page: dict[str, Any]) -> str:
    """The page's title, wherever Notion put it.

    A standalone page has a property literally named "title"; a row in a
    database has one the user named, identifiable only by its type.
    """
    for prop in (page.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return _join(prop.get("title"))
    return ""


def _plain_text(body: dict[str, Any]) -> str:
    """The readable text of one block, whatever shape it keeps it in."""
    if text := _join(body.get("rich_text")):
        return text
    if cells := body.get("cells"):  # table_row: a list of rich_text arrays
        return " | ".join(_join(cell) for cell in cells)
    if title := body.get("title"):  # child_page, child_database
        return title if isinstance(title, str) else _join(title)
    return ""


def _join(rich_text: Any) -> str:
    if not isinstance(rich_text, list):
        return ""
    return "".join(
        span.get("plain_text", "") for span in rich_text if isinstance(span, dict)
    ).strip()
