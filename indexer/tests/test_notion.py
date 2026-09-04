"""Turning a Notion page into something the parser can read.

Notion has no files, so this connector is the only one that *makes* a document
rather than downloading one. What it makes is what the model will be asked to
answer from, which is why the shape of it is pinned here rather than left to be
noticed later in a bad answer.

The API is replaced with recorded responses; the rendering is the real thing.
"""

import uuid

import pytest
from rag_shared.connectors import ConnectorSpec, Kind

from rag_indexer.connectors import notion
from rag_indexer.connectors.notion import NotionSource

PAGE = "22222222-2222-2222-2222-222222222222"


def _rich(text):
    return [{"plain_text": text}]


def _block(kind, text, *, children=False, **body):
    return {"id": f"b-{text}", "type": kind, kind: {"rich_text": _rich(text), **body}} | {
        "has_children": children
    }


class _Api:
    """Notion, as far as this connector can tell."""

    def __init__(self, pages, children):
        self.pages = pages
        self.children = children
        self.calls = 0

    def json(self, method, url, **kwargs):
        self.calls += 1
        if url.endswith("/search"):
            return {"results": self.pages, "has_more": False}
        block_id = url.rsplit("/", 2)[-2]
        return {"results": self.children.get(block_id, []), "has_more": False}

    def close(self):
        pass


@pytest.fixture
def source(monkeypatch):
    def build(pages, children):
        api = _Api(pages, children)
        monkeypatch.setattr(notion, "Http", lambda *a, **k: api)
        connector = NotionSource(
            ConnectorSpec(
                source_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                kind=Kind.NOTION,
                name="workspace",
                config={"token": "ntn_x"},
            )
        )
        return connector, api

    return build


def _page(page_id=PAGE, *, title="Заметка", **extra):
    return {
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "last_edited_time": "2026-03-01T10:00:00.000Z",
        "properties": {"Name": {"type": "title", "title": _rich(title)}},
    } | extra


def test_a_page_is_listed_as_a_markdown_document(source):
    connector, _ = source([_page()], {})

    (file,) = connector.list()

    assert file.external_id == PAGE
    # The suffix is load-bearing: it is what tells the parser to read the bytes
    # as text, and a Notion title never has one.
    assert file.filename == "Заметка.md"
    assert file.web_url == f"https://notion.so/{PAGE}"
    assert file.modified_at == 1772359200


def test_a_page_in_the_bin_is_gone(source):
    connector, _ = source([_page(in_trash=True), _page("other", title="Живая")], {})

    assert [file.filename for file in connector.list()] == ["Живая.md"]


def test_a_page_without_a_title_still_has_a_name(source):
    connector, _ = source([_page(properties={})], {})

    (file,) = connector.list()
    assert file.filename == "Без названия.md"


def test_a_database_row_titled_by_a_renamed_property(source):
    """A row's title property is named by the user; only its type identifies it."""
    connector, _ = source(
        [_page(properties={"Тема": {"type": "title", "title": _rich("Строка")}})], {}
    )

    (file,) = connector.list()
    assert file.filename == "Строка.md"


def test_blocks_become_markdown_in_reading_order(source):
    connector, _ = source(
        [_page()],
        {
            PAGE: [
                _block("heading_1", "Раздел"),
                _block("paragraph", "Текст абзаца"),
                _block("bulleted_list_item", "Первый"),
                _block("numbered_list_item", "Второй"),
                _block("to_do", "Сделать", checked=False),
                _block("quote", "Цитата"),
                {"id": "b-code", "type": "code", "code": {"rich_text": _rich("x = 1")}},
                {"id": "b-div", "type": "divider", "divider": {}},
            ]
        },
    )
    (file,) = connector.list()

    assert connector.fetch(file, 1 << 20).decode() == "\n".join(
        [
            "# Заметка",
            "",
            "# Раздел",
            "Текст абзаца",
            "- Первый",
            "1. Второй",
            "- [ ] Сделать",
            "> Цитата",
            "x = 1",
        ]
    )


def test_nested_blocks_are_indented_not_lost(source):
    connector, _ = source(
        [_page()],
        {
            PAGE: [_block("toggle", "Свернуто", children=True)],
            "b-Свернуто": [_block("paragraph", "Внутри")],
        },
    )
    (file,) = connector.list()

    assert connector.fetch(file, 1 << 20).decode().splitlines()[-2:] == ["Свернуто", "  Внутри"]


def test_a_table_row_keeps_its_cells(source):
    connector, _ = source(
        [_page()],
        {
            PAGE: [
                {
                    "id": "r",
                    "type": "table_row",
                    "table_row": {"cells": [_rich("Мск"), _rich("12")]},
                }
            ]
        },
    )
    (file,) = connector.list()

    assert "Мск | 12" in connector.fetch(file, 1 << 20).decode()


def test_a_page_that_never_stops_nesting_stops_anyway(source):
    """A cycle in the block tree, or a tree deeper than we will follow."""
    connector, api = source(
        [_page()], {PAGE: [_block("toggle", "loop", children=True)], "b-loop": None}
    )
    api.children["b-loop"] = [_block("toggle", "loop", children=True)]

    body = connector.fetch(_page_file(connector), 1 << 20)

    assert body.count(b"loop") <= 8, "the depth limit is what keeps one page finite"


def _page_file(connector):
    (file,) = connector.list()
    return file


def test_a_slash_in_a_title_is_not_a_folder(source):
    """The name becomes part of a key, where a slash is a separator."""
    connector, _ = source([_page(title="Продажи/2026")], {})

    (file,) = connector.list()
    assert file.filename == "Продажи-2026.md"
