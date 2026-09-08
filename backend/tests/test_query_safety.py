"""Anything typed reaches a query parser that panics instead of complaining.

The BM25 half of the index is Tantivy, and Tantivy's parser owns
`+ - ! ( ) { } [ ] ^ " ~ * ? : \\ / && ||`. A string it cannot parse does not
come back as an error: it **panics the engine thread**, and that takes the
whole indexer process down — for every user, not just the one who typed it.

Watched happen in production. A fragment of a markdown note sent back as a
query crashed the indexer on every call, and the restarts appeared elsewhere
as intermittent connection failures and "индекс перестраивается" from
endpoints that had nothing to do with it. It also means the search box was a
way to do the same thing, which nobody had noticed.

So this is a trust boundary, and these are the tests for it.
"""

import pytest

from app.retriever import searchable
from app.routers.search import _as_query

CRASHED = (
    "# Памятка по линуксу --- #### 1 - Настройка разрешения GRUB2 "
    "Ввести команду в терминале: ``` sudo nano /etc/default/grub ``` "
    "Изменить поле ``` #GRUB_GFXMODE=1920x1080 ```"
)

#: Every character the parser treats as syntax. None may survive.
SYNTAX = set('+-!(){}[]^"~*?:\\/&|')


def test_the_text_that_crashed_the_indexer_carries_no_syntax():
    assert not SYNTAX & set(searchable(CRASHED))


@pytest.mark.parametrize("query", ['"', "+", "-", ":", "~", "^", "*", "?", "(", ")", "\\", "//"])
def test_a_query_of_nothing_but_syntax_becomes_nothing(query):
    """Empty is a fine query — it returns the index's k best for nothing in
    particular. A crash is not."""
    assert searchable(query) == ""


@pytest.mark.parametrize(
    ("asked", "wanted"),
    [
        ("сколько дней отпуска?", "сколько дней отпуска"),
        ("что в вишлисте", "что в вишлисте"),
        ('кто сказал "нет"', "кто сказал нет"),
        ("ставка ЦБ 14,00%", "ставка ЦБ 14,00%"),
        ("файл config.yaml", "файл config.yaml"),
        ("   лишние   пробелы  ", "лишние пробелы"),
    ],
)
def test_a_real_question_survives_intact(asked, wanted):
    """The question mark at the end of every question, the comma in a number,
    the dot in a filename: none of those carry the meaning of a search, and
    none of them may be lost either."""
    assert searchable(asked) == wanted


def test_a_fragment_used_as_a_query_is_also_shortened():
    """BM25 scores a long query by everything in it, and a chunk is five
    hundred tokens of prose. The opening lines carry the heading and the
    subject."""
    long = " ".join(f"слово{n}" for n in range(200))

    assert len(_as_query(long).split()) == 40


def test_a_fragment_used_as_a_query_carries_no_syntax_either():
    assert not SYNTAX & set(_as_query(CRASHED))
