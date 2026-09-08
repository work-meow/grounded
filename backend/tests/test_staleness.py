"""How old the answer's source is, carried as far as the chip.

A right answer taken from a document nobody has touched in two years is still
a right answer to an old question. The reader is the one who can tell whether
that matters, so the age travels with the citation — beside the source rather
than inside the sentence, where the model would have to be told to write it
and would sometimes forget.
"""

from app import agent, retriever


def chunk(modified_at: int) -> retriever.Chunk:
    return retriever.Chunk(
        text="28 календарных дней",
        score=0.9,
        document_id="d1",
        filename="Политика.pdf",
        page=3,
        modified_at=modified_at,
    )


def test_the_citation_says_when_the_document_last_changed():
    citations = agent._Citations()
    citations.add(chunk(1_700_000_000))

    assert citations.items[0]["modified_at"] == 1_700_000_000


def test_a_document_with_no_date_says_nothing_rather_than_1970():
    """Zero is what the index puts there when the metadata did not say, and a
    chip reading "обновлён 56 лет назад" would be a confident lie."""
    citations = agent._Citations()
    citations.add(chunk(0))

    assert citations.items[0]["modified_at"] is None


def test_a_page_from_the_web_has_no_date_to_carry():
    from app import websearch

    citations = agent._Citations()
    citations.add_link(websearch.Source(url="https://x/", title="Страница", text="текст"))

    assert citations.items[0]["modified_at"] is None


def test_the_answers_own_sources_keep_the_field():
    """It has to survive `referenced_in`, which is the list the reader sees."""
    citations = agent._Citations()
    number = citations.add(chunk(1_700_000_000))

    [shown] = citations.referenced_in(f"Отпуск 28 дней [{number}].")

    assert shown["modified_at"] == 1_700_000_000


def test_the_trace_does_not_carry_it():
    """`shown` exists to count fragments, not to describe them; every field in
    it is one more thing written on every answer."""
    citations = agent._Citations()
    citations.add(chunk(1_700_000_000))

    assert set(citations.shown()[0]) == {"n", "document_id", "filename", "page"}
