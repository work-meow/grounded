"""Opening a document where it was quoted, not at its first line.

A citation says «Политика.pdf, стр. 3» and then used to hand over the whole
file — on a forty-page contract, the same search problem the reader came here
to avoid.

What is worth testing is that a link can never come out worse than it went
in. The url is presigned and its signature covers the path and the query, so
an anchor has to be a fragment and nothing else; a fragment is not sent to the
server, which is what makes this safe at all. And a quote that would not match
must produce no anchor rather than a broken one, because a text fragment that
misses does not fail visibly — the browser just opens the file, and nobody
learns that the feature stopped working.
"""

import pytest

from app import anchor
from app.routers.sources import _anchored

SIGNED = (
    "https://bucket.example.com/users/u/sources/s/d/Политика.pdf"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeef"
)


def test_a_pdf_opens_at_its_page():
    assert anchor.at_page(SIGNED, 3) == f"{SIGNED}#page=3"


@pytest.mark.parametrize("page", [None, 0, -1])
def test_a_page_that_is_not_one_adds_nothing(page):
    assert anchor.at_page(SIGNED, page) == SIGNED


def test_the_signature_is_untouched():
    """The query string is signed; a fragment is not part of the request at
    all, which is the only reason this can be bolted onto a presigned url."""
    linked = anchor.at_page(SIGNED, 7)

    before, _, fragment = linked.partition("#")
    assert before == SIGNED
    assert fragment == "page=7"


def test_a_quote_becomes_a_text_fragment():
    linked = anchor.at_quote("https://x/notes.md", "28 календарных дней отпуска")

    assert linked.startswith("https://x/notes.md#:~:text=")
    assert "%D0%BA%D0%B0%D0%BB%D0%B5%D0%BD%D0%B4%D0%B0%D1%80%D0%BD%D1%8B%D1%85" in linked


def test_the_characters_that_mean_something_in_a_fragment_are_escaped():
    """A comma is a range and a hyphen is the prefix syntax inside a text
    fragment; left raw they would split the phrase into pieces that match
    nothing."""
    linked = anchor.at_quote("https://x/notes.md", "срок — две недели, не позже")

    body = linked.split("#:~:text=")[1]
    assert "," not in body and "-" not in body and " " not in body


def test_only_the_first_words_are_matched():
    """A snippet is a cut of a chunk, often mid-sentence. A text fragment
    either matches exactly or does nothing, and every extra word is another
    chance to have picked up a line break."""
    phrase = anchor.words("один два три четыре пять шесть семь восемь девять десять")

    assert phrase == "один два три четыре пять шесть семь восемь"


def test_the_agents_own_markers_are_not_part_of_the_phrase():
    assert "[1]" not in anchor.words("Отпуск 28 дней [1] по политике")


def test_a_snippet_that_is_all_noise_produces_no_anchor():
    """No anchor at all rather than one that cannot match: a text fragment
    that misses opens the file silently, so nobody would notice."""
    for snippet in ("", "   ", "…", "[1]", "\n\t"):
        assert anchor.at_quote("https://x/notes.md", snippet) == "https://x/notes.md"


def test_a_url_that_already_has_a_fragment_is_left_alone():
    """Whatever is there was put there by whoever owns that url — for a
    connected source that is their page and their idea of where things are."""
    already = "https://notion.so/page#block-42"

    assert anchor.at_page(already, 3) == already
    assert anchor.at_quote(already, "какие-то слова") == already


# --- which format gets which kind of anchor ----------------------------------


def test_a_page_number_goes_only_to_the_pdf():
    assert _anchored(SIGNED, "application/pdf", 3, "слова") == f"{SIGNED}#page=3"


@pytest.mark.parametrize("mime", ["text/markdown", "text/plain"])
def test_text_gets_the_quote(mime):
    assert _anchored("https://x/n.md", mime, None, "28 календарных дней").endswith(
        "#:~:text=28%20%D0%BA%D0%B0%D0%BB%D0%B5%D0%BD%D0%B4%D0%B0%D1%80%D0%BD%D1%8B%D1%85%20%D0%B4%D0%BD%D0%B5%D0%B9"
    )


@pytest.mark.parametrize(
    "mime",
    [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        None,
    ],
)
def test_a_format_the_browser_hands_to_another_program_gets_neither(mime):
    """Word and Excel files are downloaded and opened by an application that
    never saw the url, so a fragment is a promise nothing can keep."""
    assert _anchored(SIGNED, mime, 3, "слова") == SIGNED
