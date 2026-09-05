"""Calling a scan a scan.

The failure is entirely silent, which is what makes it worth a test. A scanned
PDF uploads, indexes, and shows in the list as ready — truthfully, because the
file was read and produced nothing — and then never appears in a single search
result. Nothing anywhere says why.

The check is a warning and never a gate, so the interesting cases are the ones
where it must keep its mouth shut: a file it could not read, a format that has
no such question, a document whose text starts later than it looked.
"""

import pytest
from fpdf import FPDF

from app import pdf
from app.routers.sources import _text_layer

PDF_MIME = "application/pdf"


def _with_text(pages: int = 2) -> bytes:
    document = FPDF()
    for number in range(1, pages + 1):
        document.add_page()
        document.set_font("Helvetica", size=12)
        document.multi_cell(0, 8, f"Page {number}: notice is 30 days.")
    return bytes(document.output())


def _scanned(pages: int = 3) -> bytes:
    """Pages with marks on them and not one character of text — what a scan is.

    The text elsewhere in this file is Latin because fpdf2's built-in Helvetica
    is Latin-1 only. It makes no difference to what is being tested: the check
    asks whether there is any text at all, not what language it is in.
    """
    document = FPDF()
    for _ in range(pages):
        document.add_page()
        document.set_fill_color(120, 120, 120)
        document.rect(10, 10, 80, 60, style="F")
    return bytes(document.output())


# --- the check ---------------------------------------------------------------


def test_a_pdf_with_text_says_so():
    assert pdf.has_text_layer(_with_text()) is True


def test_a_pdf_that_is_all_pictures_says_so():
    assert pdf.has_text_layer(_scanned()) is False


def test_text_anywhere_in_the_sample_is_enough():
    """A cover sheet with nothing on it is ordinary; judging on page one alone
    would put a scan warning on documents that are perfectly readable."""
    document = FPDF()
    document.add_page()  # blank cover
    document.add_page()
    document.set_font("Helvetica", size=12)
    document.multi_cell(0, 8, "Lease agreement, page two.")

    assert pdf.has_text_layer(bytes(document.output())) is True


def test_text_beyond_the_sampled_pages_is_missed_on_purpose():
    """The bound is what keeps a 64 MB upload from being read page by page
    while somebody waits. The cost is a wrong warning on a document whose first
    five pages are blank — which is a warning, not a refusal: the file is stored
    and indexed exactly the same either way."""
    document = FPDF()
    for _ in range(pdf.PAGES_SAMPLED + 2):
        document.add_page()
    document.set_font("Helvetica", size=12)
    document.multi_cell(0, 8, "Text the check never reached.")

    assert pdf.has_text_layer(bytes(document.output())) is False


@pytest.mark.parametrize(
    "given",
    [b"", b"not a pdf at all", b"%PDF-1.4 truncated right here", b"\x00\x01\x02"],
)
def test_something_that_will_not_open_gets_no_verdict(given):
    """None, not False. "No text" would be a claim about the contents of a file
    that was never read — and it would put a scan warning on a broken upload,
    which is a different problem with a different fix."""
    assert pdf.has_text_layer(given) is None


def test_the_check_never_raises_into_the_upload():
    """It runs in the request path, between reading the body and writing to S3.
    An exception here would turn a warning into a failed upload."""
    for given in (b"", _scanned(), _with_text(), b"%PDF-1.7\n%\xc7\xec\x8f\xa2\n"):
        assert pdf.has_text_layer(given) in (True, False, None)


# --- what the upload does with it --------------------------------------------


async def test_only_pdfs_are_asked_the_question():
    """Every other format we accept is text by construction. Answering for them
    would mean claiming something about a file nothing looked inside."""
    for mime in (
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        assert await _text_layer(mime, _scanned()) is None


async def test_an_uploaded_scan_is_flagged_on_the_way_in():
    assert await _text_layer(PDF_MIME, _scanned()) is False


async def test_an_ordinary_pdf_is_not_flagged():
    assert await _text_layer(PDF_MIME, _with_text()) is True


async def test_a_large_pdf_does_not_block_the_event_loop():
    """It is C code that can run for seconds on a big file, in the process
    serving every other request — so it goes to a thread."""
    import asyncio

    document = FPDF()
    for _ in range(60):
        document.add_page()
        document.set_font("Helvetica", size=12)
        document.multi_cell(0, 8, "A page with words on it. " * 40)
    body = bytes(document.output())

    ticks = 0

    async def counting() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    counter = asyncio.create_task(counting())
    result = await _text_layer(PDF_MIME, body)
    counter.cancel()

    assert result is True
    assert ticks > 0, "the loop kept running while the PDF was being read"
