"""The parser is dispatched on magic bytes and feeds page numbers to citations.

Every format is built here and read back, so a library upgrade that changes the
shape of what it returns fails loudly instead of quietly producing empty chunks.
"""

import io
import zipfile

import pytest
from docx import Document
from fpdf import FPDF
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches

from rag_indexer import parsers
from rag_indexer.parsers import parse_document


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    pdf = FPDF()
    for page in range(1, 4):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, f"Page {page} says notice is {page * 10} days.")
    return bytes(pdf.output())


def test_pdf_yields_one_entry_per_page_with_its_number(pdf_bytes):
    parsed = parse_document(pdf_bytes)

    assert [meta["page_number"] for _, meta in parsed] == [1, 2, 3]
    # The page number must belong to the text it is attached to, or citations
    # would point at the wrong page.
    assert "notice is 20 days" in parsed[1][0]


def test_docx_reads_paragraphs_and_tables():
    document = Document()
    document.add_paragraph("Договор аренды помещения.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Срок уведомления"
    table.rows[0].cells[1].text = "30 дней"
    buffer = io.BytesIO()
    document.save(buffer)

    ((text, meta),) = parse_document(buffer.getvalue())

    assert "Договор аренды" in text
    # Contracts keep their terms in tables at least as often as in prose.
    assert "Срок уведомления | 30 дней" in text
    assert meta == {}


def test_pptx_numbers_slides_like_pages():
    presentation = Presentation()
    for slide_text in ("Первый слайд", "Второй слайд"):
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
        box.text_frame.text = slide_text
    buffer = io.BytesIO()
    presentation.save(buffer)

    parsed = parse_document(buffer.getvalue())

    assert [meta["page_number"] for _, meta in parsed] == [1, 2]
    assert "Второй слайд" in parsed[1][0]


def test_xlsx_flattens_rows_per_sheet():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Условия"
    sheet.append(["Пункт", "Значение"])
    sheet.append(["Уведомление", 30])
    buffer = io.BytesIO()
    workbook.save(buffer)

    ((text, meta),) = parse_document(buffer.getvalue())

    assert "Уведомление | 30" in text
    assert meta["sheet"] == "Условия"


@pytest.mark.parametrize(
    "raw",
    [
        "Обычный текст договора.".encode(),
        "Кириллица в cp1251.".encode("cp1251"),
        b"# Markdown\n\nWith **bold** text.\n",
    ],
)
def test_plain_text_survives_several_encodings(raw):
    ((text, _),) = parse_document(raw)
    assert text.strip()


def test_unreadable_input_returns_nothing_instead_of_raising():
    """One bad upload must not take the whole indexing pipeline down."""
    assert parse_document(b"%PDF-1.4 but truncated and broken") == []
    assert parse_document(b"") == []


def test_zip_that_is_not_an_office_document_is_ignored():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("random.txt", "not office")

    assert parse_document(buffer.getvalue()) == []


def test_a_zip_bomb_is_refused_before_anything_is_unpacked(monkeypatch):
    """A file that claims to unpack to more than the cap is never opened.

    The indexer runs under a memory limit and holds the index in RAM: being
    OOM-killed by one crafted upload would cost a full rebuild. The cap is
    lowered here rather than building a real multi-gigabyte archive — what is
    under test is the refusal, not zlib.
    """
    monkeypatch.setattr(parsers, "_MAX_UNCOMPRESSED_BYTES", 64)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\0" * 4096)

    assert parse_document(buffer.getvalue()) == []


def test_an_enormous_document_is_trimmed_rather_than_indexed_whole(monkeypatch, caplog):
    """A 64 MB text file is inside the API's upload limit and outside this one.

    Left alone it became ~130k embedding calls and 1.7 GB of resident memory on
    the deployment box, with search unavailable throughout.
    """
    monkeypatch.setattr(parsers, "_MAX_TEXT_CHARS", 1000)

    ((text, _),) = parse_document(("абзац. " * 5000).encode())

    assert len(text) == 1000
    assert "over the 1000 budget" in caplog.text


def test_the_budget_cuts_on_a_page_boundary_so_numbers_still_mean_something(monkeypatch):
    monkeypatch.setattr(parsers, "_MAX_TEXT_CHARS", 1200)

    pdf = FPDF()
    for page in range(1, 6):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, f"Page {page}. " + "filler text here. " * 40)
    parsed = parse_document(bytes(pdf.output()))

    pages = [meta["page_number"] for _, meta in parsed]
    assert pages == sorted(pages) and pages[0] == 1
    assert sum(len(text) for text, _ in parsed) <= 1200


def test_a_document_within_the_budget_is_returned_untouched():
    raw = "короткий документ".encode()
    assert parse_document(raw) == [("короткий документ", {})]


def test_the_document_name_reaches_the_text_that_is_indexed():
    """A file is findable by what it says; its name is metadata, and metadata is
    not searched. "Wishlist.md" listing errands never says "wishlist"."""
    from rag_indexer.pipeline import title_heading

    text, _ = title_heading("- [ ] Сделать карту", {"filename": "Wishlist.md"})
    assert text.startswith("# Wishlist\n\n")


def test_a_name_is_not_added_twice():
    """Notion pages already open with their title."""
    from rag_indexer.pipeline import title_heading

    text, _ = title_heading("# Заметка\n\nтекст", {"filename": "Заметка.md"})
    assert text.count("# Заметка") == 1


@pytest.mark.parametrize(
    "metadata", [{}, {"filename": ""}, {"filename": "   "}, {"filename": None}]
)
def test_a_document_with_no_usable_name_is_left_alone(metadata):
    from rag_indexer.pipeline import title_heading

    assert title_heading("текст", metadata) == ("текст", metadata)
