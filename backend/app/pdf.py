"""Whether a PDF has any text in it to find.

The failure this exists for is silent. A scan is a PDF full of pictures of
words: it uploads, it indexes, it appears in the list as ready — and it is,
truthfully, indexed, because the file was read and produced nothing. Search
then never returns it, and the only visible fact is a document that does not
answer questions about its own contents.

The indexer cannot say so. Its parser is handed the bytes and nothing else — no
document id, no path — so it has no way to name the file it could not read. The
one place that knows both the bytes and which document they are is the upload
request, which is also the moment the person is standing there able to do
something about it.

This is a warning, never a gate: the file is stored and indexed either way. A
PDF whose first pages happen to be blank is worth a wrong warning far more than
a scan is worth silence.
"""

import logging

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

#: Pages sampled before giving up on finding text. A scan is a scan on its
#: first page, but a title page or a cover sheet with no text under it is
#: ordinary, so one page is not enough to judge on. Five is also what keeps
#: this bounded: the alternative is rendering every page of a 64 MB upload
#: while the person waits.
PAGES_SAMPLED = 5


def has_text_layer(contents: bytes) -> bool | None:
    """True if any sampled page yields text, False if none do, None if unknown.

    None covers a PDF that cannot be opened at all — encrypted, truncated, not
    really a PDF. Saying nothing is right there: "no text" would be a claim
    about the contents of a file that was never read.

    The same library the indexer parses with, called the same way, so the
    warning cannot disagree with what actually reached the index.
    """
    try:
        document = pdfium.PdfDocument(contents)
    except Exception:
        logger.info("a PDF upload could not be opened to check for text")
        return None

    try:
        pages = len(document)
        for number in range(min(pages, PAGES_SAMPLED)):
            # Each of these wraps a C++ handle, closed explicitly and on the
            # failure path — the same discipline the indexer's parser keeps,
            # and for the same reason: a malformed page otherwise holds native
            # memory for as long as the collector feels like.
            page = document[number]
            try:
                textpage = page.get_textpage()
                try:
                    if textpage.get_text_bounded().strip():
                        return True
                finally:
                    textpage.close()
            finally:
                page.close()
        return False if pages else None
    except Exception:
        logger.info("a PDF upload could not be read to check for text")
        return None
    finally:
        document.close()
