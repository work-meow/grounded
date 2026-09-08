"""Opening a document at the place the answer quoted, not at its beginning.

A citation says «Политика.pdf, стр. 3» and then hands over a link to the whole
file. On a two-page memo that is fine; on a forty-page contract the reader has
just been given the same search problem again, one they came here to avoid.

Both halves of the answer are already at hand — the page number and the
snippet are in the citation — and both formats have a way to say where to
look, so this is a fragment on the end of a url rather than anything new:

* ``#page=3`` is in the PDF specification's open parameters, and Chrome,
  Firefox and Safari all honour it in their built-in viewers.
* ``#:~:text=…`` is a text fragment: the browser finds the words and scrolls
  to them. Chrome and Edge scroll and highlight, Safari scrolls; Firefox does
  neither and simply opens the file, which is what happens today anyway.

Nothing here can make a link worse. An anchor a browser does not understand is
a fragment it ignores, and a fragment never reaches the server: it is not part
of the request, so a presigned url stays valid with one attached.
"""

import re
from urllib.parse import quote

#: Enough words to be unique in a document without being so long that one
#: difference in whitespace loses the match. Text fragments match on rendered
#: text, so a quote spanning a line break in the source still matches — but a
#: quote spanning a heading or a table cell will not.
_WORDS = 8

#: Anything that is not part of a phrase a browser could find. Markers the
#: agent writes ([1]), the ellipsis a truncated snippet ends with, and the
#: separators that would split a phrase into fragments.
_NOISE = re.compile(r"\[\d{1,3}\]|[…\n\r\t]+")


def at_page(url: str, page: int | None) -> str:
    """The url, opening at ``page`` where the viewer can do that."""
    if page is None or page < 1 or "#" in url:
        return url
    return f"{url}#page={page}"


def at_quote(url: str, snippet: str) -> str:
    """The url, opening at the words in ``snippet``.

    The snippet is a fragment of a chunk, cut at 300 characters and often
    mid-sentence, so only its first words are used: a text fragment either
    matches exactly or does nothing, and a long one is far likelier to have
    picked up an ellipsis or a line break than a short one.
    """
    phrase = words(snippet)
    if not phrase or "#" in url:
        return url
    # Both the comma and the hyphen mean something inside a text fragment
    # (a range, and the prefix/suffix syntax), so they are escaped along with
    # everything else `quote` leaves alone by default.
    return f"{url}#:~:text={quote(phrase, safe='')}"


def words(snippet: str) -> str:
    """The first few words of a snippet, fit to be matched in a page."""
    cleaned = " ".join(_NOISE.sub(" ", snippet).split())
    return " ".join(cleaned.split(" ")[:_WORDS]).strip(" .,;:—-")
