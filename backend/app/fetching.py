"""Turning a web page into a document.

You can upload a file and you can connect a whole source, but the commonest
thing anybody wants to keep is neither: an article, a changelog, a page of
somebody's documentation. Copying it into a file first is a step people skip,
and then the answer is not in the base.

The page is fetched here, in the API, converted to text and written to the
bucket as markdown — so the indexer needs no new capability at all: it sees an
object appear and indexes it like any upload.

The interesting part is not the fetching, it is that this is the second place
where the *user* names a machine. `verify_public` exists because of the first
(an S3 endpoint typed into a form), and it is not enough on its own here:
a redirect is a second address, chosen by the far end, and checking only the
one somebody typed would let `https://example.com/r?to=http://127.0.0.1:8000`
through. So redirects are followed by hand, one at a time, and every hop is
checked before it is taken.
"""

import logging
import re
from dataclasses import dataclass

import httpx
from rag_shared.net import verify_public

from app import http
from app.config import Settings

logger = logging.getLogger(__name__)

#: A page bigger than this is not an article. Enforced while reading rather
#: than from Content-Length, which the far end can simply lie about.
MAX_BYTES = 5 * 1024 * 1024

#: How many hops to follow. Three is enough for the http→https→canonical chain
#: every site has; more than that is a redirect loop or somebody's tracker.
MAX_HOPS = 3

#: Whole elements whose text is never the article: scripts, styles, navigation
#: furniture. Dropped with their contents, before tags are stripped.
_DROP = re.compile(
    r"<(script|style|noscript|template|svg|nav|header|footer|aside|form)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
#: Block-level ends, so paragraphs survive as paragraphs rather than becoming
#: one run-on line.
_BREAKS = re.compile(r"</(p|div|section|article|li|tr|h[1-6]|blockquote|pre)\s*>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_BLANK = re.compile(r"\n{3,}")

_ENTITIES = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&mdash;": "—",
    "&ndash;": "–",
    "&laquo;": "«",
    "&raquo;": "»",
}


@dataclass(frozen=True, slots=True)
class Page:
    """A fetched page, ready to be stored as a document."""

    url: str
    title: str
    text: str


async def fetch(settings: Settings, url: str) -> Page:
    """Fetch a page and reduce it to text. Raises ValueError with a reason.

    Redirects are followed here rather than by httpx, because each hop is an
    address chosen by the far end and every one of them has to pass the same
    check as the address that was typed.
    """
    current = url.strip()
    for hop in range(MAX_HOPS + 1):
        verify_public(current)
        response = await http.client().get(
            current,
            timeout=settings.fetch_timeout_s,
            follow_redirects=False,
            headers={"User-Agent": settings.fetch_user_agent, "Accept": "text/html,text/*"},
        )
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise ValueError("страница отвечает переадресацией без адреса")
            if hop == MAX_HOPS:
                raise ValueError(f"слишком много переадресаций (больше {MAX_HOPS})")
            # Resolved against the current url, so a relative Location — which
            # is legal and common — becomes an absolute address that can be
            # checked before the next request.
            current = str(httpx.URL(current).join(location))
            continue
        break

    if response.status_code >= 400:
        raise ValueError(f"страница ответила {response.status_code}")
    kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind and not kind.startswith("text/"):
        raise ValueError(f"это не страница, а {kind} — такое загружайте файлом")

    body = await _read(response)
    text = to_text(body)
    if len(text) < 200:
        # A page of navigation and a login wall both look like this, and an
        # empty document in the base is worse than a refusal: it will be
        # indexed, found and cited as if it said something.
        raise ValueError("на странице почти нет текста — возможно, она требует входа")
    return Page(url=current, title=title_of(body) or _host(current), text=text)


async def _read(response: httpx.Response) -> str:
    """The body, capped while reading rather than trusted to be small."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_BYTES:
            raise ValueError(f"страница больше {MAX_BYTES // 1024 // 1024} МБ")
        chunks.append(chunk)
    raw = b"".join(chunks)
    # The declared charset, falling back to utf-8 with replacement: a page that
    # lies about its encoding should arrive slightly wrong, not not at all.
    encoding = response.charset_encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def to_text(html: str) -> str:
    """The readable text of a page.

    Deliberately hand-rolled and deliberately crude. A real extractor
    (readability, trafilatura, beautifulsoup) is a dependency and a model of
    what an article looks like; what the index needs is the words, and the
    chunker does not care about paragraphs it did not get. Everything here is
    reversible: if the text turns out to be poor, this is one function.
    """
    without = _COMMENTS.sub(" ", html)
    without = _DROP.sub(" ", without)
    without = _BREAKS.sub("\n\n", without)
    text = _TAGS.sub(" ", without)
    for entity, character in _ENTITIES.items():
        text = text.replace(entity, character)
    # Numeric entities after the named ones, so &#39; is already gone.
    text = re.sub(r"&#\d{1,6};", " ", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return _BLANK.sub("\n\n", "\n".join(line for line in lines if line)).strip()


def title_of(html: str) -> str:
    found = _TITLE.search(html)
    if not found:
        return ""
    return " ".join(_TAGS.sub(" ", found.group(1)).split())[:200]


def _host(url: str) -> str:
    return httpx.URL(url).host or url


def as_markdown(page: Page) -> str:
    """The page as the file that will be indexed.

    The title becomes a heading and the address a line under it: the first is
    what makes the document findable by name — the same reason uploads get a
    title heading — and the second is the only record of where this came from
    once it is one object among many.
    """
    return f"# {page.title}\n\n{page.url}\n\n{page.text}\n"
