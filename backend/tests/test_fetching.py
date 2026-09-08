"""Indexing a web page by its address.

This is the second place where the *user* names a machine — the first was an
S3 endpoint typed into a form, which is why `verify_public` exists. It is not
enough on its own here: a redirect is a second address chosen by the far end,
so checking only what somebody typed would let
`https://example.com/r?to=http://127.0.0.1:8000` straight through. Hops are
therefore followed by hand and every one is checked before it is taken, and
that is the first thing tested.

The rest is about refusing honestly. A login wall, a PDF behind a link, a page
that is all navigation — each of those would otherwise become a document that
is indexed, found and cited as though it said something.
"""

import httpx
import pytest

from app import fetching, http
from app.config import Settings


def settings(**overrides) -> Settings:
    return Settings(**{"jwt_secret": "x" * 40, "openrouter_api_key": "k", **overrides})


ARTICLE = (
    "<html><head><title>Как отметить задачу</title></head><body>"
    "<nav>меню и ссылки навигации</nav>"
    "<script>var x = 'скрипт не текст';</script>"
    "<p>Чтобы отметить задачу выполненной, поставьте галочку в чекбоксе.</p>"
    "<p>Отфильтруйте вид по полю Status, чтобы видеть только незавершённые.</p>"
    "<p>" + "Дополнительный абзац с содержанием страницы. " * 6 + "</p>"
    "<footer>подвал</footer></body></html>"
)


@pytest.fixture
def web():
    """The open web, answering whatever the test says, hop by hop."""
    asked: list[str] = []

    def serve(*responses):
        answers = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            asked.append(str(request.url))
            answer = answers.pop(0) if len(answers) > 1 else answers[0]
            if isinstance(answer, Exception):
                raise answer
            return answer

        http.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return asked

    yield serve
    http.set_client(None)


def page(body: str = ARTICLE, *, status: int = 200, kind: str = "text/html") -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": kind})


def redirect(to: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": to})


# --- the address, and every address after it ---------------------------------


async def test_an_address_inside_the_network_is_refused_before_anything_is_fetched(web):
    asked = web(page())

    with pytest.raises(ValueError, match="внутрь сети"):
        await fetching.fetch(settings(), "http://127.0.0.1:8000/secrets")

    assert asked == [], "и запроса не было"


async def test_a_redirect_into_the_network_is_refused_too(web):
    """The hop nobody typed. Checking only the address somebody pasted is what
    makes an open redirect on a public site into a way in."""
    web(redirect("http://127.0.0.1:8000/secrets"), page())

    with pytest.raises(ValueError, match="внутрь сети"):
        await fetching.fetch(settings(), "https://example.com/r")


async def test_a_relative_redirect_is_resolved_before_it_is_checked(web):
    """`Location: /article` is legal and common; left unresolved it would be
    neither fetchable nor checkable."""
    asked = web(redirect("/article"), page())

    fetched = await fetching.fetch(settings(), "https://example.com/start")

    assert asked[-1] == "https://example.com/article"
    assert fetched.url == "https://example.com/article"


async def test_a_redirect_loop_ends(web):
    web(redirect("https://example.com/again"))

    with pytest.raises(ValueError, match="переадресаций"):
        await fetching.fetch(settings(), "https://example.com/start")


async def test_a_scheme_that_is_not_http_is_refused(web):
    web(page())

    with pytest.raises(ValueError, match="http"):
        await fetching.fetch(settings(), "file:///etc/passwd")


# --- what comes back ---------------------------------------------------------


async def test_the_article_survives_and_the_furniture_does_not(web):
    web(page())

    fetched = await fetching.fetch(settings(), "https://example.com/a")

    assert "галочку в чекбоксе" in fetched.text
    assert "скрипт не текст" not in fetched.text
    assert "меню и ссылки навигации" not in fetched.text
    assert "подвал" not in fetched.text
    assert fetched.title == "Как отметить задачу"


async def test_paragraphs_do_not_become_one_line(web):
    web(page())

    fetched = await fetching.fetch(settings(), "https://example.com/a")

    assert "\n" in fetched.text


async def test_a_page_with_almost_no_text_is_refused(web):
    """A login wall looks exactly like this, and an empty document in the base
    is worse than a refusal: it gets indexed, found and cited as if it said
    something."""
    web(page("<html><body><p>Войдите, чтобы продолжить</p></body></html>"))

    with pytest.raises(ValueError, match="почти нет текста"):
        await fetching.fetch(settings(), "https://example.com/login")


async def test_something_that_is_not_a_page_is_refused_with_what_it_was(web):
    web(page("%PDF-1.4", kind="application/pdf"))

    with pytest.raises(ValueError, match="application/pdf"):
        await fetching.fetch(settings(), "https://example.com/a.pdf")


async def test_an_error_page_is_not_a_document(web):
    web(page("<html><body>" + "Страница не найдена. " * 30 + "</body></html>", status=404))

    with pytest.raises(ValueError, match="404"):
        await fetching.fetch(settings(), "https://example.com/gone")


async def test_a_page_bigger_than_the_ceiling_is_refused(web):
    web(page("<html><body><p>" + "х" * (fetching.MAX_BYTES + 1000) + "</p></body></html>"))

    with pytest.raises(ValueError, match="МБ"):
        await fetching.fetch(settings(), "https://example.com/big")


async def test_a_page_that_lies_about_its_encoding_still_arrives(web):
    """Slightly wrong beats not at all: the words are what the index needs."""
    web(
        httpx.Response(
            200,
            content=("<html><body><p>" + "Текст страницы. " * 20 + "</p></body></html>").encode(),
            headers={"content-type": "text/html; charset=x-mac-nonexistent"},
        )
    )

    fetched = await fetching.fetch(settings(), "https://example.com/a")

    assert "Текст" in fetched.text


# --- what gets stored --------------------------------------------------------


def test_the_stored_file_leads_with_the_title_and_the_address():
    """The title goes into the document's first chunk as a heading, which is
    how a page is findable by what it is called; the address is the only record
    of where it came from once it is one object among many."""
    stored = fetching.as_markdown(
        fetching.Page(url="https://example.com/a", title="Как отметить задачу", text="Текст.")
    )

    assert stored.startswith("# Как отметить задачу\n\nhttps://example.com/a\n\n")


def test_a_page_with_no_title_is_named_by_its_host():
    assert fetching.title_of("<html><body>без заголовка</body></html>") == ""
