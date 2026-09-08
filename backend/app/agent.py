"""The LangChain agent: a model that calls retrieval tools in a loop.

We do not hand-roll the loop — ``create_agent`` already is one, and middleware
gives us the call ceilings. What lives here is the tool surface and the
citation bookkeeping that turns retrieved chunks into clickable sources.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

from app import expansion, relevance, retriever, websearch
from app.config import Settings
from app.models import Message
from app.spend import ANSWER, Spend

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты — ассистент по личной базе знаний пользователя.

Правила:
- Отвечай на основании фрагментов, полученных через search_knowledge. Не выдумывай.
- Индекс всегда возвращает несколько лучших фрагментов, даже когда подходящих
  нет. Если ни один не отвечает на вопрос — сразу сделай ещё один поиск с
  другой формулировкой (синонимы, точные термины, без вопросительных слов).
- Действуй молча: не спрашивай разрешения, не описывай, что собираешься
  сделать, и не предлагай пользователю список вариантов. Либо ищи, либо отвечай.
- Если и после этого ничего нет — одной фразой скажи об этом. Сноска на
  нерелевантный фрагмент — это ложная ссылка: когда ответа нет, сносок тоже нет.
- Ссылайся на источники в тексте как [1], [2] — их нумерация совпадает с той,
  что возвращают инструменты.
- Отвечай на языке вопроса, кратко и по существу.
- Пиши обычным текстом, без markdown-разметки: интерфейс показывает ответ как
  есть, и «**жирный**» в нём видно звёздочками. Списки — простым дефисом.
"""

#: Added only when the user turned web search on for this message. Left out
#: entirely otherwise, rather than saying "у тебя нет доступа к интернету":
#: naming a tool that is not bound is how a model ends up apologising for not
#: calling it.
WEB_RULES = """\
- Поиск в сети включён, и порядок такой. Вопрос про содержимое документов
  пользователя — search_knowledge. Вопрос про внешний мир и сегодняшний день
  (курсы, ставки, новости, цены, законы, люди, события) — сразу search_web, не
  тратя поиск по базе.
- Никогда не отвечай «в базе этого нет», не вызвав search_web: он включён ровно
  для таких случаев. Правило выше про «скажи одной фразой» относится к базе, а
  не к концу разговора.
- Результаты из сети нумеруются вместе с фрагментами из базы: ссылайся на них
  так же, как [1], [2], и не приписывай в конце свой список источников —
  интерфейс показывает их сам, отдельными ссылками.
- Числа и даты бери только из найденного текста. Если ни сводка, ни источники
  не называют нужное число — так и скажи, а не подставляй правдоподобное:
  страницы часто приходят с историческими таблицами и навигацией, и число из
  них легко оказывается не тем.
- Если источники расходятся, скажи об этом и назови, что говорит каждый.
- Если и в сети ничего не нашлось, тогда скажи об этом прямо."""


#: Written out rather than taken from a locale. Russian month names need the
#: genitive ("5 сентября", not "сентябрь"), strftime under a ru_RU locale needs
#: that locale generated in the image, and setlocale is process-global and not
#: thread-safe. Nineteen strings cost less than either.
_WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)
_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def _today(now: datetime) -> str:
    return (
        f"Сегодня {_WEEKDAYS[now.weekday()]}, {now.day} {_MONTHS[now.month - 1]} {now.year} года."
    )


def system_prompt(settings: Settings, web: bool = False) -> str:
    """The rules, what day it is, and — when it is on — how to use the web.

    A model has no idea. Half the questions asked of a personal knowledge base
    only mean something relative to today — "что на этой неделе", "сколько
    осталось до сдачи", "последние заметки" — and without the date the model
    either invents one out of its training data or declines, and both look from
    the outside like the search having failed.

    The date and not the time, deliberately: it is the same string for a whole
    day, so the system prompt is byte-identical across a day's requests and
    stays eligible for upstream prompt caching.
    """
    today = _today(datetime.now(ZoneInfo(settings.timezone)))
    rules = f"{SYSTEM_PROMPT}{WEB_RULES}\n" if web else SYSTEM_PROMPT
    return f"{rules}\n{today} Считай «сегодня», «вчера», «на этой неделе» от этой даты.\n"


#: How much of a source is kept for the chip the reader hovers. Enough to
#: recognise the fragment, far short of reproducing the document.
SNIPPET_CHARS = 300

# The [1] / [12] markers the model is told to write, as they appear in the answer.
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass
class _Citations:
    """Chunks shown to the agent, numbered in first-seen order."""

    items: list[dict[str, Any]] = field(default_factory=list)
    _numbers: dict[tuple[str | None, int | None], int] = field(default_factory=dict)

    def add(self, chunk: retriever.Chunk) -> int:
        """Register a fragment from the knowledge base."""
        return self._number(
            (chunk.document_id, chunk.page),
            document_id=chunk.document_id,
            filename=chunk.filename,
            page=chunk.page,
            snippet=chunk.text[:SNIPPET_CHARS],
        )

    def add_link(self, source: websearch.Source) -> int:
        """Register a page found on the web.

        Numbered from the same counter as the fragments, because the answer
        mixes them: an answer that leans on a document and a news item cites
        [1] and [2] without the reader needing to know which is which.
        """
        return self._number(
            (source.url, None),
            filename=source.title or websearch.host(source.url),
            snippet=source.text[:SNIPPET_CHARS],
            url=source.url,
        )

    def _number(self, key: tuple[str | None, int | None], **item: Any) -> int:
        if key not in self._numbers:
            self._numbers[key] = len(self.items) + 1
            self.items.append(
                {
                    "n": self._numbers[key],
                    "document_id": None,
                    "filename": None,
                    "page": None,
                    "url": None,
                    **item,
                }
            )
        return self._numbers[key]

    def referenced_in(self, answer: str) -> list[dict[str, Any]]:
        """The sources the answer actually points at, in citation order.

        A search hands the model every chunk it found — eight of them by
        default — and a good answer leans on two. Returning all eight would put
        [2]..[8] under an answer whose text mentions none of them: chips that
        look like references to something the reader never sees. So the answer
        itself decides, and an answer that cites nothing gets no sources.

        The numbers are kept as issued, gaps and all: an answer that cites only
        the fifth chunk shows a single source labelled [5], because that is what
        its own text points at. Renumbering the list would silently break the
        tie between the marker in the sentence and the chip under it.
        """
        cited = {int(marker) for marker in _MARKER_RE.findall(answer)}
        return [item for item in self.items if item["n"] in cited]

    def shown(self) -> list[dict[str, Any]]:
        """Everything the model was handed, without the text of it.

        Kept because precision is a question about the fragments a turn was
        *given*, not the ones it used: eight retrieved and two cited is a
        different search from three retrieved and two cited, and today only
        the second number survives the turn.

        A count does not need the text. The cited fragments already carry
        their snippets, and twenty of those per message would be a table full
        of prose nobody reads back.
        """
        return [
            {key: item[key] for key in ("n", "document_id", "filename", "page")}
            for item in self.items
        ]


def _candidates(settings: Settings) -> int:
    """How many fragments to ask the index for.

    More when something is going to judge them, because judging twenty costs the
    same one call as judging eight and gives it more to find the answer in.
    """
    return settings.rerank_candidates if settings.rerank_enabled else settings.retrieve_k


def _render(
    query: str, chunks: list[retriever.Chunk], citations: _Citations, web: bool = False
) -> str:
    """Fragments as the model sees them, with what they are said plainly.

    The header is not decoration. The index is a ranked retriever: it returns
    its k best fragments for any query at all, and measured on the deployment,
    a question about something the base has never heard of comes back with
    eight confident-looking fragments about something else. A model given those
    with no framing reads them as "the matches" and concludes the base has
    nothing — on the first try, every time. Saying what they actually are is
    what turns one search into a loop worth having.
    """
    if not chunks:
        # Not "the index returned nothing" — it never does. Something read the
        # candidates and found none of them about this, which is a far better
        # thing to tell the model than eight paragraphs about something else.
        return f"По запросу «{query}» подходящих фрагментов нет. {_next_move(web)}"
    lines = [
        (
            f"Подходящие фрагменты по запросу «{query}» — {len(chunks)}. "
            "Если ни один всё же не отвечает на вопрос — молча вызови поиск ещё раз "
            f"с другой формулировкой. {_next_move(web)}"
        )
    ]
    for chunk in chunks:
        number = citations.add(chunk)
        where = chunk.filename or "документ"
        if chunk.page is not None:
            where += f", стр. {chunk.page}"
        lines.append(f"[{number}] ({where})\n{chunk.text}")
    return "\n\n".join(lines)


def _next_move(web: bool) -> str:
    """What to do when the knowledge base has come up empty twice.

    It has to be said here, in the tool result, and not only in the system
    prompt: this is the sentence the model is reading at the moment it decides,
    and measured on the deployment it is the one that wins. With the web
    enabled and only the prompt saying so, the agent searched the documents
    twice and answered "в базе этого нет" without ever going online — which is
    the one thing the person had just asked it not to do.
    """
    if web:
        return "Если и тогда ничего — вызови search_web, а не отвечай, что ничего нет."
    return "Если и тогда ничего — ответь одной фразой, что в базе этого нет, без сносок."


def _render_web(found: websearch.Result, citations: _Citations) -> str:
    lines = []
    if found.summary:
        lines.append(f"Сводка поиска:\n{found.summary}")
    for source in found.sources:
        number = citations.add_link(source)
        lines.append(f"[{number}] ({source.title} — {websearch.host(source.url)})\n{source.text}")
    return "\n\n".join(lines)


def _build_tools(
    settings: Settings,
    user_id: UUID,
    citations: _Citations,
    web: bool = False,
    spend: Spend | None = None,
) -> list[BaseTool]:
    """Tools bound to one user by closure.

    Binding at construction time (rather than passing the user through agent
    state) means there is no code path in which a tool could be invoked for
    anyone else.
    """

    @tool
    async def search_knowledge(query: str, days: int | None = None) -> str:
        """Найти релевантные фрагменты в базе знаний пользователя.

        Args:
            query: Поисковый запрос на естественном языке.
            days: Искать только среди документов, изменённых за последние N
                дней. Указывай, только если в вопросе есть привязка ко времени
                («на этой неделе», «вчера», «в этом месяце»); иначе не указывай.
        """
        chunks = await relevance.keep_relevant(
            settings,
            query,
            await expansion.search(
                settings, user_id, query, _candidates(settings), since=retriever.since(days)
            ),
            spend,
        )
        return _render(query, chunks, citations, web)

    @tool
    async def list_sources() -> str:
        """Перечислить документы, доступные пользователю, с их идентификаторами."""
        # Asked of the index, not the database, and only when the model calls
        # this tool. The database knows about uploads; the index knows about
        # those *and* everything reached through a connected source, which is
        # the only place a Notion page or a Drive file exists at all. Doing it
        # here rather than up front also keeps the cost off the common path —
        # most turns only ever call search_knowledge.
        try:
            found = await retriever.indexed_documents(settings, user_id)
        except Exception:
            logger.exception("could not list the documents of %s", user_id)
            return "Не удалось получить список документов. Используй search_knowledge."
        if not found:
            return "База знаний пуста — пользователь ещё ничего не загрузил и не подключил."
        return "\n".join(
            f"{document.document_id} — {document.filename}"
            + ("" if document.ready else " (индексируется)")
            for document in found
        )

    @tool
    async def read_document(document_id: str, query: str) -> str:
        """Искать внутри одного конкретного документа.

        Полезно, когда нужна точная цитата из известного файла.

        Args:
            document_id: Идентификатор из list_sources.
            query: Что именно нужно найти в этом документе.
        """
        try:
            target = UUID(document_id)
        except ValueError:
            return f"Некорректный document_id: {document_id}"
        # Not expanded into several phrasings, unlike search_knowledge: this is
        # already narrowed to one document, so there is far less for a
        # rephrasing to find, and the extra call would be spent on every read.
        #
        # No membership check: retrieve() filters on user_id *and* document_id,
        # so somebody else's document returns nothing rather than being refused.
        # Parsing to UUID first is the guard that matters — it is what keeps the
        # id out of the filter expression as anything but hex and dashes.
        chunks = await relevance.keep_relevant(
            settings,
            query,
            await retriever.retrieve(
                settings, user_id, query, _candidates(settings), document_id=target
            ),
            spend,
        )
        return _render(query, chunks, citations, web)

    @tool
    async def search_web(query: str) -> str:
        """Найти информацию в интернете.

        Вызывай, когда ответа нет в базе знаний пользователя или когда нужны
        актуальные данные: новости, курсы, цены, законы, факты о внешнем мире.

        Args:
            query: Поисковый запрос. Формулируй самодостаточно, как для
                поисковика: без местоимений и отсылок к предыдущим сообщениям,
                и на языке вопроса — ответ про российские реалии скорее найдётся
                по-русски, чем по-английски.
        """
        try:
            found = await websearch.search(settings, query, spend)
        except Exception:
            # Never fatal: the turn can still be answered from the knowledge
            # base, and a tool that raises takes the whole answer with it.
            logger.exception("web search failed for %s", user_id)
            return "Поиск в сети не сработал. Ответь по тому, что есть в базе знаний."
        if not found.sources and not found.summary:
            return f"По запросу «{query}» в сети ничего не нашлось."
        return _render_web(found, citations)

    tools: list[BaseTool] = [search_knowledge, list_sources, read_document]
    if web:
        tools.append(search_web)
    return tools


# Room for every distinct client a turn asks for — the answering model, the
# fallback and the summariser — plus a couple spare. It was 1 while there was
# one model, and when the other two arrived that silently turned into a cache
# that evicted on every call: measured at hits=0, misses=4, which is three
# fresh connection pools per question and exactly the leak this exists to
# prevent. Bounded rather than unbounded because the key holds an api key.
@lru_cache(maxsize=8)
def _model(model: str, api_key: str, temperature: float, effort: str = "") -> ChatOpenRouter:
    """One chat client per configuration, for the whole process.

    Built per request, each would open a fresh HTTP connection pool on every
    question and never close it. The client carries no per-user state — only
    the tools do — so an instance is safe to share.
    """
    reasoning = {"effort": effort} if effort else None
    # No timeout here, and not for want of trying: ChatOpenRouter takes both
    # `timeout` and `request_timeout` without complaint and then never
    # completes a request — measured on the deployment, a call that answers in
    # one second hangs past five minutes with either of them set. It does not
    # accept an http_async_client to configure instead. The turn is bounded in
    # answer() rather than here.
    return ChatOpenRouter(
        # SecretStr rather than the bare string it accepts: pydantic would
        # coerce it anyway, and this way the key cannot come back out of a
        # repr() of the client — which is what ends up in a traceback.
        model=model,
        api_key=SecretStr(api_key),
        temperature=temperature,
        reasoning=reasoning,
    )


def _history(messages: list[Message]) -> list[BaseMessage]:
    return [HumanMessage(m.content) if m.role == "user" else AIMessage(m.content) for m in messages]


def _text_of(chunk: Any) -> str:
    """AIMessageChunk content is either a string or a list of content blocks."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


@dataclass
class _Call:
    """One tool call, as it arrives in pieces."""

    tool: str
    args: str = ""
    #: Set once the arguments have parsed and the query has been announced.
    settled: bool = False


def _query_of(args: str) -> str | None:
    """The `query` argument, once enough of it has streamed in to parse.

    Arguments arrive as JSON fragments, and JSON is only valid once it is
    balanced — so this returns None for every fragment until the last one, and
    there is no half-parsed query to guard against.
    """
    try:
        parsed = json.loads(args)
    except ValueError:
        return None
    query = parsed.get("query") if isinstance(parsed, dict) else None
    return query.strip() if isinstance(query, str) and query.strip() else None


def _limits(settings: Settings, web: bool) -> list[Any]:
    """Ceilings on one turn, and what happens when a call fails inside it.

    Per tool as well as overall, and that is what makes the retry loop safe to
    ask for: the model is told to search again with different wording when the
    fragments do not answer, and without a bound on that particular tool the
    instruction is an invitation to spend the afternoon looking.
    """
    limits: list[Any] = [
        ToolCallLimitMiddleware(
            tool_name="search_knowledge",
            run_limit=settings.max_knowledge_searches_per_run,
            exit_behavior="continue",
        ),
        # Reading a document costs what a search costs — one request to the
        # index and one call to the judge — and it was bounded only by the
        # overall six. "Read it again, differently" was therefore looser than
        # "search again, differently", which is the loop that was actually
        # designed for.
        ToolCallLimitMiddleware(
            tool_name="read_document",
            run_limit=settings.max_document_reads_per_run,
            exit_behavior="continue",
        ),
    ]
    if web:
        # Tighter, because this one costs real money per call rather than
        # fractions of a cent: about $0.00125 a search.
        limits.append(
            ToolCallLimitMiddleware(
                tool_name="search_web",
                run_limit=settings.max_web_searches_per_run,
                exit_behavior="continue",
            )
        )
    limits.append(
        ToolCallLimitMiddleware(run_limit=settings.max_tool_calls_per_run, exit_behavior="continue")
    )
    limits.append(
        ModelCallLimitMiddleware(run_limit=settings.max_model_calls_per_run, exit_behavior="end")
    )

    # A blip at the provider used to cost the whole turn — an error event for
    # the reader, a 502 for the API — while being exactly the kind of failure a
    # second attempt fixes.
    if settings.agent_retries:
        limits.append(
            ModelRetryMiddleware(
                max_retries=settings.agent_retries,
                backoff_factor=2.0,
                # "error", not the default "continue". A provider that stays
                # down would otherwise end the turn with an empty answer and
                # no complaint — a 200 with nothing in it, which the caller
                # cannot tell from "the base had no answer". Failing is what
                # the error event and the 502 already exist for.
                on_failure="error",
            )
        )
    # And when it is down rather than slow, answering from the documents on
    # something cheaper beats not answering. Passed as a built client and not
    # as a name: a string goes through LangChain's init_chat_model, which knows
    # nothing about OpenRouter and would ask for another provider's key.
    if settings.agent_fallback_model:
        limits.append(
            ModelFallbackMiddleware(
                _model(
                    settings.agent_fallback_model,
                    settings.openrouter_api_key,
                    settings.agent_temperature,
                )
            )
        )

    # Twenty long messages arrive as most of a context window, and the oldest
    # of them is the least likely to matter. Summarised rather than dropped:
    # "как я говорил выше" has to keep meaning something.
    limits.append(
        SummarizationMiddleware(
            model=_model(
                settings.rerank_model, settings.openrouter_api_key, settings.agent_temperature
            ),
            trigger=("tokens", settings.history_summarise_above_tokens),
        )
    )
    # Three searches of six fragments each is a large part of what the model
    # reads, and the first search after two reformulations is rarely still
    # needed. This prunes those results rather than the conversation.
    limits.append(ContextEditingMiddleware())
    return limits


async def answer(
    settings: Settings,
    user_id: UUID,
    question: str,
    history: list[Message],
    web: bool = False,
    spend: Spend | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Stream the turn as it happens, ending with one ``("citations", list)``.

    Three kinds of event. ``("step", {"tool", "query"})`` when the model decides
    to call a tool, ``("token", str)`` for the answer as it is written, and the
    citations at the end.

    ``web`` is per message, not per user: the search tool is bound only when it
    is on, so a turn without it cannot spend money on one — and the model is
    never told about a tool it does not have, which is how one ends up
    apologising for not calling it.

    ``spend``, when given, is filled in as the turn runs — the model's own
    calls here, the judge and any web search from inside the tools. It is the
    caller's object rather than a return value because a streamed turn has no
    return: the API reads it once the iteration is over.

    The agent is compiled per request so that its tools can close over the
    user. Compiling a four-tool graph is cheap next to a single LLM call.
    """
    citations = _Citations()
    agent = create_agent(
        model=_model(
            settings.agent_model,
            settings.openrouter_api_key,
            settings.agent_temperature,
            settings.agent_reasoning_effort,
        ),
        tools=_build_tools(settings, user_id, citations, web, spend),
        system_prompt=system_prompt(settings, web),
        middleware=_limits(settings, web),
    )

    inputs = {"messages": [*_history(history), HumanMessage(question)]}
    # Kept so the citation markers can be read back out of the finished answer.
    parts: list[str] = []
    # Tool calls being assembled, by their position in the message.
    calls: dict[int, _Call] = {}

    # The whole turn, not one request: a provider that accepts the connection
    # and then says nothing would otherwise hold the SSE stream open behind it
    # for as long as it liked. The call-count middleware bounds how many
    # requests a turn makes, never how long one takes. On expiry the caller
    # sees a failed turn and keeps whatever tokens had arrived.
    async with asyncio.timeout(settings.agent_timeout_s):
        async for chunk, meta in agent.astream(inputs, stream_mode="messages"):
            if meta.get("langgraph_node") != "model":
                continue
            # A tool call arrives in pieces: the first chunk carries its name,
            # the rest carry fragments of its JSON arguments. The name goes out
            # at once — it is the earliest honest thing to show, before the
            # arguments have finished streaming and well before the tool runs —
            # and the query follows a few hundred milliseconds later, as soon as
            # the fragments parse. Verified against the live provider.
            for part in getattr(chunk, "tool_call_chunks", None) or []:
                index = part.get("index") or 0
                if name := part.get("name"):
                    calls[index] = _Call(tool=name)
                    yield "step", {"tool": name, "query": ""}
                call = calls.get(index)
                if call is None or call.settled:
                    continue
                call.args += part.get("args") or ""
                if (query := _query_of(call.args)) is not None:
                    call.settled = True
                    yield "step", {"tool": call.tool, "query": query}
            # The last chunk of every model call carries no text, only what that
            # call used and what it cost. Verified against the live provider:
            # one such chunk per call, cost included, with no request-body flag
            # asked for — and it reaches here through LangGraph unchanged.
            if spend is not None and (usage := getattr(chunk, "usage_metadata", None)):
                spend.add(
                    ANSWER,
                    settings.agent_model,
                    prompt_tokens=usage.get("input_tokens"),
                    completion_tokens=usage.get("output_tokens"),
                    cost_usd=(chunk.response_metadata or {}).get("cost"),
                )
            if text := _text_of(chunk):
                parts.append(text)
                yield "token", text

    yield "citations", citations.referenced_in("".join(parts))
    # After the citations, because every consumer written so far switches on the
    # event names it knows and ignores the rest — this one is for whatever is
    # recording the turn, not for the screen.
    yield "shown", citations.shown()
