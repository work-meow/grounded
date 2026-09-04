"""The LangChain agent: a model that calls retrieval tools in a loop.

We do not hand-roll the loop — ``create_agent`` already is one, and middleware
gives us the call ceilings. What lives here is the tool surface and the
citation bookkeeping that turns retrieved chunks into clickable sources.
"""

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openrouter import ChatOpenRouter

from app import retriever
from app.config import Settings
from app.models import Message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты — ассистент по личной базе знаний пользователя.

Правила:
- Отвечай на основании фрагментов, полученных через search_knowledge. Не выдумывай.
- Если первый поиск не дал ответа, переформулируй запрос и попробуй ещё раз.
- Если в базе ничего нет, прямо скажи об этом и не додумывай.
- Ссылайся на источники в тексте как [1], [2] — их нумерация совпадает с той,
  что возвращают инструменты.
- Отвечай на языке вопроса, кратко и по существу.
"""


# The [1] / [12] markers the model is told to write, as they appear in the answer.
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass
class _Citations:
    """Chunks shown to the agent, numbered in first-seen order."""

    items: list[dict[str, Any]] = field(default_factory=list)
    _numbers: dict[tuple[str | None, int | None], int] = field(default_factory=dict)

    def add(self, chunk: retriever.Chunk) -> int:
        """Register a chunk and return its stable citation number."""
        key = (chunk.document_id, chunk.page)
        if key not in self._numbers:
            self._numbers[key] = len(self.items) + 1
            self.items.append(
                {
                    "n": self._numbers[key],
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "snippet": chunk.text[:300],
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


def _render(chunks: list[retriever.Chunk], citations: _Citations) -> str:
    if not chunks:
        return "Ничего не найдено. Попробуй другую формулировку запроса."
    lines = []
    for chunk in chunks:
        number = citations.add(chunk)
        where = chunk.filename or "документ"
        if chunk.page is not None:
            where += f", стр. {chunk.page}"
        lines.append(f"[{number}] ({where})\n{chunk.text}")
    return "\n\n".join(lines)


def _build_tools(settings: Settings, user_id: UUID, citations: _Citations) -> list[BaseTool]:
    """Tools bound to one user by closure.

    Binding at construction time (rather than passing the user through agent
    state) means there is no code path in which a tool could be invoked for
    anyone else.
    """

    @tool
    async def search_knowledge(query: str) -> str:
        """Найти релевантные фрагменты в базе знаний пользователя.

        Args:
            query: Поисковый запрос на естественном языке.
        """
        chunks = await retriever.retrieve(settings, user_id, query, settings.retrieve_k)
        return _render(chunks, citations)

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
        # No membership check: retrieve() filters on user_id *and* document_id,
        # so somebody else's document returns nothing rather than being refused.
        # Parsing to UUID first is the guard that matters — it is what keeps the
        # id out of the filter expression as anything but hex and dashes.
        chunks = await retriever.retrieve(
            settings, user_id, query, settings.retrieve_k, document_id=target
        )
        return _render(chunks, citations)

    return [search_knowledge, list_sources, read_document]


@lru_cache(maxsize=1)
def _model(model: str, api_key: str, temperature: float, effort: str = "") -> ChatOpenRouter:
    """One chat client for the whole process.

    Built per request, this would open a fresh HTTP connection pool on every
    question and never close it. The client carries no per-user state — only
    the tools do — so a single instance is safe to share.
    """
    reasoning = {"effort": effort} if effort else None
    return ChatOpenRouter(
        model=model, api_key=api_key, temperature=temperature, reasoning=reasoning
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


async def answer(
    settings: Settings,
    user_id: UUID,
    question: str,
    history: list[Message],
) -> AsyncIterator[tuple[str, Any]]:
    """Stream ``("token", str)`` events, then one final ``("citations", list)``.

    The agent is compiled per request so that its tools can close over the
    user. Compiling a three-tool graph is cheap next to a single LLM call.
    """
    citations = _Citations()
    agent = create_agent(
        model=_model(
            settings.agent_model,
            settings.openrouter_api_key,
            settings.agent_temperature,
            settings.agent_reasoning_effort,
        ),
        tools=_build_tools(settings, user_id, citations),
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(
                run_limit=settings.max_tool_calls_per_run, exit_behavior="continue"
            ),
            ModelCallLimitMiddleware(
                run_limit=settings.max_model_calls_per_run, exit_behavior="end"
            ),
        ],
    )

    inputs = {"messages": [*_history(history), HumanMessage(question)]}
    # Kept so the citation markers can be read back out of the finished answer.
    parts: list[str] = []
    async for chunk, meta in agent.astream(inputs, stream_mode="messages"):
        if meta.get("langgraph_node") != "model":
            continue
        if text := _text_of(chunk):
            parts.append(text)
            yield "token", text

    yield "citations", citations.referenced_in("".join(parts))
