"""The LangChain agent: a model that calls retrieval tools in a loop.

We do not hand-roll the loop — ``create_agent`` already is one, and middleware
gives us the call ceilings. What lives here is the tool surface and the
citation bookkeeping that turns retrieved chunks into clickable sources.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter

from app import retriever
from app.config import Settings
from app.models import Message

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


@dataclass
class _Citations:
    """Chunks the agent actually looked at, in first-seen order."""

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


def _render(chunks: list[retriever.Chunk], citations: _Citations) -> str:
    if not chunks:
        return "Ничего не найдено. Попробуй другую формулировку запроса."
    lines = []
    for chunk in chunks:
        number = citations.add(chunk)
        where = chunk.filename or "документ"
        if chunk.page:
            where += f", стр. {chunk.page}"
        lines.append(f"[{number}] ({where})\n{chunk.text}")
    return "\n\n".join(lines)


def _build_tools(
    settings: Settings,
    user_id: UUID,
    documents: list[tuple[UUID, str]],
    citations: _Citations,
) -> list:
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
        if not documents:
            return "База знаний пуста — пользователь ещё не загрузил ни одного файла."
        return "\n".join(f"{doc_id} — {name}" for doc_id, name in documents)

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
        if target not in {doc_id for doc_id, _ in documents}:
            return "Документ не найден среди доступных пользователю."
        chunks = await retriever.retrieve(
            settings, user_id, query, settings.retrieve_k, document_id=target
        )
        return _render(chunks, citations)

    return [search_knowledge, list_sources, read_document]


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
    documents: list[tuple[UUID, str]],
) -> AsyncIterator[tuple[str, Any]]:
    """Stream ``("token", str)`` events, then one final ``("citations", list)``.

    The agent is compiled per request so that its tools can close over the
    user. Compiling a three-tool graph is cheap next to a single LLM call.
    """
    citations = _Citations()
    agent = create_agent(
        model=ChatOpenRouter(
            model=settings.agent_model,
            api_key=settings.openrouter_api_key,
            temperature=settings.agent_temperature,
        ),
        tools=_build_tools(settings, user_id, documents, citations),
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
    async for chunk, meta in agent.astream(inputs, stream_mode="messages"):
        if meta.get("langgraph_node") != "model":
            continue
        if text := _text_of(chunk):
            yield "token", text

    yield "citations", citations.items
