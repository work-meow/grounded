import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _pk()
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = _created_at()


class Source(Base):
    """A place documents come from.

    ``files`` is the implicit one: every user gets exactly one, created on their
    first upload, and it needs no configuration. The rest are connectors the
    user adds — Drive, Notion, Dropbox, OneDrive, Яндекс.Диск — and those carry
    a credential, which is why ``sealed_config`` exists.

    Their documents are not in :class:`Document`. Nothing fetches them, so there
    is nothing to record: they are listed from the index, which is the only
    process that has ever seen them.
    """

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="files")
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Fernet over the connector's configuration (rag_shared.crypto). Null for
    # ``files``, which has nothing to configure. Never leaves the process: it is
    # written here and copied into the manifest, and no response carries it.
    sealed_config: Mapped[str | None] = mapped_column(Text)
    # A digest of the fields that say *where* this source reads from
    # (rag_shared.connectors.fingerprint), so that "already connected" is a
    # question the database can answer without the credential. Null for
    # ``files``, and for connectors made before the check existed until the
    # first add fills them in.
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created_at()

    documents: Mapped[list["Document"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )

    # The same folder connected twice indexes every document in it twice and
    # pays for the embeddings twice. Checked before the insert for the sake of
    # the message; enforced here because that check and the insert are not one
    # step, and two tabs are enough to slip between them. Postgres treats nulls
    # as distinct, which is what leaves ``files`` and the not-yet-filled alone.
    __table_args__ = (UniqueConstraint("user_id", "fingerprint", name="uq_sources_user_finger"),)


class Document(Base):
    """One original file in object storage.

    Indexing state is deliberately NOT stored here: it is owned by Pathway and
    read live from ``/v1/inputs``. Keeping a copy would only create a second
    source of truth to drift out of sync.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    s3_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    mime_type: Mapped[str] = mapped_column(String(200), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Whether a PDF was found to contain any text (app.pdf). False means a scan:
    # indexed, searchable by nothing. Null for everything else — every other
    # format we accept is text by construction, and a PDF that would not open
    # is a file we have no claim to make about.
    text_layer: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = _created_at()

    source: Mapped[Source] = relationship(back_populates="documents")

    __table_args__ = (Index("ix_documents_user_created", "user_id", "created_at"),)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="Новый чат")
    created_at: Mapped[datetime] = _created_at()

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_chats_user_created", "user_id", "created_at"),)


class Spending(Base):
    """What one user was charged for in one day.

    A day per row, added to as turns finish, so the ceiling can be checked
    with one indexed read before a turn starts. Dollars rather than requests
    because that is what is being protected: two questions can differ twenty
    times in cost.

    Not derived from `messages.trace` even though the same figure is there — a
    question asked through the API without a chat is not stored anywhere, and
    those are most of what an integration does.
    """

    __tablename__ = "spending"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    #: Calendar day in UTC. One timezone for the window, deliberately: whose
    #: midnight it is would otherwise depend on a setting that exists for the
    #: agent's sense of "today", which is a different question.
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    turns: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Provider-reported, in dollars. Numeric and not float: this is money, and
    #: it is added to thousands of times.
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 8), nullable=False, default=0)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _pk()
    chat_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
    )
    # "user" | "assistant" — the only two the API ever writes.
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # [{document_id, filename, page, snippet}, ...] rendered as clickable sources.
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    # What the reader thought of it: 1, -1, or null for no opinion. Only ever
    # set on an assistant message. The point is not a score to average — it is
    # that a question somebody marked wrong is a question worth putting in the
    # eval set, and those are otherwise remembered by nobody.
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    # What the turn did: {"steps": [...], "shown": [...], "usage": {...}}, see
    # conversation.trace(). Null for every answer written before this column
    # existed, and for one the agent never finished.
    #
    # Here rather than in a table of its own because it is read exactly when
    # the message is: to show under the answer what was searched and what it
    # cost, and to turn a bad answer into a case that can be re-run. A join
    # would buy nothing and cost a migration.
    trace: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()

    chat: Mapped[Chat] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_messages_chat_created", "chat_id", "created_at"),)
