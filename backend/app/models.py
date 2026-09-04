import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
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

    For the MVP the only kind is ``files`` and each user gets exactly one,
    created on first upload. The table exists because the S3 layout and the
    connector roadmap (Drive, GitHub, Notion) are keyed on it.
    """

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="files")
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = _created_at()

    documents: Mapped[list["Document"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


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
    created_at: Mapped[datetime] = _created_at()

    chat: Mapped[Chat] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_messages_chat_created", "chat_id", "created_at"),)
