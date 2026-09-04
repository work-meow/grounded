import uuid
from datetime import datetime
from pathlib import PurePosixPath

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel
from rag_shared.doc_key import build_key
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import retriever, storage
from app.config import Settings
from app.deps import SessionDep, SettingsDep, UserDep
from app.models import Document, Source

router = APIRouter(prefix="/api/sources", tags=["sources"])

# Must stay in step with indexer/rag_indexer/parsers.py. Anything the indexer
# cannot read would sit in the bucket forever showing "processing", so it is
# rejected at the door instead.
ALLOWED_SUFFIXES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    created_at: datetime
    status: str  # "processing" | "ready"


async def _files_source(session: AsyncSession, user_id: uuid.UUID) -> Source:
    """One implicit 'Uploaded files' source per user, created on demand."""
    source = (
        await session.execute(
            select(Source).where(Source.user_id == user_id, Source.kind == "files")
        )
    ).scalar_one_or_none()
    if source is None:
        source = Source(user_id=user_id, kind="files", name="Загруженные файлы")
        session.add(source)
        await session.commit()
    return source


async def _ready_ids(settings: Settings, user_id: uuid.UUID) -> set[str]:
    """Live indexing state; an empty set when the indexer is unreachable.

    A down indexer must not take the file list down with it — the files simply
    read as still processing, which is also what the user would do about it.
    """
    try:
        return await retriever.ready_document_ids(settings, user_id)
    except (httpx.HTTPError, ValueError):
        return set()


@router.get("")
async def list_documents(
    user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> list[DocumentOut]:
    rows = (
        (
            await session.execute(
                select(Document)
                .where(Document.user_id == user_id)
                .order_by(Document.created_at.desc())
            )
        )
        .scalars()
        .all()
    )

    ready = await _ready_ids(settings, user_id)
    return [
        DocumentOut(
            id=doc.id,
            filename=doc.filename,
            mime_type=doc.mime_type,
            size_bytes=doc.size_bytes,
            created_at=doc.created_at,
            status="ready" if str(doc.id) in ready else "processing",
        )
        for doc in rows
    ]


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> DocumentOut:
    filename = PurePosixPath(file.filename or "").name
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустое имя файла")

    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Поддерживаются: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    body = await file.read(settings.max_upload_bytes + 1)
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Максимум {settings.max_upload_bytes // 1024 // 1024} МБ",
        )
    if not body:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Файл пустой")

    source = await _files_source(session, user_id)
    document_id = uuid.uuid4()
    # Trust our own extension mapping rather than the client's Content-Type.
    mime_type = ALLOWED_SUFFIXES[suffix]
    key = build_key(user_id, source.id, document_id, filename)

    # S3 first: a row without an object would show "processing" forever, while
    # an object without a row is invisible and harmless.
    await storage.put(settings, key, body, mime_type)

    document = Document(
        id=document_id,
        user_id=user_id,
        source_id=source.id,
        filename=filename,
        s3_key=key,
        mime_type=mime_type,
        size_bytes=len(body),
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    return DocumentOut(
        id=document.id,
        filename=document.filename,
        mime_type=document.mime_type,
        size_bytes=document.size_bytes,
        created_at=document.created_at,
        status="processing",
    )


@router.get("/{document_id}/link")
async def document_link(
    document_id: uuid.UUID, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> dict[str, str]:
    """Presigned URL so a citation can open the original file."""
    document = await _owned(session, user_id, document_id)
    return {"url": await storage.presigned_url(settings, document.s3_key)}


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(
    document_id: uuid.UUID, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> None:
    document = await _owned(session, user_id, document_id)
    # Delete the object first: the indexer drops the chunks when the object
    # disappears, so this is what actually removes it from search.
    await storage.delete(settings, document.s3_key)
    await session.execute(sql_delete(Document).where(Document.id == document.id))
    await session.commit()


async def _owned(session: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    document = (
        await session.execute(
            select(Document).where(Document.id == document_id, Document.user_id == user_id)
        )
    ).scalar_one_or_none()
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Документ не найден")
    return document
