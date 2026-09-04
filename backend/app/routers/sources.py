"""The documents a user has, wherever they came from.

Two populations, one list. An upload is a row in ``documents`` and an object in
our bucket; a document from a connected source is neither — nothing here ever
fetched it, so the index is the only place that has seen it. Both are read back
from the index, which is what makes "is it searchable yet?" answerable at all.
"""

import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel
from rag_shared.doc_key import build_key
from rag_shared.formats import HUMAN_READABLE, mime_for
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import retriever, storage
from app.config import Settings
from app.deps import SessionDep, SettingsDep, UserDep
from app.models import Document, Source

router = APIRouter(prefix="/api/sources", tags=["sources"])


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    #: Unknown for a Notion page, which is not a file anywhere.
    size_bytes: int | None
    created_at: datetime
    status: str  # "processing" | "ready"
    source_name: str
    #: Whether deleting it here means anything. An upload is ours to remove; a
    #: document from a connected source is removed where it lives, and would
    #: come back on the next poll.
    removable: bool


async def _files_source(session: AsyncSession, user_id: uuid.UUID) -> Source:
    """One implicit 'Uploaded files' source per user, created on demand.

    ponytail: select-then-insert, with no unique constraint behind it. Two
    uploads racing on a user's very first file can create two rows. Ceiling:
    a duplicate source row — invisible, because documents are listed and
    filtered by user_id, never by source. Upgrade path: a unique index on
    (user_id, kind) and an upsert, if this ever becomes user-visible.
    """
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


async def _indexed(settings: Settings, user_id: uuid.UUID) -> dict[str, retriever.IndexedDocument]:
    """Live indexing state, keyed by document id; empty if the indexer is down.

    A down indexer must not take the file list down with it — uploads simply
    read as still processing, which is also what the user would do about it.
    Documents from connected sources disappear from the list entirely, because
    the index is the only record of them; that is honest rather than convenient.

    This call costs about 1.5 s after any pause, and about 15 ms if another one
    just preceded it — measured on the deployment box, and stable across runs.
    Pathway answers /v1/inputs from a snapshot it refreshes on a cycle, so a
    request that arrives mid-cycle waits for the next one. The list page is
    therefore slow to first paint and cheap to poll.

    That is deliberately not cached here. Readiness has exactly one home — the
    indexer — and a cache would make the UI able to claim a document is
    searchable while the process that answers searches disagrees. A skeleton for
    a second and a half is the cheaper of the two failures.
    """
    try:
        found = await retriever.indexed_documents(settings, user_id)
    except (httpx.HTTPError, ValueError):
        return {}
    return {document.document_id: document for document in found}


@router.get("")
async def list_documents(
    user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> list[DocumentOut]:
    rows = (
        (await session.execute(select(Document).where(Document.user_id == user_id))).scalars().all()
    )
    # Every source the user has, so that both populations can be labelled from
    # one query rather than a join for one and a lookup for the other.
    names = dict(
        (
            await session.execute(select(Source.id, Source.name).where(Source.user_id == user_id))
        ).tuples()
    )
    indexed = await _indexed(settings, user_id)

    documents = [
        DocumentOut(
            id=document.id,
            filename=document.filename,
            mime_type=document.mime_type,
            size_bytes=document.size_bytes,
            created_at=document.created_at,
            status="ready" if str(document.id) in indexed else "processing",
            source_name=names.get(document.source_id, "Загруженные файлы"),
            removable=True,
        )
        for document in rows
    ]

    uploaded = {str(document.id) for document in rows}
    documents.extend(
        DocumentOut(
            id=uuid.UUID(found.document_id),
            filename=found.filename,
            # The index has no mime type: it parses by sniffing the bytes. The
            # suffix is what a browser would use anyway.
            mime_type=mime_for(found.filename) or "application/octet-stream",
            size_bytes=found.size_bytes,
            # When the far end last changed it — the only date that means
            # anything for a file this system did not create.
            created_at=datetime.fromtimestamp(found.modified_at, tz=UTC),
            status="ready" if found.ready else "processing",
            source_name=names.get(uuid.UUID(found.source_id), "Подключённый источник"),
            removable=False,
        )
        for found in indexed.values()
        if found.document_id not in uploaded
    )

    documents.sort(key=lambda document: document.created_at, reverse=True)
    return documents


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> DocumentOut:
    filename = PurePosixPath(file.filename or "").name
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустое имя файла")

    # Trust our own suffix mapping rather than the client's Content-Type.
    mime_type = mime_for(filename)
    if mime_type is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Поддерживаются: {HUMAN_READABLE}"
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
    key = build_key(user_id, source.id, document_id, filename)

    # S3 first. A row without an object would sit at "processing" forever with
    # nothing to fix it. The other way round — an object the commit below never
    # records — the file is indexed and searchable but has no row to open, so a
    # citation to it fails to resolve. That is the better failure of the two:
    # the answer is still right, and re-uploading the file repairs it.
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
        source_name=source.name,
        removable=True,
    )


@router.get("/{document_id}/link")
async def document_link(
    document_id: uuid.UUID, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> dict[str, str]:
    """Where to open the original.

    One endpoint for both populations, so a citation chip does not have to know
    which kind of document it points at: an upload gets a presigned link to our
    bucket, a document from a connected source gets the service's own page for
    it. The index is consulted only when there is no row, which is the only case
    that needs it.
    """
    document = (
        await session.execute(
            select(Document).where(Document.id == document_id, Document.user_id == user_id)
        )
    ).scalar_one_or_none()
    if document is not None:
        return {"url": await storage.presigned_url(settings, document.s3_key)}

    found = (await _indexed(settings, user_id)).get(str(document_id))
    if found is None or not found.web_url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Документ не найден")
    return {"url": found.web_url}


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(
    document_id: uuid.UUID, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> None:
    """Delete an upload.

    Only an upload: a document from a connected source is not ours to delete,
    and removing it here would achieve nothing anyway — the next poll would
    bring it straight back. It returns 404 for the same reason a document
    belonging to somebody else does.
    """
    document = (
        await session.execute(
            select(Document).where(Document.id == document_id, Document.user_id == user_id)
        )
    ).scalar_one_or_none()
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Документ не найден")

    # Delete the object first: the indexer drops the chunks when the object
    # disappears, so this is what actually removes it from search.
    await storage.delete(settings, document.s3_key)
    await session.execute(sql_delete(Document).where(Document.id == document.id))
    await session.commit()
