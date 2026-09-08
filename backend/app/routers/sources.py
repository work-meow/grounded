"""The documents a user has, wherever they came from.

Two populations, one list. An upload is a row in ``documents`` and an object in
our bucket; a document from a connected source is neither — nothing here ever
fetched it, so the index is the only place that has seen it. Both are read back
from the index, which is what makes "is it searchable yet?" answerable at all.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from rag_shared.connectors import Kind
from rag_shared.doc_key import build_key
from rag_shared.formats import HUMAN_READABLE, mime_for
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import anchor, connectors, pdf, retriever, storage
from app.config import Settings
from app.deps import SessionDep, SettingsDep, UserDep
from app.models import Document, Source

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["sources"])


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    #: Unknown for a Notion page, which is not a file anywhere.
    size_bytes: int | None
    created_at: datetime
    status: str  # "processing" | "ready"
    source_id: uuid.UUID
    source_name: str
    #: Whether deleting it here means anything. An upload is ours to remove; a
    #: document from a connected source is removed where it lives, and would
    #: come back on the next poll.
    removable: bool
    #: False for a PDF with no text in it — a scan, indexed and findable by
    #: nothing. Null means no claim: another format, an older upload, or a
    #: document from a connected source, whose bytes never passed through here.
    text_layer: bool | None = None


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

    About 85 ms, measured on the deployment box. It used to be a flat 1.48 s:
    Pathway answers a query only once its dataflow has advanced past it, and the
    frontier advances no faster than the slowest committing input connector,
    which defaulted to a tick every 1500 ms. The indexer now commits every
    100 ms (see COMMIT_INTERVAL_MS there), and this call stopped being something
    to design around.

    It is not cached here, and would not be even if it were slow. Readiness has
    exactly one home — the indexer — and a cache would let this API claim a
    document is searchable while the process that answers searches disagrees.
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
    #
    # Built with a comprehension, not dict(result): a SQLAlchemy Result has a
    # keys() method, so dict() takes it for a mapping and starts subscripting
    # it, which a Result does not support.
    names = {
        row.id: row.name
        for row in (
            await session.execute(select(Source.id, Source.name).where(Source.user_id == user_id))
        ).all()
    }
    indexed = await _indexed(settings, user_id)

    documents = [
        DocumentOut(
            id=document.id,
            filename=document.filename,
            mime_type=document.mime_type,
            size_bytes=document.size_bytes,
            created_at=document.created_at,
            status="ready" if str(document.id) in indexed else "processing",
            source_id=document.source_id,
            source_name=names.get(document.source_id, "Загруженные файлы"),
            removable=True,
            text_layer=document.text_layer,
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
            source_id=uuid.UUID(found.source_id),
            source_name=names.get(uuid.UUID(found.source_id), "Подключённый источник"),
            removable=False,
        )
        for found in indexed.values()
        if found.document_id not in uploaded
    )

    documents.sort(key=lambda document: document.created_at, reverse=True)
    return documents


class Changed(BaseModel):
    """One source and what moved in it."""

    source_id: uuid.UUID
    source_name: str
    documents: list[DocumentOut]


class ChangesOut(BaseModel):
    days: int
    #: Newest first, and only sources that had something change.
    sources: list[Changed]
    total: int


@router.get("/changes")
async def changes(
    user_id: UserDep,
    session: SessionDep,
    settings: SettingsDep,
    days: Annotated[int, Query(ge=1, le=365)] = 7,
) -> ChangesOut:
    """What was added or changed in the last few days, by source.

    A knowledge base fed by Notion, Drive and a shared folder changes without
    anybody watching, and the answer to "что я пропустил" was previously to
    scroll a list of everything sorted by date and remember where you stopped.

    A page and not a schedule: there is no job runner in this deployment, and
    introducing one so that a summary could arrive by itself would be a large
    piece of machinery for a question somebody asks on a Monday.
    """
    since = retriever.since(days)
    documents = [
        document
        for document in await list_documents(user_id=user_id, session=session, settings=settings)
        if since is None or document.created_at.timestamp() >= since
    ]

    grouped: dict[uuid.UUID, Changed] = {}
    for document in documents:
        group = grouped.get(document.source_id)
        if group is None:
            group = Changed(
                source_id=document.source_id,
                source_name=document.source_name,
                documents=[],
            )
            grouped[document.source_id] = group
        group.documents.append(document)

    # Sources ordered by their most recent change, so the one that moved today
    # is at the top rather than whichever happens to be first alphabetically.
    order = sorted(
        grouped.values(),
        key=lambda group: max(document.created_at for document in group.documents),
        reverse=True,
    )
    return ChangesOut(days=days, sources=order, total=len(documents))


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

    # Before the upload is recorded, because the answer goes into the row — and
    # while the person is still standing here, which is the whole reason this
    # happens at upload rather than after indexing.
    text_layer = await _text_layer(mime_type, body)

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
        text_layer=text_layer,
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
        source_id=source.id,
        source_name=source.name,
        removable=True,
        text_layer=document.text_layer,
    )


async def _text_layer(mime_type: str, body: bytes) -> bool | None:
    """Whether this upload has text in it, for the formats where that is a question.

    Off the event loop: reading a PDF is C code that can take a second or two on
    a large one, and this runs in the process serving every other request.
    """
    if mime_type != "application/pdf":
        return None
    return await asyncio.to_thread(pdf.has_text_layer, body)


@router.get("/{document_id}/link")
async def document_link(
    document_id: uuid.UUID,
    user_id: UserDep,
    session: SessionDep,
    settings: SettingsDep,
    page: Annotated[int | None, Query(ge=1, le=10_000)] = None,
    quote: Annotated[str | None, Query(max_length=400)] = None,
) -> dict[str, str]:
    """Where to open the original, and where in it.

    One endpoint for both populations, so a citation chip does not have to know
    which kind of document it points at: an upload gets a presigned link to our
    bucket, a document from a connected source gets the service's own page for
    it. The index is consulted only when there is no row, which is the only case
    that needs it.

    ``page`` and ``quote`` come from the citation the reader clicked and turn
    into a fragment on the end of the url — ``#page=3`` for a PDF, a text
    fragment for anything a browser renders as text. Both are hints: a viewer
    that does not understand one ignores it, and the link is what it was
    before. See :mod:`app.anchor`.
    """
    document = (
        await session.execute(
            select(Document).where(Document.id == document_id, Document.user_id == user_id)
        )
    ).scalar_one_or_none()
    if document is not None:
        signed = await storage.presigned_url(settings, document.s3_key)
        return {"url": _anchored(signed, document.mime_type, page, quote)}

    found = (await _indexed(settings, user_id)).get(str(document_id))
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Документ не найден")
    if found.web_url:
        # Somebody else's page: it has its own idea of where things are, and a
        # fragment we invented would at best do nothing.
        return {"url": found.web_url}
    if (signed := await _signed(settings, session, user_id, found)) is not None:
        return {"url": _anchored(signed, mime_for(found.filename), page, quote)}
    raise HTTPException(
        status.HTTP_404_NOT_FOUND, "У этого документа нет ссылки, которую можно открыть"
    )


def _anchored(url: str, mime_type: str | None, page: int | None, quote: str | None) -> str:
    """The link with a place in it, if this format has a way of saying one.

    A page number for a PDF, since that is the only format here paginated at
    all; a text fragment for the formats a browser renders as text. Office
    files get neither: the browser downloads them and hands them to an
    application that never saw the url.
    """
    if mime_type == "application/pdf":
        return anchor.at_page(url, page)
    if mime_type in ("text/plain", "text/markdown") and quote:
        return anchor.at_quote(url, quote)
    return url


async def _signed(
    settings: Settings,
    session: AsyncSession,
    user_id: uuid.UUID,
    found: retriever.IndexedDocument,
) -> str | None:
    """A link to an object in a connected store, or None if there is none to make.

    An object store has no page to send anyone to, so the alternative to signing
    is a citation chip that cannot be clicked. Signing needs the source's own
    credentials, which is the one place in this API that opens a sealed config
    for something other than writing the manifest — and it makes no request of
    its own, so the endpoint being user-supplied is not a fetch this server
    performs.
    """
    if not found.external_id:
        return None
    sealer = connectors.sealer(settings)
    if sealer is None:
        return None

    source = (
        await session.execute(
            select(Source).where(
                Source.id == uuid.UUID(found.source_id),
                Source.user_id == user_id,
                Source.kind == Kind.S3.value,
            )
        )
    ).scalar_one_or_none()
    if source is None or not source.sealed_config:
        return None

    try:
        config = sealer.unseal(source.sealed_config)
        url = await storage.foreign_presigned_url(config, found.external_id)
    except Exception:
        # A revoked key, an endpoint that has moved, a config sealed under an
        # older secret. None of that is worth a failed request for a link.
        logger.exception("could not sign a link for document %s", found.document_id)
        return None
    # The same rule every other link goes through: this one is built from a
    # user-supplied endpoint, and it ends up in window.open.
    return retriever.openable(url)


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
