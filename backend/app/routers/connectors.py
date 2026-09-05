"""Connecting and disconnecting a source.

Three endpoints and one rule that shapes all of them: a credential goes in and
never comes back out. There is no read endpoint for a configuration, no field
echoed in an error, and nothing about a connector in a log line beyond its id —
because the only reason to read one back would be to show it to somebody, and
the person who typed it already has it.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from rag_shared.connectors import REQUIRED_FIELDS, Kind
from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select

from app import connectors
from app.deps import SessionDep, SettingsDep, UserDep
from app.models import Source

router = APIRouter(prefix="/api/connectors", tags=["connectors"])

_KINDS = {kind.value for kind in Kind}


class ConnectorOut(BaseModel):
    id: uuid.UUID
    kind: Kind
    name: str
    created_at: datetime
    #: What the indexer last reported about it. "unknown" until it has looked
    #: once — never "ok", because a source nobody has read is not a working one.
    status: connectors.Status = connectors.Status.UNKNOWN
    #: Why it is not working, in words the user can act on. Empty otherwise.
    problem: str = ""
    #: When the indexer last looked, and how many entries it saw then. Null
    #: until it has; zero documents is a diagnosis, not a missing value.
    checked_at: datetime | None = None
    documents: int | None = None


class ConnectorsOut(BaseModel):
    """The list, plus what the UI needs to explain how to add one."""

    sources: list[ConnectorOut]
    #: False when this deployment has no SECRETS_KEY, so the UI can say why
    #: rather than offering a form that will fail.
    enabled: bool
    #: Whom to share a Drive folder with. Empty when Drive is not set up here.
    gdrive_service_account_email: str
    #: Which fields each kind needs, so the form and the API cannot disagree.
    required_fields: dict[Kind, list[str]]


class ConnectorIn(BaseModel):
    kind: Kind
    name: str = Field(min_length=1, max_length=200)
    # Bounded because it is parsed before anything looks at it: clean_config
    # keeps only the fields the kind needs, but by then a body with a hundred
    # thousand keys has already been turned into a dict. No kind needs more
    # than four.
    config: dict[str, str] = Field(max_length=16)


@router.get("")
async def list_connectors(
    user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> ConnectorsOut:
    rows = (
        (
            await session.execute(
                select(Source)
                .where(Source.user_id == user_id, Source.sealed_config.is_not(None))
                .order_by(Source.created_at)
            )
        )
        .scalars()
        .all()
    )
    # One small read from the bucket, alongside the query. The alternative — a
    # column this API keeps current — would mean the indexer reaching into the
    # database, which it has never had to do for anything else.
    health = await connectors.statuses(settings)
    return ConnectorsOut(
        sources=[
            _out(row, health.get(row.id, connectors.UNCHECKED))
            for row in rows
            # A kind this build no longer knows would break the response model
            # for every other source in the list.
            if row.kind in _KINDS
        ],
        enabled=connectors.sealer(settings) is not None,
        gdrive_service_account_email=settings.gdrive_service_account_email,
        required_fields={kind: list(fields) for kind, fields in REQUIRED_FIELDS.items()},
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def add_connector(
    body: ConnectorIn, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> ConnectorOut:
    sealer = connectors.sealer(settings)
    if sealer is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Подключение источников не настроено на этом сервере",
        )

    # Drive is the one kind the deployment has to have set up: the key lives in
    # the indexer, and this address is what a user shares their folder with.
    # Without it the source would be accepted, never index anything, and give
    # nobody a reason why.
    if body.kind is Kind.GDRIVE and not settings.gdrive_service_account_email:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Google Drive не настроен на этом сервере",
        )

    used = await session.scalar(
        select(func.count())
        .select_from(Source)
        .where(Source.user_id == user_id, Source.sealed_config.is_not(None))
    )
    if (used or 0) >= connectors.MAX_SOURCES_PER_USER:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Больше {connectors.MAX_SOURCES_PER_USER} источников подключить нельзя",
        )

    try:
        config = connectors.clean_config(body.kind, body.config)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    source = Source(
        user_id=user_id,
        kind=body.kind.value,
        name=body.name.strip(),
        sealed_config=sealer.seal(config),
    )
    session.add(source)
    await session.commit()
    await session.refresh(source)

    # After the commit: the manifest must never name a source the database does
    # not have. If this fails the row stays, unpublished, and the next change
    # republishes it — the user sees a source that is not indexing yet, which is
    # recoverable, rather than an index rebuilt around a row that vanished.
    await connectors.publish(settings, session)

    # Unchecked, and it says so: the indexer has not seen this source yet and
    # will not until it restarts around the new manifest.
    return _out(source, connectors.UNCHECKED)


def _out(row: Source, health: connectors.SourceStatus) -> ConnectorOut:
    return ConnectorOut(
        id=row.id,
        kind=Kind(row.kind),
        name=row.name,
        created_at=row.created_at,
        status=health.status,
        problem=health.problem,
        checked_at=health.checked_at,
        documents=health.documents,
    )


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_connector(
    source_id: uuid.UUID, user_id: UserDep, session: SessionDep, settings: SettingsDep
) -> None:
    source = (
        await session.execute(
            select(Source).where(
                Source.id == source_id,
                Source.user_id == user_id,
                Source.sealed_config.is_not(None),
            )
        )
    ).scalar_one_or_none()
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Источник не найден")

    await session.execute(sql_delete(Source).where(Source.id == source.id))
    await session.commit()
    # Only now, so that a failed publish leaves the index one restart behind
    # rather than pointing at a source that is gone.
    await connectors.publish(settings, session)
