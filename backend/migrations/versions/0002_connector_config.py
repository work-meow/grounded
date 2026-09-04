"""a connected source keeps its credentials

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-04

Sealed, not encrypted here: the column holds a Fernet token produced by
rag_shared.crypto, so a database dump gives up the source list but not the
credentials that reach the services behind it.
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sources", sa.Column("sealed_config", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("sources", "sealed_config")
