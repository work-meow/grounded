"""what a turn did, kept with its answer

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-07

The steps, the bill and the fragments the model was shown were all computed
already and thrown away when the turn ended. Keeping them makes two things
possible that were not: showing under an answer what it cost and what was
searched, and judging a change to retrieval against real traffic instead of
against thirteen questions that already pass.

Null for every existing row. Not backfillable — nothing recorded it.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("trace", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "trace")
