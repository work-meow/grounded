"""a reader can say an answer was wrong

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-05

Null for every existing row, and null is "no opinion" rather than "neither
good nor bad" — an answer nobody rated is not an answer nobody minded.
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("rating", sa.SmallInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "rating")
