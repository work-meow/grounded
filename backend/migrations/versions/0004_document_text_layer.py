"""a scanned pdf is called a scan

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-05

Null for every existing row, and null means "nothing was checked" rather than
"no text" — the bytes are in the bucket but not in this process, and inventing
an answer for them would put a wrong warning on documents that are fine.
Uploads from here on carry the real value.
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("text_layer", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "text_layer")
