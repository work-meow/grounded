"""a source cannot be connected twice

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-05

The column is left null for every existing row: filling it in means opening a
sealed config, and a migration does not hold SECRETS_KEY. The API fills them in
the first time a user adds a source (app.connectors.backfill_fingerprints), and
until then nulls compare as distinct, so nothing existing is refused.
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sources", sa.Column("fingerprint", sa.String(length=64), nullable=True))
    op.create_unique_constraint("uq_sources_user_finger", "sources", ["user_id", "fingerprint"])


def downgrade() -> None:
    op.drop_constraint("uq_sources_user_finger", "sources", type_="unique")
    op.drop_column("sources", "fingerprint")
