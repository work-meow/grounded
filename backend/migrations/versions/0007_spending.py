"""a ceiling on what one token can spend in a day

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-08

One turn was bounded from the start — so many searches, so many model calls, a
wall clock — but the stream of turns was not, and a token never expires early.
A leaked one could spend real money in a loop with the bill as the first
notification.

Dollars and not requests: two questions can differ twenty times in cost, and
what is being protected is money. Numeric rather than float because this is
money that gets added to thousands of times.
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "spending",
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("turns", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(12, 8), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("spending")
