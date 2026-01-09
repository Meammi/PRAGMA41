"""add normal/abnormal counts to results

Revision ID: 002_add_result_counts
Revises: 001_create_schema
Create Date: 2025-02-14 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "002_add_result_counts"
down_revision = "001_create_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("results", sa.Column("normal_count", sa.Integer(), server_default="0", nullable=False))
    op.add_column("results", sa.Column("abnormal_count", sa.Integer(), server_default="0", nullable=False))


def downgrade() -> None:
    op.drop_column("results", "abnormal_count")
    op.drop_column("results", "normal_count")
