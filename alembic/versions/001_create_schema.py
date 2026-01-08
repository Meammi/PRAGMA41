"""create users, results, images

Revision ID: 001_create_schema
Revises: 
Create Date: 2025-02-14 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_create_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "users",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("fname", sa.Text(), nullable=False),
        sa.Column("lname", sa.Text(), nullable=False),
        sa.Column("phone_number", sa.Text(), nullable=False),
        sa.Column("birth_day", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "results",
        sa.Column("result_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("medical_report", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    op.create_table(
        "images",
        sa.Column("image_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("result_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("results.result_id", ondelete="CASCADE"), nullable=False),
        sa.Column("org_img", sa.Text(), nullable=False),
        sa.Column("result_img", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("images")
    op.drop_table("results")
    op.drop_table("users")
