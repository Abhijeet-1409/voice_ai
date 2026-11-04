"""SQLAlchemy mapping for append-only Prompt Cache job status history."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class PromptCacheJobStatusHistory(Base):
    """Record one status transition for a Prompt Cache job.

    Rows are append-only from application code. The largest identity value
    for a given job determines its current status.
    """

    __tablename__ = "prompt_cache_job_status_history"
    __table_args__ = (
        Index(
            "ix_prompt_cache_job_status_history_job_id_id",
            "job_id",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    job_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("prompt_cache_jobs.id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
