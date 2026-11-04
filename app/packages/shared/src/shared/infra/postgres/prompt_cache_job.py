"""SQLAlchemy mapping for Prompt Cache creation jobs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class PromptCacheJob(Base):
    """Represent one request to create or refresh a Gemini prompt cache.

    Job status is deliberately not stored on this row. The current status is
    derived from the latest row in ``prompt_cache_job_status_history``.
    """

    __tablename__ = "prompt_cache_jobs"
    __table_args__ = (
        Index(
            "ix_prompt_cache_jobs_cache_created",
            "prompt_cache_id",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    prompt_cache_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("prompt_caches.id"),
        nullable=False,
    )
    celery_task_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    deadline_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
