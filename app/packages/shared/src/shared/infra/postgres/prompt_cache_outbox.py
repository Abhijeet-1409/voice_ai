"""SQLAlchemy mapping for the transactional Prompt Cache outbox."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class PromptCacheOutbox(Base):
    """Persist events that must be published to the Celery broker.

    The publisher claims due, unpublished rows, commits the claim before
    contacting the broker, and marks the event published after successful
    delivery. Expired claims can be reclaimed after publisher crashes.
    """

    __tablename__ = "prompt_cache_outbox"
    __table_args__ = (
        Index(
            "ix_prompt_cache_outbox_unpublished_due",
            "available_at",
            "created_at",
            postgresql_where=text("published_at IS NULL"),
        ),
        Index(
            "ix_prompt_cache_outbox_claim_expires_at",
            "claim_expires_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("prompt_cache_jobs.id"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    claim_token: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    publish_attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
