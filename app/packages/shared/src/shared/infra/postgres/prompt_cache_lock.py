"""SQLAlchemy mapping for distributed Prompt Cache worker leases."""

from __future__ import annotations

import uuid
from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, DateTime, String, func, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class PromptCacheLock(Base):
    """Persist worker ownership and fencing state for a cache signature.

    The row must be retained after release. Clearing ``owner_token`` and
    expiring the lease must not reset ``fencing_token``; each successful new
    acquisition increments it so stale workers cannot persist results.
    """

    __tablename__ = "prompt_cache_locks"

    signature: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    owner_token: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )
    fencing_token: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
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
