"""SQLAlchemy mapping for signature-based Gemini prompt caches.

The legacy persistence helpers at the bottom of this module are temporary
compatibility shims. They deliberately fail with an actionable message because
their version-based arguments cannot identify a cache in the new signature-
based schema. Remove those shims after all legacy callers have been migrated.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.config import PromptCacheType

from .base import Base


class LegacyPromptCacheAPIError(RuntimeError):
    """Raised when a caller uses the retired version-based PromptCache API."""


class PromptCache(Base):
    """Persist the identity and Gemini result of a reusable prompt cache.

    ``signature`` is the SHA-256 digest calculated by the Prompt Cache
    Service from the canonical cache payload. ``payload`` contains the
    cache-affecting input used to produce that signature. ``result`` remains
    null until the provider successfully creates a cached-content resource.

    ``prompt_cache_type`` is stored as the string value of the existing shared
    ``PromptCacheType`` enum. This model intentionally does not define a second
    enum or own database sessions/transactions.
    """

    __tablename__ = "prompt_caches"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    signature: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    prompt_cache_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
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


def _raise_legacy_api_error() -> None:
    raise LegacyPromptCacheAPIError(
        "The version-based PromptCache helper API is no longer compatible "
        "with the signature-based prompt_caches schema. Migrate this caller "
        "to the Prompt Cache Service before enabling the schema migration."
    )


# Temporary shims: retain the old import names until the repository-wide
# caller migration is complete. Do not use these functions for new code.
def save_prompt_cache_sync(
    prompt_type: PromptCacheType,
    vertex_cache_name: str,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
    created_at: datetime,
    expires_at: datetime,
) -> PromptCache:
    """Deprecated version-based API; retained temporarily for import stability."""
    _raise_legacy_api_error()


def get_prompt_cache_sync(
    prompt_type: PromptCacheType,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
) -> PromptCache | None:
    """Deprecated version-based API; retained temporarily for import stability."""
    _raise_legacy_api_error()


async def save_prompt_cache_async(
    prompt_type: PromptCacheType,
    vertex_cache_name: str,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
    created_at: datetime,
    expires_at: datetime,
) -> PromptCache:
    """Deprecated async version-based API; retained temporarily for import stability."""
    _raise_legacy_api_error()


async def get_prompt_cache_async(
    prompt_type: PromptCacheType,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
) -> PromptCache | None:
    """Deprecated async version-based API; retained temporarily for import stability."""
    _raise_legacy_api_error()
