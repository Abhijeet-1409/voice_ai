
"""Persistence operations for Prompt Cache records."""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import PromptCacheType
from shared.infra.postgres import PromptCache
from shared.logging_setup import get_logger

from ..schemas.jobs import CacheCreationResult


_LOGGER = "prompt_cache_service.repositories.cache_repositories"
logger = get_logger(_LOGGER)


class CacheRepository:
    """Database operations for signature-based prompt caches.

    Transaction ownership belongs to the calling service.
    This repository never commits or rolls back a transaction.
    """

    async def get_by_signature(
        self,
        session: AsyncSession,
        signature: str,
        *,
        for_update: bool = False,
    ) -> PromptCache | None:
        """Retrieve a cache using its unique signature."""

        stmt = select(PromptCache).where(
            PromptCache.signature == signature
        )

        if for_update:
            stmt = stmt.with_for_update()

        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(
        self,
        session: AsyncSession,
        cache_id: UUID,
    ) -> PromptCache | None:
        """Retrieve a cache using its primary key."""

        stmt = select(PromptCache).where(
            PromptCache.id == cache_id
        )

        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def ensure_row_and_lock(
        self,
        session: AsyncSession,
        *,
        signature: str,
        prompt_cache_type: PromptCacheType,
        payload: dict[str, Any],
    ) -> PromptCache:
        """Ensure a cache row exists and acquire its PostgreSQL row lock.

        Uses INSERT ... ON CONFLICT DO NOTHING to avoid duplicate rows
        for concurrent requests with the same signature.

        The caller must execute this method inside a transaction and
        retain the lock until its active-job check and job creation finish.
        """

        prompt_cache_type_value = (
            prompt_cache_type.value
            if isinstance(prompt_cache_type, PromptCacheType)
            else str(prompt_cache_type)
        )

        insert_stmt = (
            pg_insert(PromptCache)
            .values(
                id=uuid.uuid4(),
                signature=signature,
                prompt_cache_type=prompt_cache_type_value,
                payload=payload,
            )
            .on_conflict_do_nothing(
                index_elements=[PromptCache.signature]
            )
        )

        await session.execute(insert_stmt)

        select_stmt = (
            select(PromptCache)
            .where(PromptCache.signature == signature)
            .with_for_update()
        )

        result = await session.execute(select_stmt)
        cache = result.scalar_one_or_none()

        if cache is None:
            logger.error(
                "Failed to retrieve cache after insert-or-ignore: "
                "signature=%s",
                signature,
            )
            raise RuntimeError(
                "Failed to retrieve PromptCache row after "
                "INSERT ... ON CONFLICT."
            )

        logger.debug(
            "Prompt Cache row ensured and locked: cache_id=%s",
            cache.id,
        )

        return cache

    async def save_successful_result(
        self,
        session: AsyncSession,
        *,
        cache_id: UUID,
        result: CacheCreationResult,
    ) -> PromptCache:
        """Persist a successfully created Gemini cached-content resource.

        The existing successful result is replaced only after the
        provider returns a valid result.

        The cache row is locked to prevent concurrent result replacement.
        """

        stmt = (
            select(PromptCache)
            .where(PromptCache.id == cache_id)
            .with_for_update()
        )

        query_result = await session.execute(stmt)
        cache = query_result.scalar_one_or_none()

        if cache is None:
            logger.error(
                "Prompt Cache record not found: cache_id=%s",
                cache_id,
            )
            raise LookupError(
                f"PromptCache record not found: {cache_id}"
            )

        cache.result = result.model_dump(mode="json")
        cache.expires_at = result.expires_at

        # The ORM model's onupdate=func.now() handles updated_at.

        await session.flush()

        logger.info(
            "Successfully updated Prompt Cache result: cache_id=%s "
            "cache_name=%s expires_at=%s",
            cache_id,
            result.cache_name,
            result.expires_at,
        )

        return cache
