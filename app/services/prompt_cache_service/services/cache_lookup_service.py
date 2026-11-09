
"""Cache lookup orchestration with Redis acceleration and PostgreSQL fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.logging_setup import get_logger

from ..config.settings import PromptCacheSettings
from ..repositories.cache_repository import CacheRepository
from ..schemas.cache import CachePayload, CacheResult, CacheLookupResponse
from ..schemas.enums import CacheValidity
from .cache_signature import CacheSignatureService
from .cache_validity import classify_cache_validity


_LOGGER = "prompt_cache_service.services.cache_lookup_service"
logger = get_logger(_LOGGER)


class RefreshServiceProtocol(Protocol):
    """Contract required from the cache refresh service."""

    async def start_or_reuse_refresh(
        self,
        session: AsyncSession,
        payload: CachePayload,
    ) -> Any:
        """Create or reuse a background cache-refresh job."""
        ...


class CacheLookupService:
    """Find and return reusable Gemini cached content."""

    REDIS_KEY_PREFIX = "prompt_cache:signature:"

    def __init__(
        self,
        cache_repository: CacheRepository,
        signature_service: CacheSignatureService,
        refresh_service: RefreshServiceProtocol,
        redis_client: Redis,
        settings: PromptCacheSettings,
    ) -> None:
        self.cache_repository = cache_repository
        self.signature_service = signature_service
        self.refresh_service = refresh_service
        self.redis_client = redis_client
        self.settings = settings

    async def lookup(
        self,
        session: AsyncSession,
        payload: CachePayload,
    ) -> CacheLookupResponse:
        """Return an existing usable cache, or None if unavailable."""

        signature = self.signature_service.calculate(payload)
        redis_key = f"{self.REDIS_KEY_PREFIX}{signature}"

        # 1. Try Redis first.
        try:
            redis_value = await self.redis_client.get(redis_key)
        except RedisError:
            redis_value = None
            logger.warning(
                "Redis unavailable during cache lookup; "
                "falling back to PostgreSQL."
            )

        if redis_value is not None:
            entry = self._deserialize_redis_entry(redis_value)

            if entry is None:
                logger.warning(
                    "Invalid Redis cache metadata; "
                    "falling back to PostgreSQL."
                )
                await self._delete_invalid_redis_entry(redis_key)
            else:
                validity = self._classify(entry["expires_at"])

                if validity == CacheValidity.VALID:
                    return CacheLookupResponse(
                        cache=self._build_cache_result(entry)
                    )

                if validity == CacheValidity.EXPIRING:
                    await self.refresh_service.start_or_reuse_refresh(
                        session,
                        payload,
                    )
                    return CacheLookupResponse(
                        cache=self._build_cache_result(entry)
                    )

                # Cache has expired. Continue to PostgreSQL.
                await self._delete_invalid_redis_entry(redis_key)

        # 2. PostgreSQL fallback.
        cache = await self.cache_repository.get_by_signature(
            session,
            signature,
        )

        if cache is None or cache.result is None:
            return CacheLookupResponse(cache=None)

        if cache.expires_at is None:
            return CacheLookupResponse(cache=None)

        validity = self._classify(cache.expires_at)

        if validity == CacheValidity.EXPIRED:
            return CacheLookupResponse(cache=None)

        result = cache.result
        cache_name = result.get("cache_name")

        if not isinstance(cache_name, str) or not cache_name.strip():
            logger.warning(
                "Database cache result has invalid cache_name: "
                "cache_id=%s",
                cache.id,
            )
            return CacheLookupResponse(cache=None)

        entry = {
            "prompt_cache_id": str(cache.id),
            "cache_name": cache_name,
            "expires_at": cache.expires_at,
        }

        # 3. Repopulate Redis.
        await self._populate_redis(redis_key, entry)

        # 4. Return the usable cache and trigger refresh if necessary.
        if validity == CacheValidity.EXPIRING:
            await self.refresh_service.start_or_reuse_refresh(
                session,
                payload,
            )

        return CacheLookupResponse(
            cache=self._build_cache_result(entry)
        )

    def _classify(self, expires_at: datetime) -> CacheValidity:
        return classify_cache_validity(
            expires_at,
            refresh_threshold_seconds=(
                self.settings.CACHE_REFRESH_THRESHOLD_SECONDS
            ),
        )

    @staticmethod
    def _deserialize_redis_entry(
        value: str | bytes,
    ) -> dict[str, Any] | None:
        try:
            data = json.loads(value)

            if not isinstance(data, dict):
                return None

            prompt_cache_id = UUID(data["prompt_cache_id"])
            cache_name = data["cache_name"]
            expires_at = datetime.fromisoformat(data["expires_at"])

            if not isinstance(cache_name, str) or not cache_name.strip():
                return None

            if (
                expires_at.tzinfo is None
                or expires_at.utcoffset() is None
            ):
                return None

            return {
                "prompt_cache_id": str(prompt_cache_id),
                "cache_name": cache_name,
                "expires_at": expires_at,
            }

        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    async def _populate_redis(
        self,
        redis_key: str,
        entry: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc)
        expires_at = entry["expires_at"]

        remaining_seconds = (expires_at - now).total_seconds()

        ttl = int(
            remaining_seconds
            - self.settings.CACHE_REDIS_TTL_SAFETY_SECONDS
        )

        if ttl <= 0:
            return

        redis_value = json.dumps(
            {
                "prompt_cache_id": entry["prompt_cache_id"],
                "cache_name": entry["cache_name"],
                "expires_at": expires_at.isoformat(),
            },
            separators=(",", ":"),
        )

        try:
            await self.redis_client.set(
                redis_key,
                redis_value,
                ex=ttl,
            )
        except RedisError:
            logger.warning(
                "Could not populate Redis cache metadata; "
                "PostgreSQL remains the source of truth."
            )

    async def _delete_invalid_redis_entry(
        self,
        redis_key: str,
    ) -> None:
        try:
            await self.redis_client.delete(redis_key)
        except RedisError:
            logger.warning(
                "Could not delete invalid Redis cache metadata."
            )

    @staticmethod
    def _build_cache_result(
        entry: dict[str, Any],
    ) -> CacheResult:
        return CacheResult(
            cache_name=entry["cache_name"],
            reused=True,
            expires_at=entry["expires_at"],
        )
