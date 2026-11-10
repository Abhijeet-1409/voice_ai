
"""Coordinate stale-cache refresh requests."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas.cache import CachePayload
from ..schemas.enums import RefreshStatus
from .cache_job_service import CacheJobService


@dataclass(frozen=True)
class RefreshStartResult:
    """Internal result describing a cache-refresh request."""

    status: RefreshStatus
    job_id: UUID


class CacheRefreshService:
    """Start or reuse a cache-generation job for an expiring cache."""

    def __init__(
        self,
        cache_job_service: CacheJobService,
    ) -> None:
        self.cache_job_service = cache_job_service

    async def start_or_reuse_refresh(
        self,
        session: AsyncSession,
        payload: CachePayload,
    ) -> RefreshStartResult:
        """Start a refresh job or reuse an existing active job."""

        job_result = (
            await self.cache_job_service.create_or_get_active_job(
                session,
                payload,
            )
        )

        if job_result.created:
            return RefreshStartResult(
                status=RefreshStatus.STARTED,
                job_id=job_result.job.id,
            )

        return RefreshStartResult(
            status=RefreshStatus.ALREADY_RUNNING,
            job_id=job_result.job.id,
        )
