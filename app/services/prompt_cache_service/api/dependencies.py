"""FastAPI dependencies for the Prompt Cache Service."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from shared.infra.postgres.database import get_async_sessionmaker
from shared.infra.redis.client import get_async_redis_client

from ..config.settings import get_prompt_cache_settings
from ..repositories.cache_repository import CacheRepository
from ..repositories.job_repository import JobRepository
from ..repositories.lock_repository import LockRepository
from ..repositories.outbox_repository import OutboxRepository
from ..services.cache_job_service import CacheJobService
from ..services.cache_lookup_service import CacheLookupService
from ..services.cache_refresh_service import CacheRefreshService
from ..services.cache_signature import CacheSignatureService


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Provide one AsyncSession per HTTP request."""

    session_factory = get_async_sessionmaker()

    async with session_factory() as session:
        yield session


def get_cache_job_service() -> CacheJobService:
    """Construct the cache job business service."""

    settings = get_prompt_cache_settings()

    return CacheJobService(
        cache_repository=CacheRepository(),
        job_repository=JobRepository(),
        lock_repository=LockRepository(),
        outbox_repository=OutboxRepository(),
        signature_service=CacheSignatureService(),
        settings=settings,
    )


def get_cache_lookup_service() -> CacheLookupService:
    """Construct the cache lookup business service."""

    settings = get_prompt_cache_settings()

    cache_job_service = get_cache_job_service()

    refresh_service = CacheRefreshService(
        cache_job_service=cache_job_service,
    )

    return CacheLookupService(
        cache_repository=CacheRepository(),
        signature_service=CacheSignatureService(),
        refresh_service=refresh_service,
        redis_client=get_async_redis_client(),
        settings=settings,
    )