
"""Business logic for cache-generation job creation and status retrieval."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.logging_setup import get_logger

from ..config.settings import PromptCacheSettings
from ..core.exceptions import ServiceError
from ..repositories.cache_repository import CacheRepository
from ..repositories.job_repository import JobRepository
from ..repositories.lock_repository import LockRepository
from ..repositories.outbox_repository import OutboxRepository
from ..schemas.cache import CachePayload, CacheResult
from ..schemas.enums import CacheJobErrorCode, JobStatus
from ..schemas.jobs import (
    CacheJobError,
    CacheJobStatusResponse,
)
from .cache_signature import CacheSignatureService


_LOGGER = "prompt_cache_service.services.cache_job_service"
logger = get_logger(_LOGGER)


@dataclass(frozen=True)
class CreateOrGetJobResult:
    """Result of creating a new job or reusing an existing active job."""

    job: object
    status: JobStatus
    created: bool


class CacheJobService:
    """Coordinate cache-generation job creation and status retrieval."""

    OUTBOX_EVENT_TYPE = "PROMPT_CACHE_CREATION_REQUESTED"

    def __init__(
        self,
        cache_repository: CacheRepository,
        job_repository: JobRepository,
        lock_repository: LockRepository,
        outbox_repository: OutboxRepository,
        signature_service: CacheSignatureService,
        settings: PromptCacheSettings,
    ) -> None:
        self.cache_repository = cache_repository
        self.job_repository = job_repository
        self.lock_repository = lock_repository
        self.outbox_repository = outbox_repository
        self.signature_service = signature_service
        self.settings = settings

    @asynccontextmanager
    async def _transaction(
        self,
        session: AsyncSession,
    ) -> AsyncIterator[None]:
        """Use a transaction or savepoint depending on session ownership.

        A session may already have an active transaction because an earlier
        database read triggered SQLAlchemy autobegin.

        In that case, use a savepoint rather than attempting to begin a
        second top-level transaction.
        """
        if session.in_transaction():
            async with session.begin_nested():
                yield
        else:
            async with session.begin():
                yield

    async def _get_database_time(
        self,
        session: AsyncSession,
    ):
        """Read PostgreSQL's clock instead of trusting application time."""
        result = await session.execute(
            select(func.clock_timestamp())
        )
        return result.scalar_one()

    async def _calculate_deadline(
        self,
        session: AsyncSession,
    ):
        """Calculate the absolute job deadline using PostgreSQL time."""
        result = await session.execute(
            select(
                func.clock_timestamp()
                + func.make_interval(
                    0, 0, 0, 0, 0, 0,
                    self.settings.CACHE_JOB_TIMEOUT_SECONDS,
                )
            )
        )
        return result.scalar_one()

    async def create_or_get_active_job(
        self,
        session: AsyncSession,
        payload: CachePayload,
    ) -> CreateOrGetJobResult:
        """Create a cache-generation job or reuse an existing active job."""

        signature = self.signature_service.calculate(payload)

        # Convert the Pydantic model and enum fields into JSON-compatible
        # values before persisting them in PostgreSQL JSONB.
        payload_data = payload.model_dump(mode="json")

        async with self._transaction(session):
            # 1. Ensure the cache record exists and lock it.
            # This serializes concurrent job-creation requests for the
            # same signature.
            cache = await self.cache_repository.ensure_row_and_lock(
                session,
                signature=signature,
                prompt_cache_type=payload.prompt_cache_type,
                payload=payload_data,
            )

            # 2. Check whether an active job already exists.
            active_job = await self.job_repository.get_active_job_for_cache(
                session,
                cache.id,
            )

            database_now = await self._get_database_time(session)

            if active_job is not None:
                job, status = active_job

                # An active job with a future deadline must be reused.
                # Never extend or reset its original deadline.
                if job.deadline_at > database_now:
                    return CreateOrGetJobResult(
                        job=job,
                        status=status,
                        created=False,
                    )

                # 3. The active job has exceeded its deadline.
                # Lock the job and verify its latest status again.
                locked_job = await self.job_repository.get_job(
                    session,
                    job.id,
                    for_update=True,
                )

                if locked_job is not None:
                    latest_status = (
                        await self.job_repository.get_latest_status(
                            session,
                            locked_job.id,
                        )
                    )

                    if (
                        latest_status
                        in (
                            JobStatus.PENDING,
                            JobStatus.RUNNING,
                            JobStatus.RETRYING,
                        )
                        and locked_job.deadline_at <= database_now
                    ):
                        deadline_error = CacheJobError(
                            code=(
                                CacheJobErrorCode
                                .PROMPT_CACHE_JOB_DEADLINE_EXCEEDED
                            ),
                            message=(
                                "Cache creation job exceeded its deadline."
                            ),
                            retryable=False,
                        )

                        # Persist terminal job information.
                        await self.job_repository.mark_terminal(
                            session,
                            job_id=locked_job.id,
                            status=JobStatus.FAILED,
                            error=deadline_error,
                        )

                        # Append-only status history.
                        await self.job_repository.append_status(
                            session,
                            job_id=locked_job.id,
                            status=JobStatus.FAILED,
                        )

                        # Invalidate any worker still holding this
                        # signature's lease/fencing ownership.
                        await self.lock_repository.invalidate(
                            session,
                            signature=signature,
                        )

                        logger.warning(
                            "Cache job exceeded its deadline: job_id=%s",
                            locked_job.id,
                        )

            # 4. No reusable active job exists.
            # Calculate a fresh absolute deadline from PostgreSQL time.
            deadline_at = await self._calculate_deadline(session)

            # 5. Insert the new job.
            job = await self.job_repository.create_job(
                session,
                cache_id=cache.id,
                deadline_at=deadline_at,
            )

            # 6. Append initial PENDING status.
            await self.job_repository.append_status(
                session,
                job_id=job.id,
                status=JobStatus.PENDING,
            )

            # 7. Insert the outbox event in the same transaction.
            # The publisher will dispatch it to Celery after commit.
            await self.outbox_repository.add_event(
                session,
                job_id=job.id,
                event_type=self.OUTBOX_EVENT_TYPE,
                payload={
                    "job_id": str(job.id),
                },
            )

            logger.info(
                "Created cache-generation job: job_id=%s",
                job.id,
            )

            return CreateOrGetJobResult(
                job=job,
                status=JobStatus.PENDING,
                created=True,
            )

    async def get_job_status(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> CacheJobStatusResponse:
        """Retrieve a job's latest status and terminal result if available."""

        job_data = await self.job_repository.get_job_and_status(
            session,
            job_id,
        )

        if job_data is None:
            raise ServiceError(
                code=CacheJobErrorCode.PROMPT_CACHE_JOB_NOT_FOUND.value,
                message="Cache-generation job was not found.",
                http_status=404,
                retryable=False,
            )

        job, status = job_data

        # Nonterminal responses contain only ID and status.
        if status in (
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.RETRYING,
        ):
            return CacheJobStatusResponse(
                id=str(job.id),
                status=status,
            )

        # A completed job must have a successful cache result.
        if status == JobStatus.COMPLETED:
            cache = await self.cache_repository.get_by_id(
                session,
                job.prompt_cache_id,
            )

            if (
                cache is None
                or not isinstance(cache.result, dict)
                or not cache.expires_at
            ):
                logger.error(
                    "Completed job has no valid cache result: job_id=%s",
                    job.id,
                )
                raise ServiceError(
                    code=(
                        CacheJobErrorCode
                        .PROMPT_CACHE_INTERNAL_ERROR.value
                    ),
                    message=(
                        "Completed job is missing its cache result."
                    ),
                    http_status=500,
                    retryable=False,
                )

            cache_name = cache.result.get("cache_name")

            if not isinstance(cache_name, str) or not cache_name.strip():
                raise ServiceError(
                    code=(
                        CacheJobErrorCode
                        .PROMPT_CACHE_INTERNAL_ERROR.value
                    ),
                    message=(
                        "Completed job contains an invalid cache result."
                    ),
                    http_status=500,
                    retryable=False,
                )

            return CacheJobStatusResponse(
                id=str(job.id),
                status=JobStatus.COMPLETED,
                result=CacheResult(
                    cache_name=cache_name,
                    reused=False,
                    expires_at=cache.expires_at,
                ),
            )

        # A failed job must have a structured final error.
        if status == JobStatus.FAILED:
            if not isinstance(job.error, dict):
                logger.error(
                    "Failed job has no structured error: job_id=%s",
                    job.id,
                )
                raise ServiceError(
                    code=(
                        CacheJobErrorCode
                        .PROMPT_CACHE_INTERNAL_ERROR.value
                    ),
                    message=(
                        "Failed job is missing its final error details."
                    ),
                    http_status=500,
                    retryable=False,
                )

            error = CacheJobError.model_validate(job.error)

            return CacheJobStatusResponse(
                id=str(job.id),
                status=JobStatus.FAILED,
                error=error,
            )

        # Defensive guard in case a new status is introduced later.
        raise ServiceError(
            code=CacheJobErrorCode.PROMPT_CACHE_INTERNAL_ERROR.value,
            message="Job has an unsupported lifecycle status.",
            http_status=500,
            retryable=False,
        )
