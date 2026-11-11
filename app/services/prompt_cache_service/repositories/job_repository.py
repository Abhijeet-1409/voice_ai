
"""Persistence operations for Prompt Cache jobs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infra.postgres import (
    PromptCache,
    PromptCacheJob,
    PromptCacheJobStatusHistory,
    PromptCacheLock,
)
from shared.logging_setup import get_logger

from ..config.settings import get_prompt_cache_settings
from ..schemas.enums import JobStatus
from ..schemas.jobs import CacheJobError


_LOGGER = "prompt_cache_service.repositories.job_repositories"
logger = get_logger(_LOGGER)


class JobRepository:
    """Database operations for Prompt Cache jobs and their status history.

    The latest status-history row determines the current job status.

    Transaction ownership belongs to the calling service.
    This repository never commits or rolls back a transaction.
    """

    @staticmethod
    def _latest_status_subquery():
        """Return a subquery containing the latest history ID per job."""

        return (
            select(
                PromptCacheJobStatusHistory.job_id.label("job_id"),
                func.max(PromptCacheJobStatusHistory.id).label(
                    "latest_status_id"
                ),
            )
            .group_by(PromptCacheJobStatusHistory.job_id)
            .subquery()
        )

    @staticmethod
    def _to_job_status(status_value: str) -> JobStatus:
        """Convert a persisted status string into the JobStatus enum."""

        try:
            return JobStatus(status_value)
        except ValueError:
            logger.exception(
                "Unknown job status found in database: %s",
                status_value,
            )
            raise

    @staticmethod
    def _validate_limit(limit: int) -> None:
        """Prevent recovery queries from running without a valid batch limit."""

        if limit < 1:
            raise ValueError("limit must be greater than or equal to 1.")

    async def get_job(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        for_update: bool = False,
    ) -> PromptCacheJob | None:
        """Retrieve a job by ID, optionally acquiring a row lock."""

        stmt = select(PromptCacheJob).where(
            PromptCacheJob.id == job_id
        )

        if for_update:
            stmt = stmt.with_for_update()

        result = await session.execute(stmt)

        return result.scalar_one_or_none()

    async def get_latest_status(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> JobStatus | None:
        """Retrieve the latest status recorded for a job."""

        stmt = (
            select(PromptCacheJobStatusHistory.status)
            .where(PromptCacheJobStatusHistory.job_id == job_id)
            .order_by(PromptCacheJobStatusHistory.id.desc())
            .limit(1)
        )

        result = await session.execute(stmt)
        status_value = result.scalar_one_or_none()

        if status_value is None:
            return None

        return self._to_job_status(status_value)

    async def get_job_and_status(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> tuple[PromptCacheJob, JobStatus] | None:
        """Retrieve a job together with its latest status.

        Returns None when the job or its status-history record is missing.
        A missing initial status is logged as a data-integrity problem.
        """

        job = await self.get_job(session, job_id)

        if job is None:
            return None

        status = await self.get_latest_status(session, job_id)

        if status is None:
            logger.error(
                "Job %s exists but has no status-history record.",
                job_id,
            )
            return None

        return job, status

    async def get_active_job_for_cache(
        self,
        session: AsyncSession,
        cache_id: UUID,
    ) -> tuple[PromptCacheJob, JobStatus] | None:
        """Retrieve the newest active job belonging to a cache.

        Only the latest status-history record is considered. An old PENDING
        or RUNNING record must not make a terminal job appear active.
        """

        latest_status = self._latest_status_subquery()

        stmt = (
            select(
                PromptCacheJob,
                PromptCacheJobStatusHistory.status,
            )
            .join(
                latest_status,
                latest_status.c.job_id == PromptCacheJob.id,
            )
            .join(
                PromptCacheJobStatusHistory,
                PromptCacheJobStatusHistory.id
                == latest_status.c.latest_status_id,
            )
            .where(
                PromptCacheJob.prompt_cache_id == cache_id,
                PromptCacheJobStatusHistory.status.in_(
                    [
                        JobStatus.PENDING.value,
                        JobStatus.RUNNING.value,
                        JobStatus.RETRYING.value,
                    ]
                ),
            )
            .order_by(
                PromptCacheJob.created_at.desc(),
                PromptCacheJob.id.desc(),
            )
            .limit(1)
        )

        result = await session.execute(stmt)
        row = result.first()

        if row is None:
            return None

        job, status_value = row

        return job, self._to_job_status(status_value)

    async def create_job(
        self,
        session: AsyncSession,
        *,
        cache_id: UUID,
        deadline_at: datetime,
    ) -> PromptCacheJob:
        """Create a new cache-creation job.

        The caller must append the initial PENDING status and create the
        outbox event in the same transaction.
        """

        job = PromptCacheJob(
            prompt_cache_id=cache_id,
            attempt_count=0,
            deadline_at=deadline_at,
            next_attempt_at=None,
            error=None,
        )

        session.add(job)
        await session.flush()

        logger.info(
            "Created Prompt Cache job: job_id=%s cache_id=%s",
            job.id,
            cache_id,
        )

        return job

    async def append_status(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        status: JobStatus,
    ) -> PromptCacheJobStatusHistory:
        """Append a status transition without modifying previous records."""

        history = PromptCacheJobStatusHistory(
            job_id=job_id,
            status=status.value,
        )

        session.add(history)
        await session.flush()

        logger.info(
            "Prompt Cache job status transition: job_id=%s status=%s",
            job_id,
            status.value,
        )

        return history

    async def set_attempt_started(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
    ) -> int:
        """Increment the execution-attempt count and clear retry scheduling.

        Call this in the same transaction that appends RUNNING status.
        """

        stmt = (
            update(PromptCacheJob)
            .where(PromptCacheJob.id == job_id)
            .values(
                attempt_count=PromptCacheJob.attempt_count + 1,
                next_attempt_at=None,
                updated_at=func.clock_timestamp(),
            )
            .returning(PromptCacheJob.attempt_count)
        )

        result = await session.execute(stmt)
        attempt_count = result.scalar_one_or_none()

        if attempt_count is None:
            raise LookupError(
                f"Prompt Cache job not found: {job_id}"
            )

        logger.info(
            "Started Prompt Cache job attempt: job_id=%s attempt=%s",
            job_id,
            attempt_count,
        )

        return attempt_count

    async def set_retry_schedule(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        next_attempt_at: datetime,
        error: CacheJobError,
    ) -> None:
        """Persist the next execution time and structured failure details.

        The caller must append RETRYING status and insert the delayed
        outbox event in the same transaction.
        """

        stmt = (
            update(PromptCacheJob)
            .where(PromptCacheJob.id == job_id)
            .values(
                next_attempt_at=next_attempt_at,
                error=error.model_dump(mode="json"),
                updated_at=func.clock_timestamp(),
            )
        )

        result = await session.execute(stmt)

        if result.rowcount != 1:
            raise LookupError(
                f"Prompt Cache job not found: {job_id}"
            )

        logger.info(
            "Scheduled Prompt Cache job retry: "
            "job_id=%s next_attempt_at=%s error_code=%s",
            job_id,
            next_attempt_at,
            error.code.value,
        )

    async def mark_terminal(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        status: JobStatus,
        error: CacheJobError | None,
    ) -> None:
        """Mark a job COMPLETED or FAILED exactly once.

        The job row is locked before checking its latest status.
        The caller must append the corresponding terminal history record
        in this same transaction.
        """

        if status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise ValueError(
                "mark_terminal only accepts COMPLETED or FAILED."
            )

        job = await self.get_job(
            session,
            job_id,
            for_update=True,
        )

        if job is None:
            raise LookupError(
                f"Prompt Cache job not found: {job_id}"
            )

        current_status = await self.get_latest_status(
            session,
            job_id,
        )

        if current_status is None:
            logger.error(
                "Cannot mark job %s terminal: no status-history record exists.",
                job_id,
            )
            raise RuntimeError(
                f"Job {job_id} has no status-history record."
            )

        if current_status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        ):
            logger.warning(
                "Ignoring duplicate terminal transition: "
                "job_id=%s current_status=%s requested_status=%s",
                job_id,
                current_status.value,
                status.value,
            )
            return

        stmt = (
            update(PromptCacheJob)
            .where(PromptCacheJob.id == job_id)
            .values(
                error=error.model_dump(mode="json") if error else None,
                completed_at=func.clock_timestamp(),
                updated_at=func.clock_timestamp(),
            )
        )

        result = await session.execute(stmt)

        if result.rowcount != 1:
            raise LookupError(
                f"Prompt Cache job not found while marking terminal: {job_id}"
            )

    async def list_overdue_active_jobs(
        self,
        session: AsyncSession,
        *,
        limit: int,
    ) -> list[UUID]:
        """Find active jobs whose absolute deadline has elapsed."""

        self._validate_limit(limit)

        latest_status = self._latest_status_subquery()

        stmt = (
            select(PromptCacheJob.id)
            .join(
                latest_status,
                latest_status.c.job_id == PromptCacheJob.id,
            )
            .join(
                PromptCacheJobStatusHistory,
                PromptCacheJobStatusHistory.id
                == latest_status.c.latest_status_id,
            )
            .where(
                PromptCacheJobStatusHistory.status.in_(
                    [
                        JobStatus.PENDING.value,
                        JobStatus.RUNNING.value,
                        JobStatus.RETRYING.value,
                    ]
                ),
                PromptCacheJob.deadline_at <= func.clock_timestamp(),
            )
            .order_by(
                PromptCacheJob.deadline_at.asc(),
                PromptCacheJob.created_at.asc(),
            )
            .limit(limit)
        )

        result = await session.execute(stmt)

        return list(result.scalars().all())

    async def list_crashed_running_jobs(
        self,
        session: AsyncSession,
        *,
        limit: int,
    ) -> list[UUID]:
        """Find RUNNING jobs whose worker lease expired beyond the grace period.

        RETRYING jobs are deliberately excluded because they are waiting
        for their scheduled delayed outbox event.
        """

        self._validate_limit(limit)

        settings = get_prompt_cache_settings()
        grace_seconds = settings.WORKER_CRASH_GRACE_SECONDS

        latest_status = self._latest_status_subquery()

        stmt = (
            select(PromptCacheJob.id)
            .join(
                latest_status,
                latest_status.c.job_id == PromptCacheJob.id,
            )
            .join(
                PromptCacheJobStatusHistory,
                PromptCacheJobStatusHistory.id
                == latest_status.c.latest_status_id,
            )
            .join(
                PromptCache,
                PromptCache.id == PromptCacheJob.prompt_cache_id,
            )
            .join(
                PromptCacheLock,
                PromptCacheLock.signature == PromptCache.signature,
            )
            .where(
                PromptCacheJobStatusHistory.status == JobStatus.RUNNING.value,
                PromptCacheLock.lease_expires_at.is_not(None),
                (
                    PromptCacheLock.lease_expires_at
                    + func.make_interval(
                        0, 0, 0, 0, 0, 0,
                        settings.CACHE_JOB_TIMEOUT_SECONDS,
                    )
                )
                <= func.clock_timestamp(),
            )
            .order_by(
                PromptCacheLock.lease_expires_at.asc(),
                PromptCacheJob.created_at.asc(),
            )
            .limit(limit)
        )

        result = await session.execute(stmt)

        return list(result.scalars().all())
