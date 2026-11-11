"""Deadline watchdog and crashed-worker recovery."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.logging_setup import get_logger

from ..config.settings import PromptCacheSettings
from ..repositories.cache_repository import CacheRepository
from ..repositories.job_repository import JobRepository
from ..repositories.lock_repository import LockRepository
from ..schemas.enums import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    CacheJobErrorCode,
    JobStatus,
)
from ..schemas.jobs import CacheJobError


_LOGGER = "prompt_cache_service.workers.recovery"
logger = get_logger(_LOGGER)


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Counts of jobs transitioned to FAILED during one recovery cycle."""

    deadline_failed: int
    crashed_worker_failed: int


class JobRecoveryService:
    """Finds and fails overdue jobs and jobs from crashed workers."""

    def __init__(
        self,
        *,
        job_repository: JobRepository,
        lock_repository: LockRepository,
        cache_repository: CacheRepository,
        settings: PromptCacheSettings,
    ) -> None:
        self.job_repository = job_repository
        self.lock_repository = lock_repository
        self.cache_repository = cache_repository
        self.settings = settings

    async def run_cycle(
        self,
        session: AsyncSession,
    ) -> RecoveryResult:
        """Run deadline recovery before crashed-worker recovery."""

        deadline_failed = await self.fail_overdue_jobs(session)

        crashed_worker_failed = (
            await self.fail_crashed_running_jobs(session)
        )

        return RecoveryResult(
            deadline_failed=deadline_failed,
            crashed_worker_failed=crashed_worker_failed,
        )

    async def fail_overdue_jobs(
        self,
        session: AsyncSession,
    ) -> int:
        """Fail active jobs whose absolute deadline has passed."""

        # Collect a bounded list of candidate job IDs.
        async with session.begin():
            candidate_ids = (
                await self.job_repository.list_overdue_active_jobs(
                    session,
                    limit=self.settings.RECOVERY_BATCH_SIZE,
                )
            )

        failed_count = 0

        for job_id in candidate_ids:
            async with session.begin():

                # Read without a row lock first to discover the
                # associated cache ID and signature.
                initial_job = await self.job_repository.get_job(
                    session,
                    job_id,
                )

                if initial_job is None:
                    continue

                initial_cache = await self.cache_repository.get_by_id(
                    session,
                    initial_job.prompt_cache_id,
                )

                if initial_cache is None:
                    logger.error(
                        "Cannot recover overdue job because its cache "
                        "row is missing: job_id=%s, cache_id=%s",
                        job_id,
                        initial_job.prompt_cache_id,
                    )
                    continue

                # Follow the global database lock order:
                # 1. prompt_caches
                # 2. prompt_cache_jobs
                # 3. prompt_cache_locks
                cache = await self.cache_repository.get_by_signature(
                    session,
                    initial_cache.signature,
                    for_update=True,
                )

                if cache is None:
                    continue

                job = await self.job_repository.get_job(
                    session,
                    job_id,
                    for_update=True,
                )

                if job is None:
                    continue

                status = await self.job_repository.get_latest_status(
                    session,
                    job_id,
                )

                # Another transaction may already have completed or
                # failed this job.
                if status not in ACTIVE_JOB_STATUSES:
                    continue

                db_now = await session.scalar(
                    select(func.clock_timestamp())
                )

                if db_now is None:
                    raise RuntimeError(
                        "Could not obtain PostgreSQL time during "
                        "deadline recovery."
                    )

                # Recheck the deadline after acquiring the job lock.
                if job.deadline_at > db_now:
                    continue

                error = self._deadline_error()

                # Invalidate ownership and increment the fencing token.
                # A late worker holding the old token cannot persist.
                await self.lock_repository.invalidate(
                    session,
                    signature=cache.signature,
                )

                await self._mark_failed(
                    session,
                    job_id=job_id,
                    error=error,
                )

                failed_count += 1

            # The transaction has committed.
            logger.warning(
                "Overdue cache job marked FAILED: job_id=%s",
                job_id,
            )

        return failed_count

    async def fail_crashed_running_jobs(
        self,
        session: AsyncSession,
    ) -> int:
        """Fail RUNNING jobs whose lease expired beyond crash grace."""

        async with session.begin():
            candidate_ids = (
                await self.job_repository.list_crashed_running_jobs(
                    session,
                    limit=self.settings.RECOVERY_BATCH_SIZE,
                )
            )

        failed_count = 0

        for job_id in candidate_ids:
            async with session.begin():

                # Read the job first to discover its cache signature.
                initial_job = await self.job_repository.get_job(
                    session,
                    job_id,
                )

                if initial_job is None:
                    continue

                initial_cache = await self.cache_repository.get_by_id(
                    session,
                    initial_job.prompt_cache_id,
                )

                if initial_cache is None:
                    logger.error(
                        "Cannot recover crashed job because its cache "
                        "row is missing: job_id=%s, cache_id=%s",
                        job_id,
                        initial_job.prompt_cache_id,
                    )
                    continue

                # Lock cache first, followed by job.
                cache = await self.cache_repository.get_by_signature(
                    session,
                    initial_cache.signature,
                    for_update=True,
                )

                if cache is None:
                    continue

                job = await self.job_repository.get_job(
                    session,
                    job_id,
                    for_update=True,
                )

                if job is None:
                    continue

                status = await self.job_repository.get_latest_status(
                    session,
                    job_id,
                )

                # Only a currently RUNNING job can be considered
                # a crashed worker.
                #
                # RETRYING is deliberately excluded because its
                # lease is normally released while waiting for retry.
                if status != JobStatus.RUNNING:
                    continue

                db_now = await session.scalar(
                    select(func.clock_timestamp())
                )

                if db_now is None:
                    raise RuntimeError(
                        "Could not obtain PostgreSQL time during "
                        "crashed-worker recovery."
                    )

                if job.deadline_at <= db_now:
                    # Deadline failure has priority over worker
                    # interruption.
                    error = self._deadline_error()

                else:
                    # Recheck lease expiration plus grace period.
                    # The candidate list may have become stale because
                    # a heartbeat renewed the lease in the meantime.
                    still_crashed = (
                        await self.job_repository.is_crashed_running_job(
                            session,
                            job_id=job_id,
                            crash_grace_seconds=(
                                self.settings.WORKER_CRASH_GRACE_SECONDS
                            ),
                        )
                    )

                    if not still_crashed:
                        continue

                    error = CacheJobError(
                        code=(
                            CacheJobErrorCode
                            .PROMPT_CACHE_WORKER_INTERRUPTED
                        ),
                        message=(
                            "The worker stopped unexpectedly while "
                            "executing the cache creation job."
                        ),
                        retryable=False,
                    )

                # Invalidate the old fencing ownership. This prevents
                # a worker with an old lease token from persisting.
                await self.lock_repository.invalidate(
                    session,
                    signature=cache.signature,
                )

                await self._mark_failed(
                    session,
                    job_id=job_id,
                    error=error,
                )

                failed_count += 1

            logger.warning(
                "Crashed cache job marked FAILED: "
                "job_id=%s, error_code=%s",
                job_id,
                error.code,
            )

        return failed_count

    async def _mark_failed(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        error: CacheJobError,
    ) -> None:
        """Append FAILED history and persist terminal job information."""

        await self.job_repository.append_status(
            session,
            job_id=job_id,
            status=JobStatus.FAILED,
        )

        await self.job_repository.mark_terminal(
            session,
            job_id=job_id,
            status=JobStatus.FAILED,
            error=error,
        )

    @staticmethod
    def _deadline_error() -> CacheJobError:
        return CacheJobError(
            code=CacheJobErrorCode.PROMPT_CACHE_JOB_DEADLINE_EXCEEDED,
            message=(
                "The cache creation job exceeded its "
                "overall execution deadline."
            ),
            retryable=False,
        )
