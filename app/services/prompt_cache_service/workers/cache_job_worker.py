"""Coordinates one Prompt Cache creation attempt."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from shared.infra.postgres import PromptCache, PromptCacheJob
from shared.logging_setup import get_logger

from ..config.settings import PromptCacheSettings
from ..providers.protocols import CacheProvider
from ..repositories.cache_repository import CacheRepository
from ..repositories.job_repository import JobRepository
from ..repositories.lock_repository import LeaseGrant, LockRepository
from ..repositories.outbox_repository import OutboxRepository
from ..schemas.cache import CachePayload
from ..schemas.enums import CacheJobErrorCode, JobStatus
from ..schemas.jobs import CacheCreationResult, CacheJobError
from ..services.error_classifier import ClassifiedError, ErrorClassifier
from ..services.job_deadline import JobDeadlineService
from ..services.retry_policy import RetryPolicy
from .lease_heartbeat import WorkerLeaseHeartbeat


_LOGGER = "prompt_cache_service.workers.cache_job_worker"
logger = get_logger(_LOGGER)


CACHE_CREATION_EVENT = "PROMPT_CACHE_CREATION_REQUESTED"

# Keep a small amount of time available for persistence and cleanup
# after the provider request has finished.
PERSISTENCE_SAFETY_MARGIN_SECONDS = 2.0


class CacheJobWorker:
    """Executes one cache-creation attempt without automatic takeover."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        cache_repository: CacheRepository,
        job_repository: JobRepository,
        lock_repository: LockRepository,
        outbox_repository: OutboxRepository,
        provider: CacheProvider,
        error_classifier: ErrorClassifier,
        retry_policy: RetryPolicy,
        deadline_service: JobDeadlineService,
        redis_client: Redis,
        settings: PromptCacheSettings,
    ) -> None:
        self.session_factory = session_factory
        self.cache_repository = cache_repository
        self.job_repository = job_repository
        self.lock_repository = lock_repository
        self.outbox_repository = outbox_repository
        self.provider = provider
        self.error_classifier = error_classifier
        self.retry_policy = retry_policy
        self.deadline_service = deadline_service
        self.redis_client = redis_client
        self.settings = settings

    async def execute(self, job_id: UUID) -> None:
        """Execute at most one attempt for the supplied job."""

        # ---------------------------------------------------------
        # 1. Load the job and check its current state.
        # ---------------------------------------------------------
        async with self.session_factory() as session:
            job_and_status = await self.job_repository.get_job_and_status(
                session,
                job_id,
            )

            if job_and_status is None:
                logger.warning(
                    "Cache job does not exist: job_id=%s",
                    job_id,
                )
                return

            job, status = job_and_status

            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                logger.info(
                    "Ignoring delivery for terminal cache job: "
                    "job_id=%s, status=%s",
                    job_id,
                    status,
                )
                return

            deadline = await self.deadline_service.get_remaining_budget(
                session,
                job_id=job_id,
            )

            if deadline.remaining_seconds <= 0:
                overdue = True
            else:
                overdue = False

        # Do not hold the read session while transitioning the job.
        if overdue:
            await self._fail_job(
                job_id=job_id,
                lease=None,
                error=self._deadline_error(),
            )
            return

        # ---------------------------------------------------------
        # 2. Atomically start the attempt and acquire its lease.
        # ---------------------------------------------------------
        attempt = await self._start_attempt(job_id)

        if attempt is None:
            # Examples:
            # - duplicate Celery delivery
            # - job is already RUNNING
            # - job is terminal
            # - retry is not due yet
            # - another worker owns the signature lease
            logger.info(
                "Cache job was not eligible to start: job_id=%s",
                job_id,
            )
            return

        job, status, lease = attempt

        logger.info(
            "Cache job attempt started: job_id=%s, "
            "attempt=%s, fencing_token=%s",
            job_id,
            job.attempt_count,
            lease.fencing_token,
        )

        heartbeat = WorkerLeaseHeartbeat(
            session_factory=self.session_factory,
            lock_repository=self.lock_repository,
            lease=lease,
            lease_seconds=self.settings.WORKER_LEASE_SECONDS,
            heartbeat_seconds=self.settings.WORKER_HEARTBEAT_SECONDS,
        )

        try:
            await heartbeat.start()

            # -----------------------------------------------------
            # 3. Load the cache payload and calculate request budget.
            # -----------------------------------------------------
            async with self.session_factory() as session:
                cache = await self.cache_repository.get_by_id(
                    session,
                    job.prompt_cache_id,
                )

                if cache is None:
                    raise RuntimeError(
                        "Prompt cache row is missing for the job."
                    )

                payload = CachePayload.model_validate(cache.payload)

                budget = await self.deadline_service.get_remaining_budget(
                    session,
                    job_id=job_id,
                )

            request_budget = (
                budget.remaining_seconds
                - PERSISTENCE_SAFETY_MARGIN_SECONDS
            )

            if request_budget <= 0:
                await self._fail_job(
                    job_id=job_id,
                    lease=lease,
                    error=self._deadline_error(),
                )
                return

            request_timeout = min(
                self.settings.GEMINI_REQUEST_TIMEOUT_SECONDS,
                request_budget,
            )

            # -----------------------------------------------------
            # 4. Call Gemini outside a database transaction.
            # -----------------------------------------------------
            result = await self.provider.create_cached_content(
                payload=payload,
                timeout_seconds=request_timeout,
            )

            # A provider response is not enough to prove that this
            # worker still owns the job.
            if heartbeat.lease_lost:
                logger.warning(
                    "Discarding Gemini result because the lease was lost: "
                    "job_id=%s, fencing_token=%s",
                    job_id,
                    lease.fencing_token,
                )
                return

            # -----------------------------------------------------
            # 5. Persist successful result atomically.
            # -----------------------------------------------------
            completed = await self._complete_success(
                job_id=job_id,
                lease=lease,
                result=result,
            )

            if not completed:
                logger.warning(
                    "Gemini result was not persisted because the "
                    "worker no longer owns the active job: job_id=%s",
                    job_id,
                )
                return

            logger.info(
                "Cache job completed successfully: job_id=%s",
                job_id,
            )

            # Redis is only an optimization. PostgreSQL has already
            # committed, so Redis failure must not undo completion.
            await self._update_redis_after_success(
                signature=cache.signature,
                cache_id=cache.id,
                result=result,
            )

        except asyncio.CancelledError:
            # Do not convert task cancellation into a provider error.
            # The database state is handled by the recovery policy.
            raise

        except Exception as exc:
            if heartbeat.lease_lost:
                logger.warning(
                    "Not scheduling another attempt because lease "
                    "ownership was lost: job_id=%s",
                    job_id,
                )
                return

            classified_error = self.error_classifier.classify(exc)

            logger.exception(
                "Cache creation attempt failed: job_id=%s, "
                "retryable=%s, code=%s",
                job_id,
                classified_error.retryable,
                classified_error.code,
            )

            await self._schedule_retry_or_fail(
                job_id=job_id,
                lease=lease,
                classified_error=classified_error,
            )

        finally:
            await heartbeat.stop()

    async def _start_attempt(
        self,
        job_id: UUID,
    ) -> tuple[PromptCacheJob, JobStatus, LeaseGrant] | None:
        """Atomically acquire a lease and mark one attempt RUNNING."""

        async with self.session_factory() as session:
            async with session.begin():
                job = await self.job_repository.get_job(
                    session,
                    job_id,
                    for_update=True,
                )

                if job is None:
                    return None

                status = await self.job_repository.get_latest_status(
                    session,
                    job_id,
                )

                if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    return None

                # A second delivery must not start another attempt
                # while the first execution is already RUNNING.
                if status == JobStatus.RUNNING:
                    return None

                if status not in (JobStatus.PENDING, JobStatus.RETRYING):
                    return None

                db_now = await session.scalar(
                    select(func.clock_timestamp())
                )

                if db_now is None or job.deadline_at <= db_now:
                    return None

                if status == JobStatus.RETRYING:
                    if (
                        job.next_attempt_at is None
                        or job.next_attempt_at > db_now
                    ):
                        return None

                cache = await self.cache_repository.get_by_id(
                    session,
                    job.prompt_cache_id,
                )

                if cache is None:
                    return None

                owner_token = uuid4()

                lease = await self.lock_repository.try_acquire(
                    session,
                    signature=cache.signature,
                    owner_token=owner_token,
                    lease_seconds=self.settings.WORKER_LEASE_SECONDS,
                )

                if lease is None:
                    return None

                # These changes belong to the same transaction as
                # successful lease acquisition.
                await self.job_repository.set_attempt_started(
                    session,
                    job_id=job_id,
                )

                await self.job_repository.append_status(
                    session,
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                )

                # Refresh the in-memory count for logging/consumers.
                await session.refresh(job)

                return job, JobStatus.RUNNING, lease

    async def _complete_success(
        self,
        *,
        job_id: UUID,
        lease: LeaseGrant,
        result: CacheCreationResult,
    ) -> bool:
        """Persist provider success only if this worker still owns the job."""

        async with self.session_factory() as session:
            async with session.begin():
                # First read without a row lock to discover the cache ID.
                initial_job = await self.job_repository.get_job(
                    session,
                    job_id,
                )

                if initial_job is None:
                    return False

                initial_cache = await self.cache_repository.get_by_id(
                    session,
                    initial_job.prompt_cache_id,
                )

                if initial_cache is None:
                    return False

                # Global lock order:
                # 1. prompt_caches
                # 2. prompt_cache_jobs
                # 3. prompt_cache_locks
                cache = await self.cache_repository.get_by_signature(
                    session,
                    initial_cache.signature,
                    for_update=True,
                )

                if cache is None:
                    return False

                job = await self.job_repository.get_job(
                    session,
                    job_id,
                    for_update=True,
                )

                if job is None:
                    return False

                status = await self.job_repository.get_latest_status(
                    session,
                    job_id,
                )

                if status != JobStatus.RUNNING:
                    return False

                db_now = await session.scalar(
                    select(func.clock_timestamp())
                )

                if db_now is None or job.deadline_at <= db_now:
                    return False

                # Final ownership check occurs inside the same
                # transaction as the result persistence.
                owns_lease = await self.lock_repository.is_owner(
                    session,
                    signature=lease.signature,
                    owner_token=lease.owner_token,
                    fencing_token=lease.fencing_token,
                )

                if not owns_lease:
                    return False

                await self.cache_repository.save_successful_result(
                    session,
                    cache_id=cache.id,
                    result=result,
                )

                await self.job_repository.append_status(
                    session,
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                )

                await self.job_repository.mark_terminal(
                    session,
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    error=None,
                )

                released = await self.lock_repository.release(
                    session,
                    signature=lease.signature,
                    owner_token=lease.owner_token,
                    fencing_token=lease.fencing_token,
                )

                if not released:
                    # Raising rolls back the cache result and COMPLETED
                    # transition. We must not claim success without
                    # being able to release the exact lease.
                    raise RuntimeError(
                        "Could not release the worker's lease "
                        "during successful completion."
                    )

        return True

    async def _schedule_retry_or_fail(
        self,
        *,
        job_id: UUID,
        lease: LeaseGrant,
        classified_error: ClassifiedError,
    ) -> None:
        """Atomically schedule a retry or persist terminal failure."""

        async with self.session_factory() as session:
            async with session.begin():
                job = await self.job_repository.get_job(
                    session,
                    job_id,
                    for_update=True,
                )

                if job is None:
                    return

                status = await self.job_repository.get_latest_status(
                    session,
                    job_id,
                )

                if status != JobStatus.RUNNING:
                    return

                budget = await self.deadline_service.get_remaining_budget(
                    session,
                    job_id=job_id,
                )

                decision = self.retry_policy.decide(
                    error=classified_error,
                    attempt_count=job.attempt_count,
                    remaining_seconds=budget.remaining_seconds,
                )

                if decision.should_retry:
                    if decision.delay_seconds is None:
                        raise RuntimeError(
                            "Retry decision has no delay."
                        )

                    db_now = await session.scalar(
                        select(func.clock_timestamp())
                    )

                    if db_now is None:
                        raise RuntimeError(
                            "Could not obtain PostgreSQL time."
                        )

                    next_attempt_at = db_now + timedelta(
                        seconds=decision.delay_seconds
                    )

                    retry_error = self._to_job_error(classified_error)

                    await self.job_repository.set_retry_schedule(
                        session,
                        job_id=job_id,
                        next_attempt_at=next_attempt_at,
                        error=retry_error,
                    )

                    await self.job_repository.append_status(
                        session,
                        job_id=job_id,
                        status=JobStatus.RETRYING,
                    )

                    await self.outbox_repository.add_event(
                        session,
                        job_id=job_id,
                        event_type=CACHE_CREATION_EVENT,
                        payload={"job_id": str(job_id)},
                        available_at=next_attempt_at,
                    )

                    released = await self.lock_repository.release(
                        session,
                        signature=lease.signature,
                        owner_token=lease.owner_token,
                        fencing_token=lease.fencing_token,
                    )

                    if not released:
                        raise RuntimeError(
                            "Could not release the worker lease "
                            "while scheduling a retry."
                        )

                    logger.info(
                        "Cache job retry scheduled: job_id=%s, "
                        "delay_seconds=%.2f",
                        job_id,
                        decision.delay_seconds,
                    )
                    return

                terminal_error = (
                    decision.terminal_error
                    or self._to_job_error(classified_error)
                )

                await self.job_repository.append_status(
                    session,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                )

                await self.job_repository.mark_terminal(
                    session,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=terminal_error,
                )

                released = await self.lock_repository.release(
                    session,
                    signature=lease.signature,
                    owner_token=lease.owner_token,
                    fencing_token=lease.fencing_token,
                )

                if not released:
                    raise RuntimeError(
                        "Could not release the worker lease "
                        "while marking the job failed."
                    )

                logger.warning(
                    "Cache job marked FAILED: job_id=%s, code=%s",
                    job_id,
                    terminal_error.code,
                )

    async def _fail_job(
        self,
        *,
        job_id: UUID,
        lease: LeaseGrant | None,
        error: CacheJobError,
    ) -> bool:
        """Persist an authoritative failure transition."""

        async with self.session_factory() as session:
            async with session.begin():
                job = await self.job_repository.get_job(
                    session,
                    job_id,
                    for_update=True,
                )

                if job is None:
                    return False

                status = await self.job_repository.get_latest_status(
                    session,
                    job_id,
                )

                if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    return False

                if lease is not None:
                    await self.lock_repository.release(
                        session,
                        signature=lease.signature,
                        owner_token=lease.owner_token,
                        fencing_token=lease.fencing_token,
                    )
                else:
                    # No lease was supplied: this is an authoritative
                    # failure such as a deadline expiry before attempt.
                    cache = await self.cache_repository.get_by_id(
                        session,
                        job.prompt_cache_id,
                    )

                    if cache is not None:
                        await self.lock_repository.invalidate(
                            session,
                            signature=cache.signature,
                        )

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

        logger.warning(
            "Cache job marked FAILED: job_id=%s, code=%s",
            job_id,
            error.code,
        )
        return True

    async def _update_redis_after_success(
        self,
        *,
        signature: str,
        cache_id: UUID,
        result: CacheCreationResult,
    ) -> None:
        """Best-effort update of Redis lookup metadata after DB commit."""

        now = datetime.now(timezone.utc)
        remaining_seconds = (
            result.expires_at - now
        ).total_seconds()

        ttl = int(
            remaining_seconds
            - self.settings.CACHE_REDIS_TTL_SAFETY_SECONDS
        )

        if ttl <= 0:
            return

        key = f"prompt_cache:signature:{signature}"

        metadata = {
            "prompt_cache_id": str(cache_id),
            "cache_name": result.cache_name,
            "expires_at": result.expires_at.isoformat(),
        }

        try:
            await self.redis_client.set(
                key,
                json.dumps(metadata),
                ex=ttl,
            )

        except Exception:
            # PostgreSQL already contains the committed result.
            # Redis failure must not reverse that transaction.
            logger.exception(
                "Could not update Redis after cache creation: "
                "cache_id=%s",
                cache_id,
            )

    @staticmethod
    def _deadline_error() -> CacheJobError:
        return CacheJobError(
            code=CacheJobErrorCode.PROMPT_CACHE_JOB_DEADLINE_EXCEEDED,
            message="The cache creation job exceeded its deadline.",
            retryable=False,
        )

    @staticmethod
    def _to_job_error(
        classified_error: ClassifiedError,
    ) -> CacheJobError:
        """Convert internal classification into the public error schema."""

        return CacheJobError(
            code=classified_error.code,
            message=classified_error.message,
            retryable=classified_error.retryable,
        )
