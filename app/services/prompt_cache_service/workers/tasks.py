"""Synchronous Celery entry points for async Prompt Cache workflows."""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from celery.signals import worker_process_shutdown
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from shared.infra.postgres.database import (
    close_db_async,
    get_async_sessionmaker,
    init_db_async,
)
from shared.infra.redis.client import (
    close_redis_async,
    get_async_redis_client,
)
from shared.logging_setup import get_logger

from ..config.settings import (
    PromptCacheSettings,
    get_prompt_cache_settings,
)
from ..providers.gemini_cache_provider import GeminiCacheProvider
from ..repositories.cache_repository import CacheRepository
from ..repositories.job_repository import JobRepository
from ..repositories.lock_repository import LockRepository
from ..repositories.outbox_repository import OutboxRepository
from ..services.error_classifier import ErrorClassifier
from ..services.job_deadline import JobDeadlineService
from ..services.retry_policy import RetryPolicy
from .cache_job_worker import CacheJobWorker
from .celery_app import celery_app


_LOGGER = "prompt_cache_service.workers.tasks"
logger = get_logger(_LOGGER)


@dataclass(slots=True)
class WorkerRuntime:
    """Resources owned by one Celery prefork child process."""

    pid: int
    loop: asyncio.AbstractEventLoop
    settings: PromptCacheSettings
    session_factory: async_sessionmaker[AsyncSession]
    redis_client: Redis
    provider: GeminiCacheProvider


# This global is intentionally process-local.
#
# Under Celery's prefork pool, each child has its own copy of this
# module and initializes its own runtime after the fork.
_runtime: WorkerRuntime | None = None


async def _initialize_runtime(
    *,
    pid: int,
    loop: asyncio.AbstractEventLoop,
) -> WorkerRuntime:
    """Initialize shared async resources inside the worker child."""

    settings = get_prompt_cache_settings()

    db_initialization_attempted = False
    redis_initialized = False
    provider: GeminiCacheProvider | None = None

    try:
        # Initialize the shared async SQLAlchemy engine on this loop.
        db_initialization_attempted = True
        await init_db_async()

        session_factory = get_async_sessionmaker()

        # Reuse the shared async Redis singleton. The getter creates
        # the client; network operations remain asynchronous.
        redis_client = get_async_redis_client()
        redis_initialized = True

        # One provider instance per worker child, not per job.
        provider = GeminiCacheProvider(settings=settings)

        logger.info(
            "Prompt Cache worker runtime initialized: pid=%s",
            pid,
        )

        return WorkerRuntime(
            pid=pid,
            loop=loop,
            settings=settings,
            session_factory=session_factory,
            redis_client=redis_client,
            provider=provider,
        )

    except BaseException:
        # Clean up any resources that were partially initialized.
        if provider is not None:
            try:
                await provider.aclose()
            except Exception:
                logger.exception(
                    "Failed to close Gemini provider after startup failure."
                )

        if redis_initialized:
            try:
                await close_redis_async()
            except Exception:
                logger.exception(
                    "Failed to close Redis after startup failure."
                )

        if db_initialization_attempted:
            try:
                await close_db_async()
            except Exception:
                logger.exception(
                    "Failed to close database after startup failure."
                )

        raise


def _get_worker_runtime() -> WorkerRuntime:
    """Return the current process runtime, creating it when necessary."""

    global _runtime

    current_pid = os.getpid()

    if (
        _runtime is not None
        and _runtime.pid == current_pid
        and not _runtime.loop.is_closed()
    ):
        return _runtime

    # If this module was imported before fork, the parent process's
    # runtime must not be reused inside the child. In particular, do
    # not use the parent's async engine or pooled connections.
    _runtime = None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        runtime = loop.run_until_complete(
            _initialize_runtime(
                pid=current_pid,
                loop=loop,
            )
        )

    except BaseException:
        loop.close()
        asyncio.set_event_loop(None)
        raise

    _runtime = runtime
    return runtime


def _build_cache_job_worker(
    runtime: WorkerRuntime,
) -> CacheJobWorker:
    """Construct the worker's stateless dependencies."""

    cache_repository = CacheRepository()
    job_repository = JobRepository()
    lock_repository = LockRepository()
    outbox_repository = OutboxRepository()

    return CacheJobWorker(
        session_factory=runtime.session_factory,
        cache_repository=cache_repository,
        job_repository=job_repository,
        lock_repository=lock_repository,
        outbox_repository=outbox_repository,
        provider=runtime.provider,
        error_classifier=ErrorClassifier(),
        retry_policy=RetryPolicy(
            settings=runtime.settings,
            random_source=random.SystemRandom(),
        ),
        deadline_service=JobDeadlineService(),
        redis_client=runtime.redis_client,
        settings=runtime.settings,
    )


async def create_cache_async(job_id: UUID) -> None:
    """Run one cache-creation workflow on the worker's event loop."""

    runtime = _runtime

    if runtime is None or runtime.pid != os.getpid():
        raise RuntimeError(
            "Worker runtime has not been initialized in this process."
        )

    worker = _build_cache_job_worker(runtime)

    await worker.execute(job_id)


async def recover_jobs_async() -> dict[str, int]:
    """Run one bounded recovery/watchdog cycle."""

    runtime = _runtime

    if runtime is None or runtime.pid != os.getpid():
        raise RuntimeError(
            "Worker runtime has not been initialized in this process."
        )

    # Imported here because recovery.py is a separate worker component.
    from .recovery import JobRecoveryService

    job_repository = JobRepository()
    lock_repository = LockRepository()

    recovery_service = JobRecoveryService(
        job_repository=job_repository,
        lock_repository=lock_repository,
        settings=runtime.settings,
    )

    # Recovery uses one fresh session for this bounded cycle.
    # The recovery service owns its individual transaction boundaries.
    async with runtime.session_factory() as session:
        result = await recovery_service.run_cycle(session)

    counts = {
        "deadline_failed": result.deadline_failed,
        "crashed_worker_failed": result.crashed_worker_failed,
    }

    logger.info(
        "Prompt Cache recovery cycle completed: "
        "deadline_failed=%s, crashed_worker_failed=%s",
        counts["deadline_failed"],
        counts["crashed_worker_failed"],
    )

    return counts


@celery_app.task(name="prompt_cache.create_cache")
def create_cache_task(job_id: str) -> None:
    """Synchronous Celery entry point for cache creation."""

    try:
        parsed_job_id = UUID(job_id)
    except (ValueError, TypeError, AttributeError):
        logger.warning(
            "Discarding Celery message with invalid job ID: %r",
            job_id,
        )
        return

    runtime = _get_worker_runtime()

    # Reuse the same event loop for every task handled by this child.
    # Do not replace this with asyncio.run().
    runtime.loop.run_until_complete(
        create_cache_async(parsed_job_id)
    )


@celery_app.task(name="prompt_cache.recover_jobs")
def recover_jobs_task() -> dict[str, int]:
    """Synchronous Celery entry point for one recovery cycle."""

    runtime = _get_worker_runtime()

    return runtime.loop.run_until_complete(
        recover_jobs_async()
    )


async def _shutdown_runtime(runtime: WorkerRuntime) -> None:
    """Close resources owned by this worker child."""

    # Close the provider first because it may own SDK HTTP transports.
    try:
        await runtime.provider.aclose()
    except Exception:
        logger.exception(
            "Failed to close Gemini provider during worker shutdown."
        )

    try:
        await close_redis_async()
    except Exception:
        logger.exception(
            "Failed to close shared Redis client during worker shutdown."
        )

    try:
        await close_db_async()
    except Exception:
        logger.exception(
            "Failed to close shared database engine during worker shutdown."
        )


@worker_process_shutdown.connect(weak=False)
def _on_worker_process_shutdown(
    pid: int | None = None,
    **kwargs: Any,
) -> None:
    """Dispose async resources when a Celery prefork child exits."""

    global _runtime

    runtime = _runtime

    if runtime is None:
        return

    current_pid = os.getpid()

    # Never close resources that belong to another process.
    if runtime.pid != current_pid:
        return

    if runtime.loop.is_closed():
        _runtime = None
        return

    logger.info(
        "Shutting down Prompt Cache worker runtime: pid=%s",
        current_pid,
    )

    try:
        runtime.loop.run_until_complete(
            _shutdown_runtime(runtime)
        )

    except Exception:
        logger.exception(
            "Unexpected error during Prompt Cache worker shutdown."
        )

    finally:
        runtime.loop.close()
        asyncio.set_event_loop(None)
        _runtime = None

        logger.info(
            "Prompt Cache worker runtime shut down: pid=%s",
            current_pid,
        )
