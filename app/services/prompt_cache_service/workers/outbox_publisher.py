"""Publishes transactional outbox events to Celery."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from typing import Any

from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from shared.infra.postgres.database import (
    close_db_async,
    get_async_sessionmaker,
    init_db_async,
)
from shared.logging_setup import get_logger

from ..config.settings import (
    PromptCacheSettings,
    get_prompt_cache_settings,
)
from ..repositories.outbox_repository import (
    ClaimedOutboxEvent,
    OutboxRepository,
)
from .celery_app import celery_app


logger = get_logger(
    "prompt_cache_service.workers.outbox_publisher"
)

# This must match the event type inserted when a cache job is created
# and when a retry is scheduled.
CACHE_CREATION_EVENT = "PROMPT_CACHE_CREATION_REQUESTED"

# This must match the registered Celery task name in tasks.py.
CACHE_CREATION_TASK = "prompt_cache.create_cache"


class OutboxPublisher:
    """Claims due outbox events and publishes their job IDs to Celery."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        outbox_repository: OutboxRepository,
        celery_app: Celery,
        settings: PromptCacheSettings,
    ) -> None:
        self.session_factory = session_factory
        self.outbox_repository = outbox_repository
        self.celery_app = celery_app
        self.settings = settings

        # Used for graceful termination of the standalone publisher.
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        """Request graceful shutdown of the publisher loop."""

        self._stop_event.set()

    async def publish_once(self) -> int:
        """Claim and process one bounded batch of outbox events.

        Returns:
            Number of events successfully published and marked published.
        """

        # ---------------------------------------------------------
        # 1. Claim events in a short database transaction.
        # ---------------------------------------------------------
        async with self.session_factory() as session:
            async with session.begin():
                events = await self.outbox_repository.claim_batch(
                    session,
                    batch_size=self.settings.OUTBOX_BATCH_SIZE,
                    claim_lease_seconds=(
                        self.settings.OUTBOX_CLAIM_LEASE_SECONDS
                    ),
                )

        if not events:
            return 0

        logger.debug(
            "Claimed %s outbox events for publishing.",
            len(events),
        )

        # ---------------------------------------------------------
        # 2. Publish outside the claim transaction.
        # ---------------------------------------------------------
        published_count = 0

        for event in events:
            try:
                published = await self._publish_event(event)

                if published:
                    published_count += 1

            except asyncio.CancelledError:
                raise

            except Exception:
                # A failure here may occur after Celery accepted the
                # message but before PostgreSQL marked it published.
                # The claim will eventually expire and allow another
                # publisher iteration to process the event.
                logger.exception(
                    "Unexpected error processing outbox event: "
                    "event_id=%s, job_id=%s",
                    event.id,
                    event.job_id,
                )

        return published_count

    async def _publish_event(
        self,
        event: ClaimedOutboxEvent,
    ) -> bool:
        """Validate, publish, and mark one claimed event."""

        # ---------------------------------------------------------
        # 1. Validate the event before sending anything to Celery.
        # ---------------------------------------------------------
        validation_error = self._validate_event(event)

        if validation_error is not None:
            logger.error(
                "Invalid outbox event: event_id=%s, job_id=%s, reason=%s",
                event.id,
                event.job_id,
                validation_error,
            )

            await self._record_publish_failure(
                event=event,
                error_message=validation_error,
            )
            return False

        # ---------------------------------------------------------
        # 2. Publish only the job ID to Celery.
        #
        # Celery's send_task() is synchronous. Run it in a worker
        # thread so the async publisher event loop is not blocked.
        # ---------------------------------------------------------
        try:
            await asyncio.to_thread(
                self.celery_app.send_task,
                CACHE_CREATION_TASK,
                args=[str(event.job_id)],
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            # Avoid storing raw exception text because it may contain
            # broker details or other operational information.
            safe_error = (
                f"Celery publish failed ({type(exc).__name__})."
            )

            logger.exception(
                "Could not publish outbox event: "
                "event_id=%s, job_id=%s",
                event.id,
                event.job_id,
            )

            await self._record_publish_failure(
                event=event,
                error_message=safe_error,
            )
            return False

        # ---------------------------------------------------------
        # 3. Broker accepted the message.
        #
        # Mark it published in a separate transaction, and only
        # when this publisher still owns the exact claim token.
        # ---------------------------------------------------------
        async with self.session_factory() as session:
            async with session.begin():
                marked = await self.outbox_repository.mark_published(
                    session,
                    event_id=event.id,
                    claim_token=event.claim_token,
                )

        if not marked:
            # The claim may have expired or been replaced while the
            # broker call was running. Do not update another
            # publisher's claim.
            #
            # The event may be published again after its claim expires.
            # This is expected at-least-once delivery behavior.
            logger.warning(
                "Celery accepted the event, but its outbox claim "
                "could not be marked published: event_id=%s, job_id=%s",
                event.id,
                event.job_id,
            )
            return False

        logger.info(
            "Outbox event published successfully: "
            "event_id=%s, job_id=%s",
            event.id,
            event.job_id,
        )

        return True

    async def _record_publish_failure(
        self,
        *,
        event: ClaimedOutboxEvent,
        error_message: str,
    ) -> None:
        """Record a publish failure and arrange a later claim."""

        try:
            async with self.session_factory() as session:
                async with session.begin():
                    recorded = (
                        await self.outbox_repository.record_publish_failure(
                            session,
                            event_id=event.id,
                            claim_token=event.claim_token,
                            error_message=error_message,
                            backoff_seconds=(
                                self.settings
                                .OUTBOX_PUBLISH_FAILURE_BACKOFF_SECONDS
                            ),
                        )
                    )

            if not recorded:
                logger.warning(
                    "Could not record outbox failure because the "
                    "claim is no longer owned: event_id=%s",
                    event.id,
                )
                return

            logger.warning(
                "Outbox publish failure recorded; event will become "
                "eligible again after backoff: event_id=%s",
                event.id,
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "Failed to persist outbox publish failure: event_id=%s",
                event.id,
            )

    @staticmethod
    def _validate_event(
        event: ClaimedOutboxEvent,
    ) -> str | None:
        """Validate the event type and minimal payload contract."""

        if event.event_type != CACHE_CREATION_EVENT:
            return "Unsupported outbox event type."

        if not isinstance(event.payload, dict):
            return "Outbox event payload must be a JSON object."

        payload_job_id = event.payload.get("job_id")

        if not isinstance(payload_job_id, str):
            return "Outbox event payload must contain a string job_id."

        if payload_job_id != str(event.job_id):
            return "Outbox payload job_id does not match the event job_id."

        return None

    async def _wait_or_stop(self) -> None:
        """Sleep for the configured poll interval or stop immediately."""

        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=self.settings.OUTBOX_POLL_INTERVAL_SECONDS,
            )

        except asyncio.TimeoutError:
            pass

    async def run_forever(self) -> None:
        """Continuously publish due events until shutdown is requested."""

        logger.info(
            "Outbox publisher started: poll_interval=%s seconds, "
            "batch_size=%s",
            self.settings.OUTBOX_POLL_INTERVAL_SECONDS,
            self.settings.OUTBOX_BATCH_SIZE,
        )

        while not self._stop_event.is_set():
            try:
                published_count = await self.publish_once()

            except asyncio.CancelledError:
                raise

            except Exception:
                # Database failures and other unexpected iteration
                # failures should not permanently stop the publisher.
                logger.exception(
                    "Outbox publisher iteration failed."
                )

                await self._wait_or_stop()
                continue

            if published_count == 0:
                # No successfully published event means there is no
                # immediate successful backlog to drain. Any failed
                # events have been moved behind their retry backoff.
                await self._wait_or_stop()

            # If at least one event was published, immediately check
            # for another batch instead of sleeping unnecessarily.

        logger.info("Outbox publisher stopped.")


async def _main() -> None:
    """Initialize resources and run the standalone publisher process."""

    settings = get_prompt_cache_settings()

    # This process owns its shared async database lifecycle.
    # Do not use this entry point inside a process whose DB singleton
    # is owned by another application component.
    db_initialization_attempted = False

    try:
        db_initialization_attempted = True
        await init_db_async()

        session_factory = get_async_sessionmaker()

        publisher = OutboxPublisher(
            session_factory=session_factory,
            outbox_repository=OutboxRepository(),
            celery_app=celery_app,
            settings=settings,
        )

        # Register OS termination signals on the process event loop.
        loop = asyncio.get_running_loop()
        registered_signals: list[signal.Signals] = []

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    publisher.request_stop,
                )
                registered_signals.append(sig)

            except (NotImplementedError, RuntimeError):
                # Some environments do not support asyncio signal
                # handlers. Normal asyncio cancellation still works.
                logger.warning(
                    "Could not register async signal handler: signal=%s",
                    sig.name,
                )

        logger.info("Starting standalone outbox publisher.")

        try:
            await publisher.run_forever()

        finally:
            for sig in registered_signals:
                loop.remove_signal_handler(sig)

    finally:
        if db_initialization_attempted:
            try:
                await close_db_async()
            except Exception:
                logger.exception(
                    "Failed to close database during publisher shutdown."
                )


def main() -> None:
    """CLI entry point; one event loop for the process lifetime."""

    asyncio.run(_main())


if __name__ == "__main__":
    main()