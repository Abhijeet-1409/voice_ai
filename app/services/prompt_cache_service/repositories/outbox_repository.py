
"""Persistence operations for the transactional Prompt Cache outbox."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, literal, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infra.postgres import PromptCacheOutbox
from shared.logging_setup import get_logger


_LOGGER = "prompt_cache_service.repositories.outbox_repositories"
logger = get_logger(_LOGGER)


@dataclass(frozen=True)
class ClaimedOutboxEvent:
    """Represent an outbox event claimed by a publisher.

    Attributes:
        id: Unique outbox event identifier.
        job_id: Associated Prompt Cache job identifier.
        event_type: Type of event to publish.
        payload: JSON-compatible event payload.
        claim_token: Token proving ownership of this claim.
    """

    id: UUID
    job_id: UUID
    event_type: str
    payload: dict[str, Any]
    claim_token: UUID


class OutboxRepository:
    """Manage durable event delivery through PostgreSQL.

    Events are claimed inside short database transactions. The caller
    must commit the claim before making any Celery network calls.

    Transaction ownership belongs to the calling service.
    This repository never commits or rolls back a transaction.
    """

    @staticmethod
    def _seconds_interval(seconds: int):
        """Build a PostgreSQL interval expression from seconds."""

        return literal(seconds) * text("INTERVAL '1 second'")

    @staticmethod
    def _validate_positive_integer(
        value: int,
        field_name: str,
    ) -> None:
        """Validate a positive integer argument."""

        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{field_name} must be a positive integer."
            )

    @staticmethod
    def _sanitize_error_message(error_message: str) -> str:
        """Normalize and bound an error message before persistence."""

        return " ".join(error_message.split())[:1000]

    async def add_event(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        available_at: datetime | None = None,
    ) -> PromptCacheOutbox:
        """Insert an outbox event.

        When available_at is omitted, PostgreSQL assigns the current
        timestamp through the model's server default.

        This method only persists the event. It never publishes to Celery.
        """

        event = PromptCacheOutbox(
            job_id=job_id,
            event_type=event_type,
            payload=payload,
            **(
                {"available_at": available_at}
                if available_at is not None
                else {}
            ),
        )

        session.add(event)
        await session.flush()

        logger.info(
            "Created Prompt Cache outbox event: "
            "event_id=%s job_id=%s event_type=%s",
            event.id,
            job_id,
            event_type,
        )

        return event

    async def claim_batch(
        self,
        session: AsyncSession,
        *,
        batch_size: int,
        claim_lease_seconds: int,
    ) -> list[ClaimedOutboxEvent]:
        """Claim due, unpublished outbox events.

        The query uses FOR UPDATE SKIP LOCKED so multiple publisher
        processes can claim different events without waiting on one
        another.

        Each event receives its own claim token. The caller must commit
        this transaction before publishing any claimed event.

        Returns:
            A list of claimed events.
        """

        self._validate_positive_integer(batch_size, "batch_size")
        self._validate_positive_integer(
            claim_lease_seconds,
            "claim_lease_seconds",
        )

        now = func.clock_timestamp()

        stmt = (
            select(PromptCacheOutbox)
            .where(
                PromptCacheOutbox.published_at.is_(None),
                PromptCacheOutbox.available_at <= now,
                or_(
                    PromptCacheOutbox.claim_token.is_(None),
                    PromptCacheOutbox.claim_expires_at.is_(None),
                    PromptCacheOutbox.claim_expires_at <= now,
                ),
            )
            .order_by(
                PromptCacheOutbox.available_at.asc(),
                PromptCacheOutbox.created_at.asc(),
            )
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )

        result = await session.execute(stmt)
        events = result.scalars().all()

        claimed_events: list[ClaimedOutboxEvent] = []

        for event in events:
            claim_token = uuid4()

            claim_stmt = (
                update(PromptCacheOutbox)
                .where(
                    PromptCacheOutbox.id == event.id,
                    PromptCacheOutbox.published_at.is_(None),
                )
                .values(
                    claim_token=claim_token,
                    claim_expires_at=(
                        func.clock_timestamp()
                        + self._seconds_interval(claim_lease_seconds)
                    ),
                )
                .execution_options(synchronize_session=False)
            )

            await session.execute(claim_stmt)

            claimed_events.append(
                ClaimedOutboxEvent(
                    id=event.id,
                    job_id=event.job_id,
                    event_type=event.event_type,
                    payload=dict(event.payload),
                    claim_token=claim_token,
                )
            )

        await session.flush()

        if claimed_events:
            logger.info(
                "Claimed Prompt Cache outbox events: count=%s",
                len(claimed_events),
            )

        return claimed_events

    async def mark_published(
        self,
        session: AsyncSession,
        *,
        event_id: UUID,
        claim_token: UUID,
    ) -> bool:
        """Mark an event as successfully published.

        The update succeeds only when the event is still unpublished
        and the supplied claim token matches its current owner.

        Returns:
            True if the event was marked published, otherwise False.
        """

        stmt = (
            update(PromptCacheOutbox)
            .where(
                PromptCacheOutbox.id == event_id,
                PromptCacheOutbox.claim_token == claim_token,
                PromptCacheOutbox.published_at.is_(None),
            )
            .values(
                published_at=func.clock_timestamp(),
                claim_token=None,
                claim_expires_at=None,
            )
            .returning(PromptCacheOutbox.id)
        )

        result = await session.execute(stmt)
        published_event_id = result.scalar_one_or_none()

        if published_event_id is None:
            logger.warning(
                "Could not mark outbox event as published: "
                "event_id=%s claim ownership may have expired or changed.",
                event_id,
            )
            return False

        logger.info(
            "Published Prompt Cache outbox event: event_id=%s",
            event_id,
        )

        return True

    async def record_publish_failure(
        self,
        session: AsyncSession,
        *,
        event_id: UUID,
        claim_token: UUID,
        error_message: str,
        backoff_seconds: int,
    ) -> bool:
        """Record a failed publish and schedule another publishing attempt.

        The event remains unpublished. Its available_at timestamp is
        moved forward so the next publisher iteration cannot claim
        it before the configured backoff has elapsed.

        The update succeeds only when the supplied claim token still
        matches the event's current owner.

        Returns:
            True if the failure was recorded, otherwise False.
        """

        self._validate_positive_integer(
            backoff_seconds,
            "backoff_seconds",
        )

        sanitized_error = self._sanitize_error_message(error_message)

        stmt = (
            update(PromptCacheOutbox)
            .where(
                PromptCacheOutbox.id == event_id,
                PromptCacheOutbox.claim_token == claim_token,
                PromptCacheOutbox.published_at.is_(None),
            )
            .values(
                publish_attempt_count=(
                    PromptCacheOutbox.publish_attempt_count + 1
                ),
                last_error=sanitized_error,
                available_at=(
                    func.clock_timestamp()
                    + self._seconds_interval(backoff_seconds)
                ),
                claim_token=None,
                claim_expires_at=None,
            )
            .returning(
                PromptCacheOutbox.publish_attempt_count
            )
        )

        result = await session.execute(stmt)
        attempt_count = result.scalar_one_or_none()

        if attempt_count is None:
            logger.warning(
                "Could not record outbox publish failure: "
                "event_id=%s claim ownership may have expired or changed.",
                event_id,
            )
            return False

        logger.error(
            "Outbox publish failed: event_id=%s publish_attempt_count=%s "
            "backoff_seconds=%s error=%s",
            event_id,
            attempt_count,
            backoff_seconds,
            sanitized_error,
        )

        return True
