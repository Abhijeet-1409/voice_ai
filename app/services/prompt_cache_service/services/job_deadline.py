
"""Calculate and enforce cache-generation job deadlines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infra.postgres import PromptCacheJob

from ..core.exceptions import ServiceError
from ..schemas.enums import CacheJobErrorCode


@dataclass(frozen=True)
class DeadlineBudget:
    """Remaining execution budget for a cache-generation job."""

    deadline_at: datetime
    remaining_seconds: float


class JobDeadlineService:
    """Calculate remaining job time using PostgreSQL's clock."""

    async def get_remaining_budget(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
    ) -> DeadlineBudget:
        """Fetch the job deadline and database time in one query."""

        statement = select(
            PromptCacheJob.deadline_at,
            func.clock_timestamp().label("database_now"),
        ).where(
            PromptCacheJob.id == job_id,
        )

        result = await session.execute(statement)
        row = result.one_or_none()

        if row is None:
            raise ServiceError(
                code=CacheJobErrorCode.PROMPT_CACHE_JOB_NOT_FOUND.value,
                message="Cache-generation job was not found.",
                http_status=404,
                retryable=False,
            )

        deadline_at, database_now = row

        # Both timestamps should be timezone-aware because the database
        # columns use timezone-aware PostgreSQL timestamps.
        if (
            deadline_at.tzinfo is None
            or deadline_at.utcoffset() is None
            or database_now.tzinfo is None
            or database_now.utcoffset() is None
        ):
            raise ValueError(
                "Job deadline and database time must be timezone-aware."
            )

        remaining_seconds = (
            deadline_at - database_now
        ).total_seconds()

        return DeadlineBudget(
            deadline_at=deadline_at,
            remaining_seconds=remaining_seconds,
        )

    def calculate_request_timeout(
        self,
        *,
        remaining_seconds: float,
        configured_request_timeout_seconds: int,
        safety_margin_seconds: float = 1.0,
    ) -> float:
        """Calculate the maximum safe timeout for one provider request.

        Preserve some time after the provider request for result
        validation, persistence, and transaction completion.
        """

        if not math.isfinite(remaining_seconds):
            raise ValueError(
                "remaining_seconds must be finite."
            )

        if (
            isinstance(configured_request_timeout_seconds, bool)
            or configured_request_timeout_seconds <= 0
        ):
            raise ValueError(
                "configured_request_timeout_seconds must be positive."
            )

        if (
            not math.isfinite(safety_margin_seconds)
            or safety_margin_seconds < 0
        ):
            raise ValueError(
                "safety_margin_seconds must be finite and nonnegative."
            )

        usable_seconds = (
            remaining_seconds - safety_margin_seconds
        )

        # There is no useful time left for a Gemini request.
        if usable_seconds <= 0:
            raise self._deadline_exceeded()

        return min(
            float(configured_request_timeout_seconds),
            usable_seconds,
        )

    async def is_deadline_expired(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
    ) -> bool:
        """Determine whether the job deadline has passed."""

        budget = await self.get_remaining_budget(
            session,
            job_id=job_id,
        )

        return budget.remaining_seconds <= 0

    @staticmethod
    def _deadline_exceeded() -> ServiceError:
        """Build the standard terminal deadline error."""

        return ServiceError(
            code=CacheJobErrorCode.PROMPT_CACHE_JOB_DEADLINE_EXCEEDED.value,
            message="Cache-generation job exceeded its execution deadline.",
            http_status=408,
            retryable=False,
        )
