
"""Retry decisions for cache-generation jobs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ..config.settings import PromptCacheSettings
from ..schemas.enums import CacheJobErrorCode
from ..schemas.jobs import CacheJobError
from .error_classifier import ClassifiedError


class RandomSource(Protocol):
    """Interface for an injectable random-number generator."""

    def uniform(self, a: float, b: float) -> float:
        """Return a random floating-point number between a and b."""
        ...


@dataclass(frozen=True)
class RetryDecision:
    """Outcome of evaluating whether a failed attempt can be retried."""

    should_retry: bool
    delay_seconds: float | None
    terminal_error: CacheJobError | None


class RetryPolicy:
    """Decide whether and when a failed cache job should be retried."""

    def __init__(
        self,
        settings: PromptCacheSettings,
        random_source: RandomSource,
    ) -> None:
        self.settings = settings
        self.random_source = random_source

    def decide(
        self,
        *,
        error: ClassifiedError,
        attempt_count: int,
        remaining_seconds: float,
    ) -> RetryDecision:
        """Evaluate retryability, attempt limits, and deadline budget.

        attempt_count represents executions that have actually started.
        For example, attempt_count=1 means the initial execution failed
        and we are considering scheduling the second execution.
        """

        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count < 1
        ):
            raise ValueError("attempt_count must be a positive integer.")

        if (
            isinstance(remaining_seconds, bool)
            or not isinstance(remaining_seconds, (int, float))
            or not math.isfinite(remaining_seconds)
        ):
            raise ValueError("remaining_seconds must be a finite number.")

        # 1. Permanent errors must never be retried.
        if not error.retryable:
            return RetryDecision(
                should_retry=False,
                delay_seconds=None,
                terminal_error=self._to_terminal_error(error),
            )

        # 2. Do not exceed the configured total execution attempts.
        if attempt_count >= self.settings.MAX_CACHE_JOB_ATTEMPTS:
            return RetryDecision(
                should_retry=False,
                delay_seconds=None,
                terminal_error=self._to_terminal_error(error),
            )

        # 3. Calculate the delay before the next execution.
        # After attempt 1, the next execution is attempt 2 and uses
        # the first configured backoff value (normally 2 seconds).
        next_attempt_number = attempt_count + 1

        delay_seconds = self._backoff_seconds(
            next_attempt_number=next_attempt_number,
            retry_after_seconds=error.retry_after_seconds,
        )

        # 4. Ensure the retry delay plus the minimum useful provider
        # request budget fits strictly within the remaining deadline.
        required_seconds = (
            delay_seconds
            + self.settings.MIN_RETRY_REQUEST_BUDGET_SECONDS
        )

        if required_seconds >= remaining_seconds:
            return RetryDecision(
                should_retry=False,
                delay_seconds=None,
                terminal_error=self._deadline_error(),
            )

        # 5. Retry is permitted.
        return RetryDecision(
            should_retry=True,
            delay_seconds=delay_seconds,
            terminal_error=None,
        )

    def _backoff_seconds(
        self,
        *,
        next_attempt_number: int,
        retry_after_seconds: float | None,
    ) -> float:
        """Calculate backoff, jitter, and provider Retry-After.

        With defaults:
            Attempt 1 fails -> approximately 2 seconds.
            Attempt 2 fails -> approximately 5 seconds.

        Provider Retry-After takes precedence whenever it is longer
        than the locally calculated delay.
        """

        if next_attempt_number < 2:
            raise ValueError(
                "next_attempt_number must be at least 2."
            )

        configured_backoffs = self.settings.RETRY_BACKOFF_SECONDS

        retry_index = next_attempt_number - 2

        if configured_backoffs:
            # Reuse the final configured delay if additional retry
            # attempts are introduced in the future.
            backoff_index = min(
                retry_index,
                len(configured_backoffs) - 1,
            )
            base_delay = float(configured_backoffs[backoff_index])
        else:
            base_delay = 0.0

        jitter_max = self.settings.RETRY_JITTER_MAX_SECONDS

        jitter = (
            self.random_source.uniform(0.0, jitter_max)
            if jitter_max > 0
            else 0.0
        )

        delay_seconds = base_delay + jitter

        if retry_after_seconds is not None:
            if (
                not math.isfinite(retry_after_seconds)
                or retry_after_seconds < 0
            ):
                raise ValueError(
                    "retry_after_seconds must be finite and nonnegative."
                )

            delay_seconds = max(
                delay_seconds,
                retry_after_seconds,
            )

        return delay_seconds

    @staticmethod
    def _to_terminal_error(
        error: ClassifiedError,
    ) -> CacheJobError:
        """Convert a classified error into the persisted job error."""

        return CacheJobError(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
        )

    @staticmethod
    def _deadline_error() -> CacheJobError:
        """Create the terminal error used when retry budget is exhausted."""

        return CacheJobError(
            code=CacheJobErrorCode.PROMPT_CACHE_JOB_DEADLINE_EXCEEDED,
            message=(
                "Insufficient remaining job time to schedule another "
                "cache-generation attempt."
            ),
            retryable=False,
        )
