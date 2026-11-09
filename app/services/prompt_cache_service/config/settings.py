"""Prompt Cache Service settings.

This class extends the application's shared settings and adds only the
configuration owned by the Prompt Cache Service.
"""

from functools import cache

from pydantic import Field, model_validator

from shared.config.settings import AppBaseSettings


class PromptCacheSettings(AppBaseSettings):
    """Validated runtime configuration for the Prompt Cache Service."""

    API_V1_PREFIX: str = "/api/v1"

    # Cache lookup and refresh.
    CACHE_REFRESH_THRESHOLD_SECONDS: int = Field(default=300, gt=0)
    CACHE_REDIS_TTL_SAFETY_SECONDS: int = Field(default=5, ge=0)

    # Job execution and provider request budget.
    CACHE_JOB_TIMEOUT_SECONDS: int = Field(default=60, gt=0)
    GEMINI_REQUEST_TIMEOUT_SECONDS: int = Field(default=20, gt=0)
    MAX_CACHE_JOB_ATTEMPTS: int = Field(default=3, ge=1)

    # Delayed job retries. The service owns job-level retries.
    RETRY_BACKOFF_SECONDS: tuple[int, ...] = (2, 5)
    RETRY_JITTER_MAX_SECONDS: float = Field(default=1.0, ge=0)
    MIN_RETRY_REQUEST_BUDGET_SECONDS: int = Field(default=5, gt=0)

    # Worker lease and heartbeat.
    WORKER_LEASE_SECONDS: int = Field(default=90, gt=0)
    WORKER_HEARTBEAT_SECONDS: int = Field(default=20, gt=0)
    WORKER_CRASH_GRACE_SECONDS: int = Field(default=15, ge=0)

    # Transactional outbox publisher.
    OUTBOX_POLL_INTERVAL_SECONDS: float = Field(default=1.0, gt=0)
    OUTBOX_BATCH_SIZE: int = Field(default=10, ge=1)
    OUTBOX_CLAIM_LEASE_SECONDS: int = Field(default=30, gt=0)
    OUTBOX_PUBLISH_FAILURE_BACKOFF_SECONDS: int = Field(default=5, gt=0)

    # Recovery process.
    RECOVERY_INTERVAL_SECONDS: int = Field(default=5, gt=0)
    RECOVERY_BATCH_SIZE: int = Field(default=100, ge=1)

    # Existing LiveKit worker / Vertex AI configuration names.
    PROJECT: str
    LOCATION: str
    GEMINI_MODEL: str
    CACHED_CONTENT_TTL: int = Field(gt=0)

    # Path to the same service-account JSON file used by the LiveKit worker.
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prompt_cache_settings(self) -> "PromptCacheSettings":
        """Validate relationships between related runtime settings."""
        if self.WORKER_HEARTBEAT_SECONDS >= self.WORKER_LEASE_SECONDS:
            raise ValueError(
                "WORKER_HEARTBEAT_SECONDS must be shorter than "
                "WORKER_LEASE_SECONDS."
            )

        if any(delay < 0 for delay in self.RETRY_BACKOFF_SECONDS):
            raise ValueError(
                "RETRY_BACKOFF_SECONDS cannot contain negative values."
            )

        return self


@cache
def get_prompt_cache_settings() -> PromptCacheSettings:
    """Return the cached Prompt Cache Service settings instance."""
    return PromptCacheSettings()
