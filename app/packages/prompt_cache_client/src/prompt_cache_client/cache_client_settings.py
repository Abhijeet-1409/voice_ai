"""Configuration settings for clients communicating with Prompt Cache Service.

The settings extend the shared application configuration so the client uses
consistent environment loading and validation conventions. Only the
Prompt Cache Service URL, endpoint paths, and HTTP operation timing are
specific to this package.
"""

from functools import cache

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from shared.config import AppBaseSettings


class PromptCacheClientSettings(AppBaseSettings):
    """Configure HTTP communication with the Prompt Cache Service.

    Attributes:
        PROMPT_CACHE_BASE_URL: Base URL of the Prompt Cache Service.
        PROMPT_CACHE_LOOKUP_ENDPOINT: Relative path used to look up an
            existing reusable cache.
        PROMPT_CACHE_CREATE_JOB_ENDPOINT: Relative path used to submit a
            cache-creation job.
        PROMPT_CACHE_GET_JOB_ENDPOINT: Relative path template used to retrieve
            a job's current status. It must contain the ``{job_id}``
            placeholder.
        PROMPT_CACHE_REQUEST_TIMEOUT: Maximum duration, in seconds, for an
            individual HTTP request. The client implementation must cap this
            value by the remaining overall operation budget.
        PROMPT_CACHE_POLL_INTERVAL: Delay, in seconds, between consecutive
            job-status polling requests.
        PROMPT_CACHE_OPERATION_TIMEOUT: Maximum duration, in seconds, for the
            complete client-side cache operation, measured from the beginning
            of cache lookup. It includes lookup, job submission, and polling.
        PROMPT_CACHE_POLL_TIMEOUT: Legacy polling-only timeout retained
            temporarily for compatibility with the current client adapters.
            It should be removed once both sync and async adapters use
            ``PROMPT_CACHE_OPERATION_TIMEOUT``.

    Notes:
        The client operation timeout only limits how long the caller waits.
        It does not cancel, fail, or otherwise modify a job running in the
        Prompt Cache Service. The backend enforces its own independent job
        deadline.
    """

    # Prompt Cache Service connection.
    PROMPT_CACHE_BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL of the Prompt Cache Service.",
    )

    # Prompt Cache Service endpoints.
    PROMPT_CACHE_LOOKUP_ENDPOINT: str = Field(
        default="/api/v1/caches/lookup",
        description="Endpoint for looking up an existing prompt cache.",
    )
    PROMPT_CACHE_CREATE_JOB_ENDPOINT: str = Field(
        default="/api/v1/cache-jobs",
        description="Endpoint for submitting a cache-creation job.",
    )
    PROMPT_CACHE_GET_JOB_ENDPOINT: str = Field(
        default="/api/v1/cache-jobs/{job_id}",
        description="Endpoint template for retrieving a cache job's status.",
    )

    # Per-request and overall client timing.
    PROMPT_CACHE_REQUEST_TIMEOUT: float = Field(
        default=3.0,
        gt=0,
        description="Maximum timeout in seconds for an individual HTTP request.",
    )
    PROMPT_CACHE_POLL_INTERVAL: float = Field(
        default=1.0,
        gt=0,
        description="Delay in seconds between job-status polling requests.",
    )
    PROMPT_CACHE_OPERATION_TIMEOUT: float = Field(
        default=15.0,
        gt=0,
        description=(
            "Maximum total time in seconds for lookup, job submission, and "
            "job-status polling. Expiration stops client waiting but does "
            "not cancel the backend job."
        ),
    )

    # Temporary compatibility setting. Remove after both client adapters have
    # been updated to enforce PROMPT_CACHE_OPERATION_TIMEOUT end to end.
    PROMPT_CACHE_POLL_TIMEOUT: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Deprecated polling-only timeout retained for compatibility with "
            "the current client adapters. Replaced by "
            "PROMPT_CACHE_OPERATION_TIMEOUT once adapters are updated."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )


@cache
def get_prompt_cache_settings() -> PromptCacheClientSettings:
    """Return the cached, validated Prompt Cache Client settings.

    Settings are loaded from the configured environment and ``.env`` file
    when this function is first called. Subsequent calls return the same
    settings instance.

    Returns:
        PromptCacheClientSettings: The validated client configuration.

    Raises:
        pydantic.ValidationError: If required inherited settings or configured
            values fail validation.
    """
    return PromptCacheClientSettings()
