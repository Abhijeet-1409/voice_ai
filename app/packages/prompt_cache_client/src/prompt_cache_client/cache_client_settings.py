"""Configuration settings for the synchronous prompt-cache client."""

from functools import cache

from pydantic import Field
from pydantic_settings import SettingsConfigDict

# Adjust this import to match the location of your shared AppBaseSettings.
from shared.config import AppBaseSettings


class PromptCacheClientSettings(AppBaseSettings):
    """
    Settings for connecting to and communicating with the Prompt Cache Service.

    Inherits the shared application settings, including Redis, database, and
    logging configuration.

    Attributes:
        PROMPT_CACHE_BASE_URL (str): Base URL of the Prompt Cache Service.
        PROMPT_CACHE_LOOKUP_ENDPOINT (str): API path used to look up an
            existing reusable prompt cache.
        PROMPT_CACHE_CREATE_JOB_ENDPOINT (str): API path used to submit a
            prompt-cache creation job.
        PROMPT_CACHE_GET_JOB_ENDPOINT (str): API path used to retrieve a job's
            status. Must contain the ``{job_id}`` placeholder.
        PROMPT_CACHE_REQUEST_TIMEOUT (float): Maximum duration, in seconds,
            allowed for an individual HTTP request.
        PROMPT_CACHE_POLL_INTERVAL (float): Delay, in seconds, between job
            status polling requests.
        PROMPT_CACHE_POLL_TIMEOUT (float): Maximum total duration, in seconds,
            allowed for polling a prompt-cache creation job.
    """

    # Prompt Cache Service connection
    PROMPT_CACHE_BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL of the Prompt Cache Service.",
    )

    # Prompt Cache Service endpoints
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
        description="Endpoint template for retrieving a cache job.",
    )

    # HTTP request and polling configuration
    PROMPT_CACHE_REQUEST_TIMEOUT: float = Field(
        default=10.0,
        gt=0,
        description="Timeout in seconds for an individual HTTP request.",
    )
    PROMPT_CACHE_POLL_INTERVAL: float = Field(
        default=2.0,
        gt=0,
        description="Delay in seconds between job status polling requests.",
    )
    PROMPT_CACHE_POLL_TIMEOUT: float = Field(
        default=120.0,
        gt=0,
        description="Maximum total duration in seconds for job polling.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )


@cache
def get_prompt_cache_settings() -> PromptCacheClientSettings:
    """
    Get the validated prompt-cache client settings.

    Settings are loaded from environment variables and the configured .env
    file when this function is first called. The result is cached so
    subsequent calls return the same settings instance.

    Returns:
        PromptCacheClientSettings: Validated prompt-cache client settings.
    """
    return PromptCacheClientSettings()
