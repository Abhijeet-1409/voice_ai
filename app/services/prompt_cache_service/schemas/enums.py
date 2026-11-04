"""Domain enums used by the Prompt Cache Service.

PromptCacheType is imported from the shared package so the service and the
rest of the application use one authoritative cache-type definition.
"""

from enum import StrEnum

from shared.config import PromptCacheType


class CacheValidity(StrEnum):
    """Internal cache expiration classifications."""

    VALID = "VALID"
    EXPIRING = "EXPIRING"
    EXPIRED = "EXPIRED"


class JobStatus(StrEnum):
    """Lifecycle states of a prompt-cache creation job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RefreshStatus(StrEnum):
    """Internal status of a cache refresh operation."""

    STARTED = "STARTED"
    ALREADY_RUNNING = "ALREADY_RUNNING"


class CacheJobErrorCode(StrEnum):
    """Stable error codes shared between the client and service."""

    PROMPT_CACHE_DATABASE_ERROR = "prompt_cache_database_error"
    PROMPT_CACHE_GEMINI_API_ERROR = "prompt_cache_gemini_api_error"
    PROMPT_CACHE_INVALID_PAYLOAD = "prompt_cache_invalid_payload"
    PROMPT_CACHE_GEMINI_AUTH_ERROR = "prompt_cache_gemini_auth_error"
    PROMPT_CACHE_UNSUPPORTED_CONFIGURATION = (
        "prompt_cache_unsupported_configuration"
    )
    PROMPT_CACHE_JOB_DEADLINE_EXCEEDED = (
        "prompt_cache_job_deadline_exceeded"
    )
    PROMPT_CACHE_WORKER_INTERRUPTED = "prompt_cache_worker_interrupted"
    PROMPT_CACHE_INTERNAL_ERROR = "prompt_cache_internal_error"
    PROMPT_CACHE_JOB_NOT_FOUND = "prompt_cache_job_not_found"


ACTIVE_JOB_STATUSES = frozenset(
    {
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.RETRYING,
    }
)

TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
    }
)
