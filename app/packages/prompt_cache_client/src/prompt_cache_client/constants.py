from enum import StrEnum


class JobStatus(StrEnum):
    """
    Defines the possible execution states of a prompt-cache job.

    Members:
        PENDING: The job has been submitted and is waiting for execution.
        RUNNING: The job is currently being processed.
        RETRYING: The job encountered a retryable error and is scheduled
            for another execution attempt.
        COMPLETED: The cache-creation job completed successfully.
        FAILED: The job failed after exhausting its retry attempts or
            encountering a non-retryable error.
    """

    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


class CacheJobErrorCode(StrEnum):
    """
    Defines the error codes representing failures during cache-job execution.

    Members:
        PROMPT_CACHE_DATABASE_ERROR: Indicates a database operation
            failed during cache-job execution.
        PROMPT_CACHE_GEMINI_API_ERROR: Indicates a Gemini or Vertex AI
            API operation failed during cache creation.
    """

    PROMPT_CACHE_DATABASE_ERROR = "prompt_cache_database_error"
    PROMPT_CACHE_GEMINI_API_ERROR = "prompt_cache_gemini_api_error"
