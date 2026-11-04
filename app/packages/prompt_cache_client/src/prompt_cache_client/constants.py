from enum import StrEnum


class JobStatus(StrEnum):
    """Represent the lifecycle states of a prompt-cache creation job.

    The enum values are serialized directly in API request and response
    payloads. They therefore use the uppercase values defined by the Prompt
    Cache Service API contract.

    Members:
        PENDING: The job has been persisted and is waiting to be executed.
        RUNNING: A worker currently owns the job and is executing an attempt.
        RETRYING: An attempt failed with a retryable error and another attempt
            has been scheduled within the job's overall deadline.
        COMPLETED: Cache creation succeeded and the result was persisted.
        FAILED: The job reached a terminal failure state. No further attempt
            will be made.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CacheJobErrorCode(StrEnum):
    """Identify errors reported by the Prompt Cache Service.

    These codes are serialized in API error responses and, for failed jobs,
    in the ``error.code`` field returned by the job-status endpoint. The
    client uses them to distinguish database, provider, validation,
    configuration, execution-deadline, worker, and unexpected failures.

    Members:
        PROMPT_CACHE_DATABASE_ERROR: A required PostgreSQL operation failed.
        PROMPT_CACHE_GEMINI_API_ERROR: A Gemini or Vertex AI API operation
            failed for a reason not represented by a more specific provider
            error code, such as a transient network, rate-limit, or server
            error.
        PROMPT_CACHE_INVALID_PAYLOAD: The submitted cache payload failed
            validation.
        PROMPT_CACHE_GEMINI_AUTH_ERROR: Gemini or Vertex AI rejected the
            request because authentication or permissions were invalid.
        PROMPT_CACHE_UNSUPPORTED_CONFIGURATION: The requested model or cache
            configuration is not supported by the provider or service.
        PROMPT_CACHE_JOB_DEADLINE_EXCEEDED: The job exceeded its overall
            execution deadline, including queue and retry time.
        PROMPT_CACHE_WORKER_INTERRUPTED: A worker stopped unexpectedly and
            recovery marked its job as failed.
        PROMPT_CACHE_INTERNAL_ERROR: An unexpected internal service error
            prevented the operation from completing.
        PROMPT_CACHE_JOB_NOT_FOUND: The requested job identifier does not
            identify a job known to the service. This is an API error rather
            than a job lifecycle status.
    """

    PROMPT_CACHE_DATABASE_ERROR = "prompt_cache_database_error"
    PROMPT_CACHE_GEMINI_API_ERROR = "prompt_cache_gemini_api_error"
    PROMPT_CACHE_INVALID_PAYLOAD = "prompt_cache_invalid_payload"
    PROMPT_CACHE_GEMINI_AUTH_ERROR = "prompt_cache_gemini_auth_error"
    PROMPT_CACHE_UNSUPPORTED_CONFIGURATION = "prompt_cache_unsupported_configuration"
    PROMPT_CACHE_JOB_DEADLINE_EXCEEDED = "prompt_cache_job_deadline_exceeded"
    PROMPT_CACHE_WORKER_INTERRUPTED = "prompt_cache_worker_interrupted"
    PROMPT_CACHE_INTERNAL_ERROR = "prompt_cache_internal_error"
    PROMPT_CACHE_JOB_NOT_FOUND = "prompt_cache_job_not_found"
