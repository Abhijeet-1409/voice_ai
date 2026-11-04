"""Exception hierarchy for the prompt-cache client and service.

The hierarchy separates failures in the HTTP client itself from failures
reported by the Prompt Cache Service while processing a request or job.
Client adapters should translate transport and response failures into the
client-side exceptions, and translate structured service error codes into
the corresponding service-side exceptions.
"""


class PromptCacheError(Exception):
    """Base exception for all prompt-cache client and service errors.

    Args:
        message: Human-readable description of the failure.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Client-side exceptions
# ---------------------------------------------------------------------------


class PromptCacheClientError(PromptCacheError):
    """Base exception for failures originating in the prompt-cache client.

    Use this for invalid client configuration, locally rejected input, or
    other client-side failures that do not represent a structured error
    reported by the Prompt Cache Service.
    """


class PromptCacheConnectionError(PromptCacheClientError):
    """Raised when the client cannot establish or maintain service communication.

    Examples include connection refusal, DNS resolution failure, and other
    network-level errors that prevent a valid HTTP response from being received.
    """


class PromptCacheTimeoutError(PromptCacheClientError):
    """Raised when a client-side HTTP request or overall wait exceeds its deadline.

    This exception describes the client's decision to stop waiting. It does
    not imply that the server-side cache job was cancelled; the backend job
    may continue and complete after the client has timed out.
    """


class PromptCacheResponseError(PromptCacheClientError):
    """Raised when the service response cannot be decoded or validated.

    Examples include malformed JSON, missing required response fields, an
    unexpected job identifier, or a terminal job response without its result
    or error details.
    """


# ---------------------------------------------------------------------------
# Service-side exceptions
# ---------------------------------------------------------------------------


class PromptCacheServiceError(PromptCacheError):
    """Base exception for failures reported by the Prompt Cache Service.

    Concrete subclasses represent structured service error codes. Client
    adapters should raise the most specific matching subclass when parsing a
    failed job response or an error response from an API endpoint.
    """


class PromptCacheDatabaseError(PromptCacheServiceError):
    """Raised when the service reports a required database operation failure.

    This generally indicates that the service could not complete a required
    PostgreSQL operation. It is not a client-side connection-to-service error.
    """


class PromptCacheGeminiAPIError(PromptCacheServiceError):
    """Raised when Gemini or Vertex AI cache creation fails.

    Use this for provider API failures that are not more specifically
    classified as authentication/permission failures. The service's structured
    error response determines whether the underlying failure was retryable.
    """


class PromptCacheGeminiAuthError(PromptCacheGeminiAPIError):
    """Raised when the service reports a Gemini authentication or permission error.

    This is a specialized Gemini API failure and normally represents a
    non-retryable configuration or credential problem until corrected.
    """


class PromptCacheInvalidPayloadError(PromptCacheServiceError):
    """Raised when the service rejects a cache payload as invalid.

    This represents the service's ``PROMPT_CACHE_INVALID_PAYLOAD`` error.
    It is distinct from local Pydantic validation performed before sending
    the request, which should be reported as a client-side error.
    """


class PromptCacheUnsupportedConfigurationError(PromptCacheServiceError):
    """Raised when the service cannot use the requested model or configuration.

    Examples include an unsupported model, an unavailable model/region
    combination, or a cache configuration that the provider does not support.
    """


class PromptCacheJobDeadlineExceededError(PromptCacheServiceError):
    """Raised when the service reports that a cache job exceeded its deadline.

    This is the backend job's overall execution deadline, which includes
    queue waiting and retry delays. It is different from
    :class:`PromptCacheTimeoutError`, where the client simply stops waiting.
    """


class PromptCacheWorkerInterruptedError(PromptCacheServiceError):
    """Raised when the service reports that a cache worker was interrupted.

    The service uses this error when recovery determines that a worker
    stopped making progress and applies the agreed no-automatic-takeover
    policy to the stale job.
    """


class PromptCacheInternalError(PromptCacheServiceError):
    """Raised when the service reports an unexpected internal failure.

    This is the general service-side error for failures that do not have a
    more specific public error classification.
    """


class PromptCacheJobNotFoundError(PromptCacheServiceError):
    """Raised when the requested cache-job identifier does not exist.

    This normally corresponds to an HTTP 404 response from the job-status
    endpoint with the ``PROMPT_CACHE_JOB_NOT_FOUND`` error code.
    """
