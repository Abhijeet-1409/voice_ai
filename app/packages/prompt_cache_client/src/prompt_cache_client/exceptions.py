"""Exception hierarchy for the prompt-cache client and service."""


class PromptCacheError(Exception):
    """Base exception for all prompt-cache errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# Client-side exceptions

class PromptCacheClientError(PromptCacheError):
    """Base exception for client-side failures."""


class PromptCacheConnectionError(PromptCacheClientError):
    """Raised when the client cannot connect to the prompt-cache service."""


class PromptCacheTimeoutError(PromptCacheClientError):
    """Raised when an HTTP request or job-status wait exceeds its deadline."""


class PromptCacheResponseError(PromptCacheClientError):
    """Raised when the service response is malformed or unexpected."""


# Service-side exceptions

class PromptCacheServiceError(PromptCacheError):
    """Base exception for failures reported by the prompt-cache service."""


class PromptCacheDatabaseError(PromptCacheServiceError):
    """Raised when a required database operation fails within the service."""


class PromptCacheGeminiAPIError(PromptCacheServiceError):
    """Raised when a Gemini or Vertex AI operation fails within the service."""