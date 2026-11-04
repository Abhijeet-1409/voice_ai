"""Abstract interface for synchronous prompt-cache clients.

Concrete adapters implement the transport-specific details required to
retrieve an existing prompt cache or request its creation from the Prompt
Cache Service. Callers use this interface without depending on HTTPX, API
endpoints, job submission, or status-polling implementation details.
"""

from abc import ABC, abstractmethod

from .schema import CachePayloadOrDict, CacheResult


class PromptCacheClient(ABC):
    """Define the synchronous public contract for prompt-cache clients.

    Implementations encapsulate the complete cache retrieval-or-creation
    workflow and return a validated :class:`CacheResult` when a usable cache
    is obtained. Transport, lookup, job submission, status polling, and
    response parsing are implementation details and are not exposed through
    this interface.
    """

    @abstractmethod
    def create(self, cache_payload: CachePayloadOrDict) -> CacheResult:
        """Retrieve an existing cache or request creation of a new cache.

        The implementation validates the supplied payload, checks for a
        reusable cache, and, when necessary, submits a cache-creation job and
        waits for its terminal result. The precise HTTP requests and polling
        strategy are defined by the concrete adapter.

        Args:
            cache_payload: Cache type, system instruction, voice-agent guide,
                and tool declarations required by the Prompt Cache Service.
                The dictionary form must satisfy the same contract as
                :class:`CachePayload`.

        Returns:
            The cache resource name, whether the cache was reused, and its
            expiration timestamp when supplied by the service.

        Raises:
            PromptCacheClientError: The payload or client configuration is
                invalid, the service cannot be reached, an HTTP request times
                out, or the service response is malformed or unexpected.
            PromptCacheServiceError: The service reports that the requested
                cache operation failed. Concrete service-specific exceptions
                may provide more detail, such as a database or Gemini failure.

        Implementations should preserve the original exception as the cause
        when translating lower-level transport or validation exceptions.
        """
        ...
