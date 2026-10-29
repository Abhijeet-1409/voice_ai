"""Abstract interfaces for prompt-cache clients."""

from abc import ABC, abstractmethod

from .schema import CachePayloadOrDict, CacheResult


class PromptCacheClient(ABC):
    """Define the synchronous interface implemented by prompt-cache clients.

    A concrete implementation is responsible for communicating with the
    prompt-cache service and returning the completed cache result. Submission,
    status polling, HTTP transport, and response parsing belong in the concrete
    implementation rather than in callers.
    """

    @abstractmethod
    def create(self, cache_payload: CachePayloadOrDict) -> CacheResult:
        """Create or obtain a cache and return its final result.

        Args:
            cache_payload: Instructions, guide, tool schema, and cache type
                required by the prompt-cache service.

        Returns:
            The completed cache result returned by the service.

        Raises:
            PromptCacheClientError: If the client cannot communicate with the
                service or cannot parse its response.
            PromptCacheServiceError: If the service reports that cache creation
                failed.

        Concrete implementations should import these exception types from the
        package's ``exceptions`` module and preserve the original exception as
        the cause when re-raising.
        """
        ...
