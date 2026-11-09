
"""Provider interfaces used by the Prompt Cache Service."""

from typing import Protocol

from ..schemas.cache import CachePayload
from ..schemas.jobs import CacheCreationResult


class CacheProvider(Protocol):
    """Contract for external cached-content providers."""

    async def create_cached_content(
        self,
        *,
        payload: CachePayload,
        timeout_seconds: float,
    ) -> CacheCreationResult:
        """Create cached content and return its normalized result."""
        ...
