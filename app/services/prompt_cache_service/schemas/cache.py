"""Pydantic schemas for prompt-cache lookup and request payloads."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import PromptCacheType


class CachePayload(BaseModel):
    """Validate the content used to look up or create a prompt cache.

    The same request model is used by the cache-lookup and cache-job
    submission endpoints. All four fields are required. ``tool_schema`` may
    be an empty list, while ``instruction`` and ``voice_agent_guide`` must
    contain at least one non-whitespace character.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_cache_type: PromptCacheType = Field(
        description="The existing shared prompt-cache type to use.",
    )
    instruction: str = Field(
        min_length=1,
        description="System-level instructions for the voice agent.",
    )
    voice_agent_guide: str = Field(
        min_length=1,
        description="Business and conversational guidance for the voice agent.",
    )
    tool_schema: list[dict[str, Any]] = Field(
        description="Ordered JSON-compatible tool declarations; an empty list is valid.",
    )

    @field_validator("instruction", "voice_agent_guide")
    @classmethod
    def validate_non_whitespace_content(cls, value: str) -> str:
        """Reject instruction or guide values containing only whitespace."""
        if not value.strip():
            raise ValueError(
                "Value must contain at least one non-whitespace character."
            )
        return value


class CacheResult(BaseModel):
    """Describe a reusable or newly created Gemini cached-content resource.

    ``expires_at`` is optional for compatibility with cache records where the
    provider expiration timestamp is unavailable. When supplied, it must be
    timezone-aware.
    """

    model_config = ConfigDict(extra="forbid")

    cache_name: str = Field(
        min_length=1,
        description="Fully qualified Gemini cached-content resource name.",
    )
    reused: bool = Field(
        description="Whether the service returned an existing cache.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Timezone-aware cache expiration timestamp, when available.",
    )

    @field_validator("expires_at")
    @classmethod
    def validate_timezone_aware_expiration(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Reject naive expiration timestamps."""
        if value is not None and value.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware.")
        return value


class CacheLookupResponse(BaseModel):
    """Return a reusable cache or indicate an ordinary cache miss.

    Cache validity classification and stale-while-revalidate coordination
    remain internal service concerns. The public response contains only the
    reusable cache or ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    cache: CacheResult | None = Field(
        default=None,
        description="Reusable cache if one exists and has not expired; otherwise null.",
    )
