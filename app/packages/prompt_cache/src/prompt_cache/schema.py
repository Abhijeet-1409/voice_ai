"""Pydantic request and result models shared by prompt-cache clients."""

from datetime import datetime
from typing import Any, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field

from shared.config import PromptCacheType


class CachePayloadDict(TypedDict):
    """
    Defines the expected structure of a prompt-cache payload
    when represented as a dictionary.

    Attributes:
        prompt_cache_type: Identifies the type of prompt cache to
            retrieve or create.
        instruction: System instructions that define the voice
            agent's role, behavior, and response guidelines.
        voice_agent_guide: Business-specific and conversational
            guidance provided to the voice agent.
        tool_schema: List of tool definitions describing the
            functions available to the voice agent.
    """

    prompt_cache_type: PromptCacheType
    instruction: str
    voice_agent_guide: str
    tool_schema: list[dict[str, Any]]


class CachePayload(BaseModel):
    """Describe the content needed to create or retrieve a prompt cache.

    Attributes:
        prompt_cache_type: Identifies which cache configuration is requested.
        instruction: System-level instructions for the voice agent.
        voice_agent_guide: Static business and conversation guidance to cache.
        tool_schema: JSON-compatible tool declarations accepted by the service.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    prompt_cache_type: PromptCacheType = Field(
        description="The type of prompt cache requested."
    )
    instruction: str = Field(
        min_length=700,
        description="System-level instructions for the voice agent.",
    )
    voice_agent_guide: str = Field(
        min_length=700,
        description="Static business and conversation guidance to cache.",
    )
    tool_schema: list[dict[str, Any]] = Field(
        description="JSON-compatible tool declarations for the requested cache."
    )


CachePayloadOrDict = Union[CachePayload, CachePayloadDict]


class CacheResult(BaseModel):
    """Represent the final result of a prompt-cache request.

    This is a proposed client-facing result contract. Align its field names and
    types with the actual prompt-cache service response before integrating it.

    Attributes:
        cache_name: Fully qualified cached-content resource name.
        reused: Whether the service reused an existing cache rather than
            creating a new one.
        expires_at: Expiration timestamp, if the service provides it.
    """

    model_config = ConfigDict(extra="forbid")

    cache_name: str = Field(
        min_length=1,
        description="Fully qualified cached-content resource name.",
    )
    reused: bool = Field(
        description="Whether an existing cache was reused."
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Cache expiration time, if provided by the service.",
    )


class CacheJob(BaseModel):
    """
    Represents a cache-creation job submitted to the prompt-cache service.

    This schema contains the job identifier and its current status,
    allowing the client to track the job throughout the asynchronous
    cache-creation workflow.

    Attributes:
        job_id: Unique identifier assigned by the service to the
            cache-creation job. Used to query the job's status and
            retrieve its final result.
        status: Current execution state of the job, indicating
            whether it is pending, running, completed, or failed.
    """

    job_id: str = Field(
        ...,
        description="Unique identifier assigned to the cache-creation job.",
    )
    status: str = Field(
        ...,
        description="Current execution status of the cache-creation job.",
    )

    model_config = ConfigDict(extra="forbid")