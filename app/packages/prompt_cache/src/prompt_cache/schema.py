"""Pydantic request and result models shared by prompt-cache clients."""

from datetime import datetime
from typing import Any, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field

from shared.config import PromptCacheType

from .constants import CacheJobErrorCode, JobStatus


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


class CacheJobError(BaseModel):
    """
    Represents an error encountered during the execution of a
    prompt-cache creation job.

    Attributes:
        code: Machine-readable error code identifying the type
            of failure. Used by the client to map the error to
            the appropriate exception. Must be a member of
            CacheJobErrorCode.
        message: Human-readable description of the error that
            occurred during job execution.
    """

    code: CacheJobErrorCode = Field(
        ...,
        description="Machine-readable code identifying the job failure.",
    )
    message: str = Field(
        ...,
        description="Description of the error encountered during job execution.",
    )

    model_config = ConfigDict(extra="forbid")


class CacheJob(BaseModel):
    """
    Represents the current state and execution details of a
    cache-creation job managed by the prompt-cache service.

    Attributes:
        id: Unique identifier assigned by the service to the
            cache-creation job. Used to retrieve its status and
            track its execution.
        status: Current execution state of the job, indicating
            whether it is pending, running, completed, or failed.
        result: Cache creation result containing the cached-content
            resource details. Available when the job completes
            successfully.
        error: Error details containing the error code and message.
            Available when the job fails.
        created_at: Timestamp indicating when the job was created.
        updated_at: Timestamp indicating when the job was last updated.
        completed_at: Timestamp indicating when the job reached a
            terminal state. None while the job is pending or running.
    """

    id: str = Field(
        ...,
        description="Unique identifier assigned to the cache-creation job.",
    )
    status: JobStatus = Field(
        ...,
        description="Current execution status of the cache-creation job.",
    )
    result: CacheResult | None = Field(
        default=None,
        description="Cache details returned when the job completes successfully.",
    )
    error: CacheJobError | None = Field(
        default=None,
        description="Error details returned when the job fails.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp indicating when the job was created.",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp indicating when the job was last updated.",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp indicating when the job reached a terminal state.",
    )

    model_config = ConfigDict(extra="forbid")