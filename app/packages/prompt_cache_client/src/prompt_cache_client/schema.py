"""Pydantic and typing models for the prompt-cache client API contract.

These models describe the payloads exchanged between the voice-agent client
and the Prompt Cache Service. They intentionally contain no service-internal
models such as cache validity, refresh-job details, database identifiers, or
worker lease information.
"""

from datetime import datetime
from typing import Any, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.config import PromptCacheType

from .constants import CacheJobErrorCode, JobStatus


class CachePayloadDict(TypedDict):
    """Describe the required keys in a dictionary-form cache payload.

    All four keys are required. ``tool_schema`` may contain an empty list when
    the requested prompt-cache configuration does not use tools.

    Attributes:
        prompt_cache_type: Existing shared prompt-cache type, such as inbound,
            outbound, or default prompt-cache configuration.
        instruction: System instructions that define the voice agent's role,
            behavior, and response guidelines.
        voice_agent_guide: Business-specific and conversational guidance for
            the voice agent.
        tool_schema: JSON-compatible tool declarations. An empty list is valid.
    """

    prompt_cache_type: PromptCacheType
    instruction: str
    voice_agent_guide: str
    tool_schema: list[dict[str, Any]]


class CachePayload(BaseModel):
    """Validate the content used to look up or create a prompt cache.

    The request shape is shared by the cache-lookup and cache-job submission
    endpoints. All fields are required; ``tool_schema`` can be an empty list.
    Instructions and guide content must contain at least one non-whitespace
    character. Unknown fields are rejected to prevent accidental contract drift.

    Attributes:
        prompt_cache_type: Existing shared prompt-cache type. All supported
            values, including ``DEFAULT_PROMPT_CACHE``, remain valid.
        instruction: Non-empty system-level instructions for the voice agent.
        voice_agent_guide: Non-empty business and conversational guidance.
        tool_schema: JSON-compatible tool declarations; ``[]`` is valid.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_cache_type: PromptCacheType = Field(
        description="The existing shared prompt-cache type to use."
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
        description="JSON-compatible tool declarations; an empty list is valid."
    )

    @field_validator("instruction", "voice_agent_guide")
    @classmethod
    def validate_non_whitespace_content(cls, value: str) -> str:
        """Reject instruction or guide values containing only whitespace.

        Args:
            value: The instruction or guide string being validated.

        Returns:
            The original string when it contains non-whitespace content.

        Raises:
            ValueError: If the string is empty after whitespace is stripped.
        """
        if not value.strip():
            raise ValueError("Value must contain at least one non-whitespace character.")
        return value


CachePayloadOrDict = Union[CachePayload, CachePayloadDict]
"""Supported input forms for a prompt-cache request."""


class CacheResult(BaseModel):
    """Describe a reusable or newly created Gemini cached-content resource.

    Attributes:
        cache_name: Fully qualified Gemini/Vertex AI cached-content resource
            name, for example ``cachedContents/abc123``.
        reused: Whether the service returned an existing cache rather than a
            cache created by the current job.
        expires_at: Provider-side expiration timestamp when known. The value
            is expected to be timezone-aware when supplied by the service.
    """

    model_config = ConfigDict(extra="forbid")

    cache_name: str = Field(
        min_length=1,
        description="Fully qualified Gemini cached-content resource name.",
    )
    reused: bool = Field(
        description="Whether the returned cache was reused rather than newly created."
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Timezone-aware cache expiration timestamp, when available.",
    )

    @field_validator("expires_at")
    @classmethod
    def validate_timezone_aware_expiration(
        cls, value: datetime | None
    ) -> datetime | None:
        """Require a timezone offset when an expiration timestamp is supplied.

        Args:
            value: The cache expiration timestamp, or ``None`` if unavailable.

        Returns:
            The original timezone-aware timestamp or ``None``.

        Raises:
            ValueError: If a supplied timestamp is naive (has no UTC offset).
        """
        if value is not None and value.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware.")
        return value


class CacheLookupResponse(BaseModel):
    """Return a reusable cache or indicate an ordinary cache miss.

    The service handles cache validity classification and stale-while-revalidate
    refresh internally. The client does not receive ``VALID``/``EXPIRING``
    state or a refresh job ID: both a valid cache and an expiring-but-still-valid
    cache are returned in the same ``cache`` field and can be used immediately.

    Attributes:
        cache: Existing reusable cache, or ``None`` when the cache is missing
            or expired.

    Notes:
        A cache miss is a normal result, not an error. The endpoint returns
        HTTP 200 for both a hit and a miss.
    """

    model_config = ConfigDict(extra="forbid")

    cache: CacheResult | None = Field(
        default=None,
        description="Reusable cache if one exists and has not expired; otherwise null.",
    )


class CacheJobError(BaseModel):
    """Describe a terminal failure reported for a cache-creation job.

    Attributes:
        code: Machine-readable service error code used for client-side error
            classification.
        message: Human-readable explanation of the failure.
        retryable: Whether the underlying error category is generally
            transient. This is diagnostic metadata; the client must not
            automatically resubmit the job. The service owns retry policy.
    """

    model_config = ConfigDict(extra="forbid")

    code: CacheJobErrorCode = Field(
        description="Machine-readable code identifying the failure.",
    )
    message: str = Field(
        min_length=1,
        description="Human-readable description of the job failure.",
    )
    retryable: bool = Field(
        default=False,
        description=(
            "Whether the underlying error category is generally transient; "
            "the client does not retry the job automatically."
        ),
    )


class CacheJobSubmissionResponse(BaseModel):
    """Acknowledge a newly created job or return an existing active job.

    The create-job endpoint returns this lightweight representation whether
    it creates a new job or finds an existing ``PENDING``, ``RUNNING``, or
    ``RETRYING`` job for the same cache signature.

    Attributes:
        id: Unique identifier of the cache-creation job.
        status: Current job status at the time the service responds.
        created: ``True`` when this request created the job; ``False`` when
            an existing active job was returned.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        description="Unique identifier assigned to the cache-creation job.",
    )
    status: JobStatus = Field(
        description="Current status of the newly created or reused active job.",
    )
    created: bool = Field(
        description="Whether this request created a new job.",
    )

    @field_validator("status")
    @classmethod
    def validate_active_status(cls, value: JobStatus) -> JobStatus:
        """Ensure job submission only returns an active job status.

        Args:
            value: The status returned by the create-job endpoint.

        Returns:
            The active job status.

        Raises:
            ValueError: If the endpoint returns a terminal job status.
        """
        if value in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise ValueError("Job submission must return a nonterminal status.")
        return value


class CacheJobStatusResponse(BaseModel):
    """Return the current state of a cache-creation job.

    Nonterminal responses contain only ``id`` and ``status``. A completed
    response additionally contains ``result``; a failed response additionally
    contains ``error``. The model accepts omitted optional fields for
    nonterminal responses and validates the required terminal payload.

    Attributes:
        id: Unique identifier of the requested cache-creation job.
        status: Current lifecycle status of the job.
        result: Cache details, present only when the job is ``COMPLETED``.
        error: Failure details, present only when the job is ``FAILED``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        description="Unique identifier of the cache-creation job.",
    )
    status: JobStatus = Field(
        description="Current lifecycle status of the cache-creation job.",
    )
    result: CacheResult | None = Field(
        default=None,
        description="Cache result; present only for a completed job.",
    )
    error: CacheJobError | None = Field(
        default=None,
        description="Failure details; present only for a failed job.",
    )

    @model_validator(mode="after")
    def validate_status_payload(self) -> "CacheJobStatusResponse":
        """Ensure result and error fields agree with the job's terminal state.

        Returns:
            The validated job-status response.

        Raises:
            ValueError: If a completed job lacks a result, a failed job lacks
                an error, or a nonterminal response includes terminal data.
        """
        if self.status == JobStatus.COMPLETED:
            if self.result is None:
                raise ValueError("A completed job response must include result.")
            if self.error is not None:
                raise ValueError("A completed job response must not include error.")
        elif self.status == JobStatus.FAILED:
            if self.error is None:
                raise ValueError("A failed job response must include error.")
            if self.result is not None:
                raise ValueError("A failed job response must not include result.")
        elif self.result is not None or self.error is not None:
            raise ValueError(
                "A nonterminal job response must not include result or error."
            )

        return self
