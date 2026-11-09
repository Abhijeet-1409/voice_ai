"""Pydantic schemas for cache-creation jobs."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .cache import CacheResult
from .enums import CacheJobErrorCode, JobStatus


class CacheCreationResult(BaseModel):
    """Normalized result returned by the Gemini provider."""

    model_config = ConfigDict(extra="forbid")

    cache_name: str = Field(
        min_length=1,
        description="Fully qualified Gemini cached-content resource name.",
    )
    expires_at: datetime = Field(
        description="Timezone-aware expiration timestamp returned by Gemini.",
    )

    @field_validator("expires_at")
    @classmethod
    def validate_timezone_aware_expiration(cls, value: datetime) -> datetime:
        """Require the provider expiration timestamp to include a timezone."""
        if value.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware.")
        return value


class CacheJobError(BaseModel):
    """Structured error attached to a failed cache-creation job."""

    model_config = ConfigDict(extra="forbid")

    code: CacheJobErrorCode = Field(
        description="Stable machine-readable service error code.",
    )
    message: str = Field(
        min_length=1,
        description="Safe human-readable description of the failure.",
    )
    retryable: bool = Field(
        default=False,
        description=(
            "Whether the underlying error is generally transient. "
            "The client must not automatically resubmit the job."
        ),
    )


class CacheJobSubmissionResponse(BaseModel):
    """Acknowledge a new job or return an existing active job."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        description="Unique identifier assigned to the cache-creation job.",
    )
    status: JobStatus = Field(
        description="Current status of the newly created or reused active job.",
    )
    created: bool = Field(
        description="True when this request created the job; otherwise false.",
    )

    @field_validator("status")
    @classmethod
    def validate_active_status(cls, value: JobStatus) -> JobStatus:
        """Only active jobs may be returned by the submission endpoint."""
        if value in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise ValueError(
                "Job submission must return a nonterminal job status."
            )
        return value


class CacheJobStatusResponse(BaseModel):
    """Return the current state of a cache-creation job.

    Nonterminal responses contain only ``id`` and ``status``. A completed
    response additionally contains ``result``; a failed response additionally
    contains ``error``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        description="Unique identifier of the cache-creation job.",
    )
    status: JobStatus = Field(
        description="Current lifecycle status of the job.",
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
        """Ensure result/error fields agree with the terminal job status."""
        if self.status == JobStatus.COMPLETED:
            if self.result is None:
                raise ValueError(
                    "A completed job response must include result."
                )
            if self.error is not None:
                raise ValueError(
                    "A completed job response must not include error."
                )
        elif self.status == JobStatus.FAILED:
            if self.error is None:
                raise ValueError(
                    "A failed job response must include error."
                )
            if self.result is not None:
                raise ValueError(
                    "A failed job response must not include result."
                )
        elif self.result is not None or self.error is not None:
            raise ValueError(
                "A nonterminal job response must not include result or error."
            )

        return self
