"""Public API error response schemas."""

from pydantic import BaseModel, ConfigDict, Field


class APIErrorDetail(BaseModel):
    """Safe, machine-readable details for an API error."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        description="Stable machine-readable error code.",
    )
    message: str = Field(
        min_length=1,
        description="Safe human-readable error message.",
    )
    retryable: bool = Field(
        default=False,
        description="Whether the underlying failure is generally transient.",
    )


class APIErrorResponse(BaseModel):
    """Standard error envelope returned by Prompt Cache Service endpoints."""

    model_config = ConfigDict(extra="forbid")

    error: APIErrorDetail
