"""Domain exceptions for the Prompt Cache Service."""

from dataclasses import dataclass

from ..schemas.enums import CacheJobErrorCode


@dataclass(eq=False)
class ServiceError(Exception):
    """Expected service failure with a stable public error representation.

    Do not put raw provider responses, stack traces, credentials, or prompt
    content in ``message``. Log internal exception details at the boundary
    where they are caught, while returning only this safe message to clients.
    """

    code: CacheJobErrorCode | str
    message: str
    http_status: int = 500
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("ServiceError message must not be empty.")
        if not 400 <= self.http_status <= 599:
            raise ValueError("ServiceError http_status must be between 400 and 599.")
        Exception.__init__(self, self.message)
