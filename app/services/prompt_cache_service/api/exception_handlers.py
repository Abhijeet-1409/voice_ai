"""Exception handlers for the Prompt Cache Service API."""

import re

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.responses import JSONResponse

from shared.logging_setup import get_logger

from ..core.exceptions import ServiceError
from ..schemas.enums import CacheJobErrorCode
from ..schemas.errors import APIErrorDetail, APIErrorResponse


_LOGGER = "prompt_cache_service.api"
logger = get_logger(_LOGGER)


def _get_request_id(request: Request) -> str:
    """Extract a safe request ID for server-side logging."""

    request_id = (
        getattr(request.state, "request_id", None)
        or request.headers.get("x-request-id")
    )

    if not isinstance(request_id, str):
        return "unknown"

    # Prevent excessively long or malformed values in log messages.
    safe_request_id = re.sub(r"[^a-zA-Z0-9_.:-]", "", request_id[:128])

    return safe_request_id or "unknown"


async def service_error_handler(
    request: Request,
    exc: ServiceError,
) -> JSONResponse:
    """Convert a domain ServiceError into a consistent API response."""

    response = APIErrorResponse(
        error=APIErrorDetail(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
        )
    )

    return JSONResponse(
        status_code=exc.http_status,
        content=response.model_dump(),
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError | ValidationError,
) -> JSONResponse:
    """Return a safe and consistent response for validation errors."""

    response = APIErrorResponse(
        error=APIErrorDetail(
            code=CacheJobErrorCode.PROMPT_CACHE_INVALID_PAYLOAD.value,
            message="Request validation failed.",
            retryable=False,
        )
    )

    return JSONResponse(
        status_code=422,
        content=response.model_dump(),
    )


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions without exposing internal details."""

    request_id = _get_request_id(request)

    # Log the traceback without including the exception's potentially
    # sensitive message, which could contain prompt or provider data.
    logger.error(
        "Unhandled API exception. request_id=%s exception_type=%s",
        request_id,
        type(exc).__name__,
        exc_info=(type(exc), None, exc.__traceback__),
    )

    response = APIErrorResponse(
        error=APIErrorDetail(
            code=CacheJobErrorCode.PROMPT_CACHE_INTERNAL_ERROR.value,
            message="An unexpected internal error occurred.",
            retryable=False,
        )
    )

    return JSONResponse(
        status_code=500,
        content=response.model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all Prompt Cache Service API exception handlers."""

    app.add_exception_handler(
        ServiceError,
        service_error_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )

    app.add_exception_handler(
        ValidationError,
        validation_error_handler,
    )

    app.add_exception_handler(
        Exception,
        unexpected_exception_handler,
    )