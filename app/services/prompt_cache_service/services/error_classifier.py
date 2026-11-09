
"""Normalize provider and infrastructure exceptions."""

from __future__ import annotations

import math
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from google.genai.errors import APIError, ClientError, ServerError

from ..schemas.enums import CacheJobErrorCode


@dataclass(frozen=True)
class ClassifiedError:
    """Normalized error information consumed by the retry policy."""

    code: CacheJobErrorCode
    message: str
    retryable: bool
    retry_after_seconds: float | None = None


class ErrorClassifier:
    """Classify external and unexpected exceptions into service errors."""

    _TRANSIENT_EXCEPTIONS = (
        TimeoutError,
        ConnectionError,
        socket.timeout,
        socket.gaierror,
        httpx.TransportError,
    )

    def classify(self, exc: Exception) -> ClassifiedError:
        """Convert an exception into a normalized error classification."""

        # Google Gen AI SDK errors.
        if isinstance(exc, APIError):
            status_code = self._get_http_status_code(exc)

            if status_code is not None:
                return self._classify_http_error(
                    exc,
                    status_code,
                )

        # Transient network and timeout errors.
        if isinstance(exc, self._TRANSIENT_EXCEPTIONS):
            return ClassifiedError(
                code=CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR,
                message="Gemini cache API request failed temporarily.",
                retryable=True,
            )

        # Unknown errors are permanent by default.
        return ClassifiedError(
            code=CacheJobErrorCode.PROMPT_CACHE_INTERNAL_ERROR,
            message="An unexpected internal error occurred.",
            retryable=False,
        )

    def _classify_http_error(
        self,
        exc: APIError,
        status_code: int,
    ) -> ClassifiedError:
        """Map Google Gen AI HTTP errors to service error categories."""

        if status_code == 400:
            return ClassifiedError(
                code=CacheJobErrorCode.PROMPT_CACHE_INVALID_PAYLOAD,
                message="Gemini rejected the cache payload.",
                retryable=False,
            )

        if status_code in (401, 403):
            return ClassifiedError(
                code=CacheJobErrorCode.PROMPT_CACHE_GEMINI_AUTH_ERROR,
                message="Gemini authentication or authorization failed.",
                retryable=False,
            )

        if status_code == 429:
            return ClassifiedError(
                code=CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR,
                message="Gemini API rate limit was reached.",
                retryable=True,
                retry_after_seconds=self._extract_retry_after(exc),
            )

        if 500 <= status_code <= 599 or isinstance(exc, ServerError):
            return ClassifiedError(
                code=CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR,
                message="Gemini cache API encountered a temporary server error.",
                retryable=True,
                retry_after_seconds=self._extract_retry_after(exc),
            )

        if status_code == 404:
            if self._is_unsupported_configuration(exc):
                return ClassifiedError(
                    code=(
                        CacheJobErrorCode
                        .PROMPT_CACHE_UNSUPPORTED_CONFIGURATION
                    ),
                    message=(
                        "The configured Gemini model or location "
                        "is not supported."
                    ),
                    retryable=False,
                )

            return ClassifiedError(
                code=CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR,
                message="Gemini could not find the requested resource.",
                retryable=False,
            )

        # Other HTTP 4xx errors are treated as nonretryable provider errors.
        if 400 <= status_code <= 499:
            return ClassifiedError(
                code=CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR,
                message="Gemini rejected the API request.",
                retryable=False,
            )

        return ClassifiedError(
            code=CacheJobErrorCode.PROMPT_CACHE_INTERNAL_ERROR,
            message="An unexpected Gemini API error occurred.",
            retryable=False,
        )

    @staticmethod
    def _get_http_status_code(exc: APIError) -> int | None:
        """Extract the numeric HTTP status from the SDK exception."""

        # The inspected SDK exposes code and status. Different SDK
        # versions may represent these as integers or strings.
        for attribute in ("code", "status"):
            value = getattr(exc, attribute, None)

            if isinstance(value, int) and not isinstance(value, bool):
                if 100 <= value <= 599:
                    return value

            if isinstance(value, str):
                match = re.match(r"^\s*(\d{3})(?:\s|$)", value)

                if match:
                    status_code = int(match.group(1))
                    if 100 <= status_code <= 599:
                        return status_code

        # Fall back to the attached HTTP response when available.
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)

        if (
            isinstance(response_status, int)
            and not isinstance(response_status, bool)
            and 100 <= response_status <= 599
        ):
            return response_status

        return None

    @staticmethod
    def _is_unsupported_configuration(exc: APIError) -> bool:
        """Check whether a 404 specifically indicates model/location support."""

        message = getattr(exc, "message", None)
        details = getattr(exc, "details", None)

        # Inspect only provider error information for classification.
        # The extracted text is never returned to the API.
        searchable_text = " ".join(
            str(value)
            for value in (message, details)
            if value is not None
        ).lower()

        normalized_text = re.sub(
            r"[_\-]+",
            " ",
            searchable_text,
        )

        unsupported_terms = (
            "not found",
            "does not exist",
            "not supported",
            "unsupported",
            "not available",
        )

        model_pattern = (
            r"\bmodel\b.{0,160}\b("
            + "|".join(re.escape(term) for term in unsupported_terms)
            + r")\b"
        )

        reverse_model_pattern = (
            r"\b("
            + "|".join(re.escape(term) for term in unsupported_terms)
            + r")\b.{0,160}\bmodel\b"
        )

        location_pattern = (
            r"\b(location|region)\b.{0,120}\b("
            + "|".join(re.escape(term) for term in unsupported_terms)
            + r")\b"
        )

        reverse_location_pattern = (
            r"\b("
            + "|".join(re.escape(term) for term in unsupported_terms)
            + r")\b.{0,120}\b(location|region)\b"
        )

        return any(
            re.search(pattern, normalized_text)
            for pattern in (
                model_pattern,
                reverse_model_pattern,
                location_pattern,
                reverse_location_pattern,
            )
        )

    @staticmethod
    def _extract_retry_after(exc: APIError) -> float | None:
        """Extract Retry-After from the provider response when available."""

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)

        if headers is None:
            return None

        try:
            retry_after = headers.get("Retry-After")
        except (AttributeError, TypeError):
            return None

        if not isinstance(retry_after, str) or not retry_after.strip():
            return None

        retry_after = retry_after.strip()

        # Retry-After may be expressed as a number of seconds.
        try:
            seconds = float(retry_after)

            if math.isfinite(seconds) and seconds >= 0:
                return seconds

            return None

        except ValueError:
            pass

        # Alternatively, HTTP allows an absolute HTTP-date.
        try:
            retry_at = parsedate_to_datetime(retry_after)

            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)

            seconds = (
                retry_at - datetime.now(timezone.utc)
            ).total_seconds()

            return max(0.0, seconds)

        except (TypeError, ValueError, OverflowError):
            return None
