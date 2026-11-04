from __future__ import annotations

from time import monotonic, sleep
from typing import Any, NoReturn, Type

import httpx
from pydantic import ValidationError

from shared.logging_setup import get_logger

from ...base import PromptCacheClient
from ...constants import CacheJobErrorCode, JobStatus
from ...cache_client_settings import get_prompt_cache_settings
from ...schema import (
    CacheJobStatusResponse,
    CacheJobSubmissionResponse,
    CacheLookupResponse,
    CachePayload,
    CachePayloadOrDict,
    CacheResult,
)
from ...exceptions import (
    PromptCacheClientError,
    PromptCacheConnectionError,
    PromptCacheDatabaseError,
    PromptCacheGeminiAPIError,
    PromptCacheGeminiAuthError,
    PromptCacheInternalError,
    PromptCacheInvalidPayloadError,
    PromptCacheJobDeadlineExceededError,
    PromptCacheJobNotFoundError,
    PromptCacheResponseError,
    PromptCacheServiceError,
    PromptCacheTimeoutError,
    PromptCacheUnsupportedConfigurationError,
    PromptCacheWorkerInterruptedError,
)


_LOGGER = "prompt_cache.adapters.sync_client"

_SERVICE_ERROR_TYPES: dict[str, Type[PromptCacheServiceError]] = {
    CacheJobErrorCode.PROMPT_CACHE_DATABASE_ERROR.value: PromptCacheDatabaseError,
    CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR.value: PromptCacheGeminiAPIError,
    CacheJobErrorCode.PROMPT_CACHE_INVALID_PAYLOAD.value: PromptCacheInvalidPayloadError,
    CacheJobErrorCode.PROMPT_CACHE_GEMINI_AUTH_ERROR.value: PromptCacheGeminiAuthError,
    CacheJobErrorCode.PROMPT_CACHE_UNSUPPORTED_CONFIGURATION.value: (
        PromptCacheUnsupportedConfigurationError
    ),
    CacheJobErrorCode.PROMPT_CACHE_JOB_DEADLINE_EXCEEDED.value: (
        PromptCacheJobDeadlineExceededError
    ),
    CacheJobErrorCode.PROMPT_CACHE_WORKER_INTERRUPTED.value: (
        PromptCacheWorkerInterruptedError
    ),
    CacheJobErrorCode.PROMPT_CACHE_INTERNAL_ERROR.value: PromptCacheInternalError,
    CacheJobErrorCode.PROMPT_CACHE_JOB_NOT_FOUND.value: PromptCacheJobNotFoundError,
}


class SyncPromptCacheClient(PromptCacheClient):
    """Synchronous HTTP client for the Prompt Cache Service.

    The client implements the complete client-side cache retrieval workflow:
    validate the payload, look up an existing cache, submit a job on a cache
    miss, and poll the job until it completes or fails.

    One instance is intended for one ``create`` operation. The underlying
    :class:`httpx.Client` is closed automatically when ``create`` finishes,
    including when the operation raises an exception. Create a new instance
    for each independent cache operation.

    Cache validity classification, background refresh, job creation,
    persistence, provider retries, worker leases, and backend job deadlines
    are owned by the Prompt Cache Service. The client only waits up to its
    configured overall operation timeout; timing out locally does not cancel
    the backend job.

    Attributes:
        base_url: Base URL of the Prompt Cache Service.
        lookup_endpoint: Relative endpoint used to look up a reusable cache.
        create_job_endpoint: Relative endpoint used to submit a cache job.
        get_job_endpoint: Endpoint template used to retrieve a job's status.
        request_timeout: Maximum duration of one HTTP request, further capped
            by the remaining overall operation budget.
        poll_interval: Delay between requests while a job is nonterminal.
        operation_timeout: Maximum duration of lookup, submission, and polling
            combined.
    """

    def __init__(
        self,
        base_url: str,
        *,
        lookup_endpoint: str = "/api/v1/caches/lookup",
        create_job_endpoint: str = "/api/v1/cache-jobs",
        get_job_endpoint: str = "/api/v1/cache-jobs/{job_id}",
        request_timeout: float = 3.0,
        poll_interval: float = 1.0,
        operation_timeout: float = 15.0,
    ) -> None:
        """Initialize a one-operation synchronous HTTP client.

        Args:
            base_url: Absolute base URL of the Prompt Cache Service.
            lookup_endpoint: Relative path for ``POST /caches/lookup``.
            create_job_endpoint: Relative path for ``POST /cache-jobs``.
            get_job_endpoint: Relative path for ``GET /cache-jobs/{job_id}``.
            request_timeout: Positive maximum timeout for an individual HTTP
                request. Every request is additionally limited by the
                remaining overall operation time.
            poll_interval: Positive delay between job-status polling requests.
            operation_timeout: Positive total client-side budget covering
                lookup, job submission, and status polling.

        Raises:
            ValueError: If the URL, endpoint paths, or timeout values are
                invalid.
        """
        if not base_url.strip():
            raise ValueError("base_url must not be empty.")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero.")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero.")
        if operation_timeout <= 0:
            raise ValueError("operation_timeout must be greater than zero.")
        if not lookup_endpoint.strip():
            raise ValueError("lookup_endpoint must not be empty.")
        if not create_job_endpoint.strip():
            raise ValueError("create_job_endpoint must not be empty.")
        if "{job_id}" not in get_job_endpoint:
            raise ValueError(
                "get_job_endpoint must contain the '{job_id}' placeholder."
            )

        self._logger = get_logger(_LOGGER)
        self._httpx_client = httpx.Client(
            base_url=base_url,
            timeout=request_timeout,
            follow_redirects=False,
        )
        self._base_url = base_url
        self._lookup_endpoint = lookup_endpoint
        self._create_job_endpoint = create_job_endpoint
        self._get_job_endpoint = get_job_endpoint
        self._request_timeout = request_timeout
        self._poll_interval = poll_interval
        self._operation_timeout = operation_timeout
        self._logger.info("Initialized prompt-cache client for %s.", base_url)

    def create(self, cache_payload: CachePayloadOrDict) -> CacheResult:
        """Retrieve a reusable cache or request creation of a new cache.

        The payload is validated locally before the overall operation timer
        starts. The client then performs a cache lookup. If the service returns
        a reusable cache, it is returned immediately. Otherwise, the client
        submits a cache-creation job and polls the job-status endpoint until
        the job completes, fails, or the overall operation deadline expires.

        The same deadline is shared by lookup, submission, and every polling
        request; it is not reset between stages. When the client deadline
        expires, the client stops waiting but does not cancel the server-side
        job. The Prompt Cache Service may continue processing that job.

        The underlying HTTPX client is always closed before this method exits.
        Consequently, this client instance must not be reused after calling
        ``create``.

        Args:
            cache_payload: A :class:`CachePayload` instance or dictionary
                containing the required prompt-cache type, instruction,
                voice-agent guide, and tool-schema keys. ``tool_schema`` may
                be an empty list, but the key itself is required.

        Returns:
            The existing or newly created cache resource, including its name,
            reuse indicator, and expiration timestamp when supplied.

        Raises:
            PromptCacheClientError: The payload is invalid locally, the client
                is closed, or the HTTP request configuration is invalid.
            PromptCacheConnectionError: The service cannot be reached because
                of a network or connection failure.
            PromptCacheTimeoutError: An individual request or the overall
                client-side operation exceeds its configured deadline. This
                does not indicate that the backend job was cancelled.
            PromptCacheResponseError: A response is malformed, violates the
                API schema, or refers to an unexpected job identifier.
            PromptCacheServiceError: The service reports a job or request
                failure. More specific subclasses identify known error codes.
        """
        try:
            self._logger.info("Starting prompt-cache retrieval or creation.")
            validated_payload = self._validate_payload(cache_payload)

            # One monotonic deadline is shared across all network stages.
            deadline = monotonic() + self._operation_timeout
            self._logger.debug(
                "Looking up reusable cache for type '%s'.",
                validated_payload.prompt_cache_type,
            )
            cached_result = self._retrieve(validated_payload, deadline)

            if cached_result is not None:
                self._logger.info(
                    "Reusing existing prompt cache '%s'.", cached_result.cache_name
                )
                return cached_result

            self._logger.info(
                "No reusable cache found for type '%s'; submitting a job.",
                validated_payload.prompt_cache_type,
            )
            job = self._submit_job(validated_payload, deadline)
            self._logger.info(
                "Cache job accepted: job_id=%s, status=%s, created=%s.",
                job.id,
                job.status.value,
                job.created,
            )

            result = self._wait_for_result(job, deadline)
            self._logger.info(
                "Prompt-cache operation completed: cache_name=%s.",
                result.cache_name,
            )
            return result
        finally:
            self.close()

    def _validate_payload(self, payload: CachePayloadOrDict) -> CachePayload:
        """Normalize an accepted payload into the validated Pydantic model.

        Args:
            payload: A Pydantic payload or dictionary-form payload.

        Returns:
            A validated :class:`CachePayload` instance.

        Raises:
            PromptCacheClientError: If local payload validation fails.
        """
        if isinstance(payload, CachePayload):
            return payload

        try:
            return CachePayload.model_validate(payload)
        except ValidationError as exc:
            self._logger.error("Invalid prompt-cache payload: %s", exc)
            raise PromptCacheClientError(
                f"Invalid prompt-cache payload: {exc}"
            ) from exc

    def _retrieve(
        self, payload: CachePayload, deadline: float
    ) -> CacheResult | None:
        """Look up a reusable cache, returning ``None`` for an ordinary miss.

        The service may transparently initiate a background refresh for a
        cache that is approaching expiration. The client does not need to know
        whether a returned cache is classified internally as valid or expiring;
        either cache is still usable and is returned through the same field.

        Args:
            payload: Validated cache lookup payload.
            deadline: Absolute monotonic deadline shared by the full operation.

        Returns:
            A reusable cache result, or ``None`` when the cache is missing or
            already expired.

        Raises:
            PromptCacheTimeoutError: The overall operation deadline expires.
            PromptCacheConnectionError: The service cannot be reached.
            PromptCacheResponseError: The lookup response is invalid.
            PromptCacheServiceError: The service reports a structured failure.
        """
        self._logger.debug("Sending cache lookup request to %s.", self._lookup_endpoint)
        response = self._request(
            "POST",
            self._lookup_endpoint,
            deadline=deadline,
            json=payload.model_dump(mode="json"),
        )

        try:
            lookup = CacheLookupResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            self._logger.exception("Invalid cache-lookup response.")
            raise PromptCacheResponseError(
                f"Invalid cache-lookup response: {exc}"
            ) from exc

        if lookup.cache is None:
            self._logger.info("Cache lookup completed: no reusable cache found.")
            return None

        self._logger.info(
            "Cache lookup completed: found '%s'.", lookup.cache.cache_name
        )
        return lookup.cache

    def _submit_job(
        self, payload: CachePayload, deadline: float
    ) -> CacheJobSubmissionResponse:
        """Submit a cache-creation job and parse its submission acknowledgement.

        The service may create a new job or return an existing active job for
        the same cache signature. The ``created`` field indicates which case
        occurred; both cases use the same subsequent status-polling workflow.

        Args:
            payload: Validated cache creation payload.
            deadline: Absolute monotonic deadline shared by the full operation.

        Returns:
            The job identifier, its initial active status, and whether this
            request created the job.

        Raises:
            PromptCacheTimeoutError: The overall operation deadline expires.
            PromptCacheConnectionError: The service cannot be reached.
            PromptCacheResponseError: The submission response is invalid.
            PromptCacheServiceError: The service reports a structured failure.
        """
        self._logger.debug(
            "Submitting cache-creation job to %s.", self._create_job_endpoint
        )
        response = self._request(
            "POST",
            self._create_job_endpoint,
            deadline=deadline,
            json=payload.model_dump(mode="json"),
        )

        try:
            job = CacheJobSubmissionResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            self._logger.exception("Invalid cache-job submission response.")
            raise PromptCacheResponseError(
                f"Invalid cache-job submission response: {exc}"
            ) from exc

        return job

    def _wait_for_result(
        self, job: CacheJobSubmissionResponse, deadline: float
    ) -> CacheResult:
        """Poll a job until completion, failure, or the shared deadline.

        Each GET request retrieves only the job's current status while the job
        is nonterminal. A ``COMPLETED`` response must include a cache result; a
        ``FAILED`` response must include structured error information. The
        overall deadline is the same deadline established before cache lookup;
        it is not restarted when job polling begins.

        Args:
            job: Job acknowledgement returned by the create-job endpoint.
            deadline: Absolute monotonic deadline for the complete client
                operation, including lookup and job submission.

        Returns:
            The cache result persisted by the service for the completed job.

        Raises:
            PromptCacheTimeoutError: The complete client operation exceeds its
                configured deadline.
            PromptCacheResponseError: A polling response is malformed, belongs
                to a different job, or violates terminal-state requirements.
            PromptCacheServiceError: The job reaches ``FAILED`` and the service
                reports a structured failure.
        """
        endpoint = self._get_job_endpoint.format(job_id=job.id)
        last_logged_status: JobStatus | None = None
        self._logger.info(
            "Waiting for cache job %s; poll_interval=%ss.",
            job.id,
            self._poll_interval,
        )

        while True:
            response = self._request("GET", endpoint, deadline=deadline)

            try:
                polled_job = CacheJobStatusResponse.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                self._logger.exception(
                    "Invalid polling response for cache job %s.", job.id
                )
                raise PromptCacheResponseError(
                    f"Invalid polling response for cache job {job.id}: {exc}"
                ) from exc

            if polled_job.id != job.id:
                message = (
                    f"Polling endpoint returned job '{polled_job.id}' "
                    f"when '{job.id}' was requested."
                )
                self._logger.error(message)
                raise PromptCacheResponseError(message)

            if polled_job.status != last_logged_status:
                self._logger.info(
                    "Cache job %s status: %s.", job.id, polled_job.status.value
                )
                last_logged_status = polled_job.status

            match polled_job.status:
                case JobStatus.COMPLETED:
                    # CacheJobStatusResponse validates that result is present.
                    assert polled_job.result is not None
                    return polled_job.result

                case JobStatus.FAILED:
                    # CacheJobStatusResponse validates that error is present.
                    assert polled_job.error is not None
                    self._raise_service_error(
                        polled_job.error.code,
                        polled_job.error.message,
                    )

                case JobStatus.PENDING | JobStatus.RUNNING | JobStatus.RETRYING:
                    remaining = self._remaining_time(
                        deadline,
                        operation=f"waiting for cache job {job.id}",
                    )
                    sleep(min(self._poll_interval, remaining))

                case _:
                    raise PromptCacheResponseError(
                        f"Unexpected status for cache job {job.id}: "
                        f"{polled_job.status!r}."
                    )

    def _remaining_time(self, deadline: float, *, operation: str) -> float:
        """Return the remaining operation budget or raise on deadline expiry.

        Args:
            deadline: Absolute deadline expressed in ``time.monotonic`` units.
            operation: Short description used in the timeout message.

        Returns:
            Positive number of seconds remaining before the operation deadline.

        Raises:
            PromptCacheTimeoutError: No time remains in the overall operation
                budget. The backend job, if already submitted, is not cancelled.
        """
        remaining = deadline - monotonic()
        if remaining <= 0:
            message = (
                f"Prompt-cache client operation exceeded its configured "
                f"{self._operation_timeout}-second deadline while {operation}. "
                "The backend job, if already submitted, was not cancelled."
            )
            self._logger.warning(message)
            raise PromptCacheTimeoutError(message)
        return remaining

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        deadline: float,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send one HTTP request within the remaining overall time budget.

        The effective HTTP timeout is the smaller of the configured per-request
        timeout and the remaining overall operation budget. This prevents an
        individual lookup, submission, or polling request from extending the
        complete client operation beyond its deadline.

        Args:
            method: HTTP method, such as ``POST`` or ``GET``.
            endpoint: Relative service endpoint.
            deadline: Absolute monotonic deadline for the complete operation.
            **kwargs: Additional keyword arguments passed to
                :meth:`httpx.Client.request`.

        Returns:
            The successful HTTP response.

        Raises:
            PromptCacheClientError: The HTTPX client is already closed or the
                request configuration is invalid.
            PromptCacheConnectionError: A network or connection failure occurs.
            PromptCacheTimeoutError: The request or overall operation times out.
            PromptCacheResponseError: The service returns an invalid protocol
                response.
            PromptCacheServiceError: The service returns a non-success status.
        """
        if self._httpx_client.is_closed:
            raise PromptCacheClientError("The prompt-cache HTTP client is closed.")

        remaining = self._remaining_time(
            deadline,
            operation=f"sending {method} {endpoint}",
        )
        kwargs["timeout"] = min(self._request_timeout, remaining)

        try:
            response = self._httpx_client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response

        except httpx.TimeoutException as exc:
            self._logger.warning("HTTP timeout during %s %s: %s", method, endpoint, exc)
            raise PromptCacheTimeoutError(
                f"The prompt-cache service request timed out: {exc}"
            ) from exc

        except httpx.ProxyError as exc:
            self._logger.exception(
                "Proxy connection failure during %s %s: %s", method, endpoint, exc
            )
            raise PromptCacheConnectionError(
                f"Unable to communicate through the configured proxy: {exc}"
            ) from exc

        except httpx.NetworkError as exc:
            self._logger.exception(
                "Network failure during %s %s: %s", method, endpoint, exc
            )
            raise PromptCacheConnectionError(
                f"Network communication with the prompt-cache service failed: {exc}"
            ) from exc

        except httpx.RemoteProtocolError as exc:
            self._logger.exception(
                "Invalid HTTP response during %s %s: %s", method, endpoint, exc
            )
            raise PromptCacheResponseError(
                f"The prompt-cache service returned an invalid HTTP response: {exc}"
            ) from exc

        except httpx.DecodingError as exc:
            self._logger.exception(
                "Response decoding failed during %s %s: %s", method, endpoint, exc
            )
            raise PromptCacheResponseError(
                f"Unable to decode the prompt-cache service response: {exc}"
            ) from exc

        except httpx.TooManyRedirects as exc:
            self._logger.exception(
                "Too many redirects during %s %s: %s", method, endpoint, exc
            )
            raise PromptCacheClientError(
                f"The prompt-cache request exceeded the redirect limit: {exc}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            self._raise_http_status_error(exc)

        except httpx.InvalidURL as exc:
            self._logger.exception("Invalid URL for %s %s: %s", method, endpoint, exc)
            raise PromptCacheClientError(
                f"Invalid prompt-cache service URL: {exc}"
            ) from exc

        except (httpx.UnsupportedProtocol, httpx.LocalProtocolError) as exc:
            self._logger.exception(
                "Invalid HTTP configuration during %s %s: %s", method, endpoint, exc
            )
            raise PromptCacheClientError(
                f"Invalid prompt-cache HTTP request configuration: {exc}"
            ) from exc

        except httpx.RequestError as exc:
            self._logger.exception(
                "Unexpected HTTPX request failure during %s %s: %s",
                method,
                endpoint,
                exc,
            )
            raise PromptCacheClientError(
                f"The prompt-cache HTTP request failed: {exc}"
            ) from exc

    def _raise_http_status_error(self, exc: httpx.HTTPStatusError) -> NoReturn:
        """Translate an HTTP error response into a typed client exception.

        Structured service error codes take precedence over HTTP status-based
        fallbacks. This allows the service to return a precise error category
        in a consistent response envelope.

        Args:
            exc: HTTPX exception containing the unsuccessful response.

        Raises:
            PromptCacheServiceError: A structured or generic service failure.
            PromptCacheTimeoutError: The HTTP response indicates a request
                timeout or gateway timeout without a more specific error code.
        """
        response = exc.response
        status_code = response.status_code

        try:
            body: Any = response.json()
        except ValueError:
            body = None

        detail: Any = body
        if isinstance(body, dict):
            detail = body.get("detail", body.get("error", body))

        code = ""
        if isinstance(detail, dict):
            code = str(detail.get("code", "")).lower()
            message = str(
                detail.get("message")
                or detail.get("detail")
                or f"Prompt-cache service returned HTTP {status_code}."
            )
        else:
            message = str(detail or response.text[:500] or exc)

        self._logger.error(
            "Prompt-cache service returned HTTP %s: %s", status_code, message
        )

        if code:
            self._raise_service_error(code, message)

        if status_code in (408, 504):
            raise PromptCacheTimeoutError(message) from exc
        if status_code == 404:
            raise PromptCacheJobNotFoundError(message) from exc
        if status_code in (400, 422):
            raise PromptCacheInvalidPayloadError(message) from exc

        raise PromptCacheServiceError(
            f"Prompt-cache service returned HTTP {status_code}: {message}"
        ) from exc

    def _raise_service_error(self, code: str | CacheJobErrorCode, message: str) -> NoReturn:
        """Raise the service exception corresponding to a structured error code.

        Args:
            code: Service error code from the API response or failed job.
            message: Human-readable service-provided error message.

        Raises:
            PromptCacheServiceError: Always raises the most specific known
                service exception, or the generic service exception when the
                code is unknown.
        """
        code_value = code.value if isinstance(code, CacheJobErrorCode) else str(code).lower()
        exception_type = _SERVICE_ERROR_TYPES.get(code_value)
        if exception_type is None:
            raise PromptCacheServiceError(
                f"Prompt-cache service reported an unrecognized error "
                f"({code_value}): {message}"
            )
        raise exception_type(message)

    def close(self) -> None:
        """Close the HTTPX client and release its network resources.

        ``create`` invokes this method in a ``finally`` block. Calling it more
        than once is safe; if the HTTPX client is already closed, this method
        returns without performing additional work.

        Raises:
            PromptCacheClientError: If HTTPX cannot close the client cleanly.
        """
        if self._httpx_client.is_closed:
            return

        self._logger.debug("Closing prompt-cache HTTP client.")
        try:
            self._httpx_client.close()
        except httpx.CloseError as exc:
            self._logger.exception("Failed to close prompt-cache HTTP client: %s", exc)
            raise PromptCacheClientError(
                f"Failed to close prompt-cache HTTP client: {exc}"
            ) from exc


def create_prompt_cache_client() -> SyncPromptCacheClient:
    """Create a one-operation synchronous client from cached settings.

    The factory loads the validated client settings and constructs a new
    :class:`SyncPromptCacheClient`. A new instance is returned for each call
    because ``create`` closes the underlying HTTPX client when its operation
    finishes.

    Returns:
        A synchronous client configured with the service URL, endpoint paths,
        per-request timeout, polling interval, and overall operation timeout.

    Raises:
        pydantic.ValidationError: If inherited or client-specific settings
            cannot be validated.
        ValueError: If the configured URL, endpoints, or timeout values are
            invalid for the client constructor.
    """
    settings = get_prompt_cache_settings()

    return SyncPromptCacheClient(
        base_url=settings.PROMPT_CACHE_BASE_URL,
        lookup_endpoint=settings.PROMPT_CACHE_LOOKUP_ENDPOINT,
        create_job_endpoint=settings.PROMPT_CACHE_CREATE_JOB_ENDPOINT,
        get_job_endpoint=settings.PROMPT_CACHE_GET_JOB_ENDPOINT,
        request_timeout=settings.PROMPT_CACHE_REQUEST_TIMEOUT,
        poll_interval=settings.PROMPT_CACHE_POLL_INTERVAL,
        operation_timeout=settings.PROMPT_CACHE_OPERATION_TIMEOUT,
    )
