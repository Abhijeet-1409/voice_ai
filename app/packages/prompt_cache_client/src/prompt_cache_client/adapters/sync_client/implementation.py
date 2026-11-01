from __future__ import annotations

from time import monotonic, sleep
from typing import Any, NoReturn

import httpx
from pydantic import ValidationError

from shared.logging_setup import get_logger

from ...base import PromptCacheClient
from ...constants import CacheJobErrorCode, JobStatus
from ...cache_client_settings import get_prompt_cache_settings
from ...schema import (
    CacheJob,
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
    PromptCacheResponseError,
    PromptCacheServiceError,
    PromptCacheTimeoutError,
)


_LOGGER = "prompt_cache.adapters.sync_client"


class SyncPromptCacheClient(PromptCacheClient):
    """Synchronous HTTP client for the Prompt Cache Service.

    Provides the complete client-side workflow for obtaining a prompt cache:
    validate the cache payload, look up an existing reusable cache, submit a
    cache-creation job when no reusable cache exists, and poll that job until
    it reaches a terminal state.

    Each instance is intended for one prompt-cache operation. The underlying
    ``httpx.Client`` is closed automatically by ``create()`` in a ``finally``
    block, including when validation, HTTP communication, response parsing,
    or job processing raises an exception. Create a new instance for each
    independent operation; do not reuse an instance after ``create()`` returns
    or raises.

    This class is responsible for HTTP transport, response validation,
    polling, and translating service errors into the client's exception
    hierarchy. Cache creation, persistence, locking, and retry execution
    remain responsibilities of the Prompt Cache Service.
    """

    def __init__(
        self,
        base_url: str,
        *,
        lookup_endpoint: str = "/api/v1/caches/lookup",
        create_job_endpoint: str = "/api/v1/cache-jobs",
        get_job_endpoint: str = "/api/v1/cache-jobs/{job_id}",
        request_timeout: float = 10.0,
        poll_interval: float = 2.0,
        poll_timeout: float = 120.0,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero.")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero.")
        if poll_timeout <= 0:
            raise ValueError("poll_timeout must be greater than zero.")
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
        self._lookup_endpoint = lookup_endpoint
        self._create_job_endpoint = create_job_endpoint
        self._get_job_endpoint = get_job_endpoint
        self._request_timeout = request_timeout
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout

        self._logger.info(f"Initialized prompt-cache client for {base_url}.")

    def create(self, payload: CachePayloadOrDict) -> CacheResult:
        """Get a reusable prompt cache or create a new one.

        The operation validates ``payload`` and first asks the service whether
        a reusable cache already exists. On a cache hit, the existing cache
        result is returned immediately. On a cache miss, the client submits a
        creation job and polls its status until the job completes, fails, or
        the configured polling deadline expires.

        The HTTPX client is closed in the ``finally`` block after the operation
        finishes, regardless of its outcome. Consequently, this client
        instance must not be reused after this method has been called.

        Args:
            payload: A ``CachePayload`` instance or a dictionary that can be
                validated as a ``CachePayload``.

        Returns:
            The existing or newly created cache, including its resource name,
            reuse indicator, and expiry information.

        Raises:
            PromptCacheClientError: The payload is invalid, the client is
                closed, or the request configuration is invalid.
            PromptCacheConnectionError: The service cannot be reached due to
                a network or connection failure.
            PromptCacheTimeoutError: An HTTP request or the overall job
                polling operation exceeds its configured deadline.
            PromptCacheResponseError: A service response is invalid,
                unexpected, or inconsistent with the requested job.
            PromptCacheDatabaseError: The service reports a database failure.
            PromptCacheGeminiAPIError: The service reports a Gemini/Vertex AI
                cache-creation failure.
            PromptCacheServiceError: The service reports another job or
                server-side failure.
        """
        try:
            self._logger.info("Starting prompt-cache retrieval or creation.")
            validated_payload = self._validate_payload(payload)

            self._logger.debug(
                f"Looking up reusable cache for type "
                f"'{validated_payload.prompt_cache_type}'."
            )
            cached_result = self._retrieve(validated_payload)

            if cached_result is not None:
                self._logger.info(
                    f"Reusing existing prompt cache '{cached_result.cache_name}'."
                )
                return cached_result

            self._logger.info(
                f"No reusable cache found for type "
                f"'{validated_payload.prompt_cache_type}'; submitting a job."
            )
            job = self._submit_job(validated_payload)
            self._logger.info(
                f"Cache-creation job accepted: job_id={job.id}, "
                f"status={job.status.value}."
            )

            result = self._wait_for_result(job)
            self._logger.info(
                f"Prompt-cache operation completed: cache_name={result.cache_name}."
            )
            return result
        finally:
            self.close()

    def _validate_payload(self, payload: CachePayloadOrDict) -> CachePayload:
        """Normalize supported input forms into a validated Pydantic model."""
        if isinstance(payload, CachePayload):
            return payload

        try:
            return CachePayload.model_validate(payload)
        except ValidationError as exc:
            self._logger.error(f"Invalid prompt-cache payload: {exc}")
            raise PromptCacheClientError(
                f"Invalid prompt-cache payload: {exc}"
            ) from exc

    def _retrieve(self, payload: CachePayload) -> CacheResult | None:
        """Look up a reusable cache; return None for an ordinary cache miss."""
        endpoint = self._lookup_endpoint
        self._logger.debug(f"Sending cache lookup request to {endpoint}.")

        response = self._request(
            "POST",
            endpoint,
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
            f"Cache lookup completed: found '{lookup.cache.cache_name}'."
        )
        return lookup.cache

    def _submit_job(self, payload: CachePayload) -> CacheJob:
        """Submit a cache-creation job and return its initial persisted state."""
        endpoint = self._create_job_endpoint
        self._logger.debug(f"Submitting cache-creation job to {endpoint}.")

        response = self._request(
            "POST",
            endpoint,
            json=payload.model_dump(mode="json"),
        )

        try:
            job = CacheJob.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            self._logger.exception("Invalid cache-job submission response.")
            raise PromptCacheResponseError(
                f"Invalid cache-job submission response: {exc}"
            ) from exc

        self._logger.info(
            f"Received cache job: id={job.id}, status={job.status.value}."
        )
        return job

    def _wait_for_result(self, job: CacheJob) -> CacheResult:
        """Poll the service until the submitted job reaches a terminal state.

        A new GET request is made on each iteration. ``PENDING``, ``RUNNING``,
        and ``RETRYING`` are non-terminal states, so the method waits for the
        configured poll interval and then checks the job again. ``COMPLETED``
        returns the job result. ``FAILED`` is translated into a client
        exception. The overall polling deadline prevents an endless wait.

        Args:
            job: The initial job representation returned by job submission.

        Returns:
            The result produced by the completed cache-creation job.

        Raises:
            PromptCacheTimeoutError: The job does not reach a terminal state
                before the polling deadline.
            PromptCacheResponseError: The polling response is invalid or
                inconsistent with the submitted job.
            PromptCacheServiceError: The job reaches ``FAILED`` and the
                service reports an error.
        """
        endpoint = self._get_job_endpoint.format(job_id=job.id)
        deadline = monotonic() + self._poll_timeout
        last_logged_status: JobStatus | None = None

        self._logger.info(
            f"Waiting for cache job {job.id}; "
            f"poll_interval={self._poll_interval}s, "
            f"poll_timeout={self._poll_timeout}s."
        )

        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                message = (
                    f"Timed out waiting for cache job {job.id} "
                    f"after {self._poll_timeout} seconds."
                )
                self._logger.error(message)
                raise PromptCacheTimeoutError(message)

            # Bound each individual GET by the remaining overall polling time.
            response = self._request(
                "GET",
                endpoint,
                timeout=min(self._request_timeout, remaining),
            )

            try:
                polled_job = CacheJob.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                self._logger.exception(
                    f"Invalid polling response for cache job {job.id}."
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

            # Log state transitions rather than repeating the same log every poll.
            if polled_job.status != last_logged_status:
                self._logger.info(
                    f"Cache job {job.id} status: {polled_job.status.value}."
                )
                last_logged_status = polled_job.status

            match polled_job.status:
                case JobStatus.COMPLETED:
                    if polled_job.result is None:
                        message = (
                            f"Cache job {job.id} is completed but has no result."
                        )
                        self._logger.error(message)
                        raise PromptCacheResponseError(message)

                    return polled_job.result

                case JobStatus.FAILED:
                    if polled_job.error is None:
                        message = (
                            f"Cache job {job.id} failed without error details."
                        )
                        self._logger.error(message)
                        raise PromptCacheResponseError(message)

                    self._raise_job_error(polled_job)

                case JobStatus.PENDING | JobStatus.RUNNING | JobStatus.RETRYING:
                    # Non-terminal state: wait, then GET the latest job state.
                    remaining = deadline - monotonic()
                    if remaining > 0:
                        sleep(min(self._poll_interval, remaining))

                case _:
                    # Defensive fallback if the status model is extended later.
                    raise PromptCacheResponseError(
                        f"Unexpected status for cache job {job.id}: "
                        f"{polled_job.status!r}."
                    )

    def _raise_job_error(self, job: CacheJob) -> NoReturn:
        """Raise the client exception corresponding to a failed service job.

        This method never returns normally: it always raises an exception.

        Args:
            job: A failed job containing the service-reported error code and
                message.

        Raises:
            PromptCacheResponseError: The failed job has no error details.
            PromptCacheDatabaseError: The service reports a database failure.
            PromptCacheGeminiAPIError: The service reports a Gemini API failure.
            PromptCacheServiceError: The service reports an unrecognized error.
        """
        if job.error is None:
            raise PromptCacheResponseError(
                f"Cache job {job.id} failed without an error object."
            )

        message = job.error.message
        code = job.error.code
        self._logger.error(
            f"Cache job {job.id} failed: code={code.value}, message={message}"
        )

        if code == CacheJobErrorCode.PROMPT_CACHE_DATABASE_ERROR:
            raise PromptCacheDatabaseError(message)

        if code == CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR:
            raise PromptCacheGeminiAPIError(message)

        raise PromptCacheServiceError(message)

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> httpx.Response:
        """Send an HTTP request and translate HTTPX exceptions consistently."""
        if self._httpx_client.is_closed:
            raise PromptCacheClientError(
                "The prompt-cache HTTP client is closed."
            )

        try:
            response = self._httpx_client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response

        except httpx.TimeoutException as exc:
            self._logger.exception(
                f"HTTP timeout during {method} {endpoint}: {exc}"
            )
            raise PromptCacheTimeoutError(
                f"The prompt-cache service request timed out: {exc}"
            ) from exc

        except httpx.NetworkError as exc:
            self._logger.exception(
                f"Network failure during {method} {endpoint}: {exc}"
            )
            raise PromptCacheConnectionError(
                f"Network communication with the prompt-cache service failed: {exc}"
            ) from exc

        except httpx.ProxyError as exc:
            self._logger.exception(
                f"Proxy connection failure during {method} {endpoint}: {exc}"
            )
            raise PromptCacheConnectionError(
                f"Unable to communicate through the configured proxy: {exc}"
            ) from exc

        except httpx.RemoteProtocolError as exc:
            self._logger.exception(
                f"Invalid HTTP response from service during {method} {endpoint}: {exc}"
            )
            raise PromptCacheResponseError(
                f"The prompt-cache service returned an invalid HTTP response: {exc}"
            ) from exc

        except httpx.DecodingError as exc:
            self._logger.exception(
                f"Response decoding failed during {method} {endpoint}: {exc}"
            )
            raise PromptCacheResponseError(
                f"Unable to decode the prompt-cache service response: {exc}"
            ) from exc

        except httpx.TooManyRedirects as exc:
            self._logger.exception(
                f"Too many redirects during {method} {endpoint}: {exc}"
            )
            raise PromptCacheClientError(
                f"The prompt-cache request exceeded the redirect limit: {exc}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            self._raise_http_status_error(exc)

        except httpx.InvalidURL as exc:
            self._logger.exception(f"Invalid URL for {method} {endpoint}: {exc}")
            raise PromptCacheClientError(
                f"Invalid prompt-cache service URL: {exc}"
            ) from exc

        except (httpx.UnsupportedProtocol, httpx.LocalProtocolError) as exc:
            self._logger.exception(
                f"Invalid HTTP client configuration during {method} {endpoint}: {exc}"
            )
            raise PromptCacheClientError(
                f"Invalid prompt-cache HTTP request configuration: {exc}"
            ) from exc

        except httpx.RequestError as exc:
            self._logger.exception(
                f"Unexpected HTTPX request failure during {method} {endpoint}: {exc}"
            )
            raise PromptCacheClientError(
                f"The prompt-cache HTTP request failed: {exc}"
            ) from exc

    def _raise_http_status_error(self, exc: httpx.HTTPStatusError) -> NoReturn:
        """Map a non-2xx service response to the appropriate client exception."""
        response = exc.response
        status_code = response.status_code
        body: Any = None

        try:
            body = response.json()
        except ValueError:
            # The response may be plain text or may not contain valid JSON.
            body = None

        detail: Any = body
        if isinstance(body, dict):
            detail = body.get("detail", body.get("error", body))

        if isinstance(detail, dict):
            code = str(detail.get("code", "")).lower()
            message = str(
                detail.get("message")
                or f"Prompt-cache service returned HTTP {status_code}."
            )
        else:
            code = ""
            message = str(detail or response.text[:500] or exc)

        self._logger.error(
            f"Prompt-cache service returned HTTP {status_code}: {message}"
        )

        if code == CacheJobErrorCode.PROMPT_CACHE_DATABASE_ERROR.value:
            raise PromptCacheDatabaseError(message) from exc

        if code == CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR.value:
            raise PromptCacheGeminiAPIError(message) from exc

        if status_code in (408, 504):
            raise PromptCacheTimeoutError(message) from exc

        if status_code in (400, 422):
            raise PromptCacheClientError(message) from exc

        raise PromptCacheServiceError(
            f"Prompt-cache service returned HTTP {status_code}: {message}"
        ) from exc

    def close(self) -> None:
        """Close the HTTPX client and release this operation's network resources.

        ``create()`` calls this method automatically in a ``finally`` block,
        so cleanup occurs after both successful and unsuccessful operations.
        Calling this method explicitly is also safe; if the client is already
        closed, the method returns without doing anything.
        """
        if self._httpx_client.is_closed:
            return

        self._logger.info("Closing prompt-cache HTTP client.")
        try:
            self._httpx_client.close()
        except httpx.CloseError as exc:
            self._logger.exception(f"Failed to close prompt-cache HTTP client: {exc}")
            raise PromptCacheClientError(
                f"Failed to close prompt-cache HTTP client: {exc}"
            ) from exc



def create_prompt_cache_client() -> SyncPromptCacheClient:
    """Create a prompt-cache client configured from application settings.

    Loads the cached, validated ``PromptCacheClientSettings`` and passes the
    service URL, endpoint paths, request timeout, and polling configuration to
    a new ``SyncPromptCacheClient`` instance.

    A new client is created for each call because ``create()`` closes its
    underlying HTTPX client when the prompt-cache operation finishes. Do not
    reuse the returned client for a second ``create()`` operation.

    Returns:
        SyncPromptCacheClient: A newly configured client for one operation.

    Raises:
        ValidationError: If the prompt-cache settings cannot be loaded or
            validated.
        ValueError: If the configured endpoint or timeout values are invalid.
    """

    settings = get_prompt_cache_settings()

    return SyncPromptCacheClient(
        base_url=settings.PROMPT_CACHE_BASE_URL,
        lookup_endpoint=settings.PROMPT_CACHE_LOOKUP_ENDPOINT,
        create_job_endpoint=settings.PROMPT_CACHE_CREATE_JOB_ENDPOINT,
        get_job_endpoint=settings.PROMPT_CACHE_GET_JOB_ENDPOINT,
        request_timeout=settings.PROMPT_CACHE_REQUEST_TIMEOUT,
        poll_interval=settings.PROMPT_CACHE_POLL_INTERVAL,
        poll_timeout=settings.PROMPT_CACHE_POLL_TIMEOUT,
    )
