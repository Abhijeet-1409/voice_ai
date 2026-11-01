"""Asynchronous HTTP client for the Prompt Cache Service."""

import asyncio
from time import monotonic
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


_LOGGER = "prompt_cache.adapters.async_client"


class AsyncPromptCacheClient(PromptCacheClient):
    """Asynchronous HTTP client for the Prompt Cache Service.

    Validates a prompt-cache payload, looks up an existing reusable cache,
    submits a cache-creation job on a cache miss, and asynchronously polls
    the job until it completes, fails, or reaches the configured deadline.

    Each instance is intended for one prompt-cache operation. The underlying
    ``httpx.AsyncClient`` is closed automatically by ``create()`` in a
    ``finally`` block. Create a new instance for each independent operation;
    do not reuse an instance after ``create()`` returns or raises.

    All network operations and polling waits are asynchronous. This allows the
    caller's event loop to run other tasks while the client waits for network
    responses or for the background cache-creation job.
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
        self._httpx_client = httpx.AsyncClient(
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

        self._logger.info(
            f"Initialized asynchronous prompt-cache client for {base_url}."
        )

    async def create(self, payload: CachePayloadOrDict) -> CacheResult:
        """Get a reusable prompt cache or asynchronously create a new one.

        Validates ``payload`` and first checks whether a reusable cache exists.
        On a cache hit, returns it immediately. On a cache miss, submits a
        creation job and asynchronously polls its status until completion,
        failure, or expiration of the configured polling deadline.

        The underlying ``httpx.AsyncClient`` is closed in ``finally``,
        regardless of the operation's outcome. This instance must therefore
        not be reused after this method has been called.

        Args:
            payload: A ``CachePayload`` instance or a dictionary that can be
                validated as a ``CachePayload``.

        Returns:
            The existing or newly created cache result.

        Raises:
            PromptCacheClientError: The payload is invalid or the client is
                already closed.
            PromptCacheConnectionError: Network communication fails.
            PromptCacheTimeoutError: An HTTP request or the overall polling
                operation exceeds its deadline.
            PromptCacheResponseError: A service response is invalid or
                inconsistent with the requested job.
            PromptCacheDatabaseError: The service reports a database failure.
            PromptCacheGeminiAPIError: The service reports a Gemini/Vertex AI
                cache-creation failure.
            PromptCacheServiceError: The service reports another failure.
        """
        try:
            self._logger.info("Starting asynchronous prompt-cache operation.")
            validated_payload = self._validate_payload(payload)

            cached_result = await self._retrieve(validated_payload)
            if cached_result is not None:
                self._logger.info(
                    f"Reusing existing prompt cache '{cached_result.cache_name}'."
                )
                return cached_result

            self._logger.info(
                f"No reusable cache found for type "
                f"'{validated_payload.prompt_cache_type}'; submitting a job."
            )
            job = await self._submit_job(validated_payload)
            self._logger.info(
                f"Cache-creation job accepted: job_id={job.id}, "
                f"status={job.status.value}."
            )

            result = await self._wait_for_result(job)
            self._logger.info(
                f"Prompt-cache operation completed: cache_name={result.cache_name}."
            )
            return result
        finally:
            await self.aclose()

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

    async def _retrieve(self, payload: CachePayload) -> CacheResult | None:
        """Look up a reusable cache; return None for an ordinary cache miss."""
        response = await self._request(
            "POST",
            self._lookup_endpoint,
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

    async def _submit_job(self, payload: CachePayload) -> CacheJob:
        """Submit a cache-creation job and return its initial persisted state."""
        response = await self._request(
            "POST",
            self._create_job_endpoint,
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

    async def _get_job_status(
        self,
        job_id: str,
        *,
        timeout: float,
    ) -> CacheJob:
        """Fetch and validate the latest status for a cache-creation job.

        Args:
            job_id: Identifier of the job to retrieve.
            timeout: Maximum duration allowed for this individual GET request.

        Returns:
            The validated, current representation of the requested job.

        Raises:
            PromptCacheResponseError: The response is invalid or contains a
                different job identifier.
        """
        endpoint = self._get_job_endpoint.format(job_id=job_id)
        response = await self._request(
            "GET",
            endpoint,
            timeout=timeout,
        )

        try:
            polled_job = CacheJob.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            self._logger.exception(
                f"Invalid polling response for cache job {job_id}."
            )
            raise PromptCacheResponseError(
                f"Invalid polling response for cache job {job_id}: {exc}"
            ) from exc

        if polled_job.id != job_id:
            message = (
                f"Polling endpoint returned job '{polled_job.id}' "
                f"when '{job_id}' was requested."
            )
            self._logger.error(message)
            raise PromptCacheResponseError(message)

        return polled_job

    async def _wait_for_result(self, job: CacheJob) -> CacheResult:
        """Asynchronously poll until the job reaches a terminal state.

        ``PENDING``, ``RUNNING``, and ``RETRYING`` are non-terminal states.
        The method waits asynchronously between status requests, allowing
        other tasks on the event loop to run. ``COMPLETED`` returns the result;
        ``FAILED`` is translated into a client exception. A monotonic deadline
        bounds the complete polling operation.

        Args:
            job: Initial job representation returned by job submission.

        Returns:
            The result produced by the completed cache-creation job.

        Raises:
            PromptCacheTimeoutError: The job does not reach a terminal state
                before the polling deadline.
            PromptCacheResponseError: The job response is invalid.
            PromptCacheServiceError: The service reports a terminal job failure.
        """
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

            polled_job = await self._get_job_status(
                job.id,
                timeout=min(self._request_timeout, remaining),
            )

            if polled_job.status != last_logged_status:
                self._logger.info(
                    f"Cache job {job.id} status: {polled_job.status.value}."
                )
                last_logged_status = polled_job.status

            match polled_job.status:
                case JobStatus.COMPLETED:
                    if polled_job.result is None:
                        raise PromptCacheResponseError(
                            f"Cache job {job.id} is completed but has no result."
                        )
                    return polled_job.result

                case JobStatus.FAILED:
                    self._raise_job_error(polled_job)

                case JobStatus.PENDING | JobStatus.RUNNING | JobStatus.RETRYING:
                    remaining = deadline - monotonic()
                    if remaining > 0:
                        await asyncio.sleep(
                            min(self._poll_interval, remaining)
                        )

                case _:
                    raise PromptCacheResponseError(
                        f"Unexpected status for cache job {job.id}: "
                        f"{polled_job.status!r}."
                    )

    def _raise_job_error(self, job: CacheJob) -> NoReturn:
        """Raise the client exception corresponding to a failed service job."""
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

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send an asynchronous HTTP request and translate HTTPX exceptions."""
        if self._httpx_client.is_closed:
            raise PromptCacheClientError(
                "The prompt-cache HTTP client is closed."
            )

        try:
            response = await self._httpx_client.request(
                method,
                endpoint,
                **kwargs,
            )
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
                f"Invalid HTTP response during {method} {endpoint}: {exc}"
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

        try:
            body: Any = response.json()
        except ValueError:
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

    async def aclose(self) -> None:
        """Close the AsyncClient and release its network resources.

        ``create()`` invokes this method automatically in its ``finally``
        block. Calling it more than once is safe.
        """
        if self._httpx_client.is_closed:
            return

        self._logger.info("Closing asynchronous prompt-cache HTTP client.")
        try:
            await self._httpx_client.aclose()
        except httpx.CloseError as exc:
            self._logger.exception(
                f"Failed to close prompt-cache HTTP client: {exc}"
            )
            raise PromptCacheClientError(
                f"Failed to close prompt-cache HTTP client: {exc}"
            ) from exc


def create_async_prompt_cache_client() -> AsyncPromptCacheClient:
    """Create a new asynchronous prompt-cache client from application settings.

    Loads the cached, validated prompt-cache settings and configures a fresh
    ``AsyncPromptCacheClient`` with the service URL, endpoint paths, request
    timeout, and polling configuration.

    The factory itself is synchronous because constructing ``httpx.AsyncClient``
    does not perform network I/O. Await the client's ``create()`` method to
    execute the prompt-cache operation.

    Returns:
        A newly configured asynchronous client for one operation.
    """

    settings = get_prompt_cache_settings()

    return AsyncPromptCacheClient(
        base_url=settings.PROMPT_CACHE_BASE_URL,
        lookup_endpoint=settings.PROMPT_CACHE_LOOKUP_ENDPOINT,
        create_job_endpoint=settings.PROMPT_CACHE_CREATE_JOB_ENDPOINT,
        get_job_endpoint=settings.PROMPT_CACHE_GET_JOB_ENDPOINT,
        request_timeout=settings.PROMPT_CACHE_REQUEST_TIMEOUT,
        poll_interval=settings.PROMPT_CACHE_POLL_INTERVAL,
        poll_timeout=settings.PROMPT_CACHE_POLL_TIMEOUT,
    )
