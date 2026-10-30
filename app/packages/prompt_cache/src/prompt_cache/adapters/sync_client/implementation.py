from time import sleep
from typing import Optional, Any

from httpx import Client, Response, HTTPStatusError

from pydantic import ValidationError

from shared.logging_setup import get_logger

from ...base import PromptCacheClient
from ...constants import JobStatus, CacheJobErrorCode
from ...schema import (
    CachePayload,
    CachePayloadOrDict, 
    CacheResult, 
    CacheJob
)
from ...exceptions import (
    PromptCacheClientError,
    PromptCacheConnectionError,
    PromptCacheTimeoutError,
    PromptCacheResponseError,
    PromptCacheServiceError,
    PromptCacheDatabaseError,
    PromptCacheGeminiAPIError
)


_LOGGER = "prompt_cache.adapters.sync_client"


class SyncPromptCacheClient(PromptCacheClient):

    def __init__(self, base_url: str) -> None:
        self._logger = get_logger(_LOGGER)
        self._httpx_client = Client(base_url=base_url)
        self._attempt_interval = 2
        self._attempt = 10

    def create(self, payload: CachePayloadOrDict) -> CacheResult:
        """
        Retrieve an existing cache or create a new one if none is available.

        Attempts to retrieve a reusable cache from the prompt-cache service.
        If no reusable cache exists, submits a cache-creation job and waits
        for the job to complete.

        Args:
            payload: Cache configuration supplied as a CachePayload model
                or a dictionary matching the CachePayloadDict structure.

        Returns:
            CacheResult: The existing or newly created cache result.

        Raises:
            PromptCacheClientError: If input validation or a client-side
                operation fails.
            PromptCacheConnectionError: If the client cannot connect to
                the prompt-cache service.
            PromptCacheTimeoutError: If an HTTP request or job-status
                polling exceeds its deadline.
            PromptCacheResponseError: If the service returns a malformed
                or unexpected response.
            PromptCacheServiceError: If the service reports a cache-related
                failure.
            PromptCacheDatabaseError: If a required database operation
                fails within the service.
            PromptCacheGeminiAPIError: If a Gemini or Vertex AI operation
                fails during cache creation.
        """
        self._logger.info("Starting prompt-cache retrieval or creation.")

        try:
            if not isinstance(payload, CachePayload):
                try:
                    validated_payload = CachePayload.model_validate(payload)
                except ValidationError as exc:
                    self._logger.error(f"Invalid prompt-cache payload: {exc}")
                    raise PromptCacheClientError(
                        f"Invalid prompt-cache payload: {exc}"
                    ) from exc
            else:
                validated_payload = payload

            self._logger.debug("Attempting to retrieve an existing cache.")
            result: Optional[CacheResult] = self._retrieve(validated_payload)

            if result is not None:
                self._logger.info("Reusing an existing prompt cache.")
                return result

            self._logger.info(
                "No reusable cache found; submitting a creation job."
            )
            job: CacheJob = self._submit_job(validated_payload)

            self._logger.info(
                f"Cache-creation job submitted successfully. job_id={job.job_id}"
            )

            result: CacheResult = self._wait_for_result(job)

            self._logger.info(
                f"Prompt-cache creation completed successfully. job_id={job.job_id}"
            )

            return result

        finally:
            self._close()

    def _retrieve(self, payload: CachePayload) -> Optional[CacheResult]:
        """
        """
        _endpoint = "/api/v1/caches/lookup"

        try:
        
            res: Response = self._httpx_client.post(
                _endpoint,
                json=payload.model_dump_json()
            )

            res.raise_for_status()

            content = res.json()

            if content['cache'] is None:
                return None

            return  CacheResult(**content['cache'])  

        except HTTPStatusError as http_err: 
            self._logger.exception()
            raise PromptCacheServiceError()

        except ValidationError as valid_err:
            self._logger.exception()
            raise PromptCacheResponseError()

    def _submit_job(self, payload: CachePayload) -> CacheJob:
        """
        """
        _endpoint = "/api/v1/cache-jobs"

        try:

            res: Response = self._httpx_client.post(
                _endpoint,
                json=payload.model_dump_json()
            )

            res.raise_for_status()

            content = res.json()

            return CacheJob(**content)

        except HTTPStatusError as http_err: 
            self._logger.exception()
            raise PromptCacheServiceError()

        except ValidationError as valid_err:
            self._logger.exception()
            raise PromptCacheResponseError()
    
    def _wait_for_result(self, job: CacheJob) -> CacheResult:
        """
        """
        _endpoint = f"/api/v1/cache-jobs/{job.id}"
        polled_job: Optional[CacheJob] = None
        attempt = self._attempt

        try:

            while attempt > 0:

                res: Response = self._httpx_client(_endpoint)
                
                res.raise_for_status()

                content = res.json()
                polled_job_dict: dict[str, Any] = content['polled_job_dict']
                service_last_attempt: bool = content['service_last_attempt']

                polled_job = CacheJob(**polled_job_dict)

                match polled_job.status:

                    case JobStatus.PENDING:
                        self._logger.info()

                    case JobStatus.RUNNING:
                        self._logger.info()

                    case JobStatus.RETRYING:
                        self._logger.info() 

                    case JobStatus.COMPLETED:
                        self._logger.info()
                        return polled_job

                    case JobStatus.FAILED:
                        self._logger.info()
                        if service_last_attempt:
                            self._logger.info()
                            break 
                        
                    case _:
                        pass

                sleep(self._attempt_interval)        
                
                attempt -= 1

            job_error_code = polled_job.error.code    
            job_error_msg = polled_job.error.message
            
            match job_error_code:

                case CacheJobErrorCode.PROMPT_CACHE_DATABASE_ERROR:
                    raise PromptCacheDatabaseError(message=job_error_msg)

                case CacheJobErrorCode.PROMPT_CACHE_GEMINI_API_ERROR:
                    raise PromptCacheGeminiAPIError(message=job_error_msg)            

        except HTTPStatusError as http_err:
            self._logger.exception()
            raise PromptCacheServiceError()

    def _close(self) -> None:
        self.httpx_client.close()