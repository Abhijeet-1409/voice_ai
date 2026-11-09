"""Cache job creation and status API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_cache_job_service,
    get_db_session,
)
from ...schemas.cache import CachePayload
from ...schemas.jobs import (
    CacheJobStatusResponse,
    CacheJobSubmissionResponse,
)
from ...services.cache_job_service import CacheJobService


router = APIRouter(tags=["Cache Jobs"])


@router.post(
    "/cache-jobs",
    response_model=CacheJobSubmissionResponse,
)
async def create_cache_job(
    payload: CachePayload,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
    service: CacheJobService = Depends(get_cache_job_service),
) -> CacheJobSubmissionResponse:
    """Create a cache-creation job or return an existing active job."""

    job_result = await service.create_or_get_active_job(
        session=session,
        payload=payload,
    )

    response.status_code = (
        status.HTTP_202_ACCEPTED
        if job_result.created
        else status.HTTP_200_OK
    )

    return CacheJobSubmissionResponse(
        id=job_result.job.id,
        status=job_result.status,
        created=job_result.created,
    )


@router.get(
    "/cache-jobs/{job_id}",
    response_model=CacheJobStatusResponse,
)
async def get_cache_job_status(
    job_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    service: CacheJobService = Depends(get_cache_job_service),
) -> CacheJobStatusResponse:
    """Retrieve the current status of a cache-creation job."""

    return await service.get_job_status(
        session=session,
        job_id=job_id,
    )