"""Cache lookup API endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_cache_lookup_service,
    get_db_session,
)
from ...schemas.cache import CacheLookupResponse, CachePayload
from ...services.cache_lookup_service import CacheLookupService


router = APIRouter(tags=["Prompt Cache"])


@router.post(
    "/caches/lookup",
    response_model=CacheLookupResponse,
)
async def lookup_cache(
    payload: CachePayload,
    session: AsyncSession = Depends(get_db_session),
    service: CacheLookupService = Depends(get_cache_lookup_service),
) -> CacheLookupResponse:
    """Find an existing reusable Gemini cached resource."""

    return await service.lookup(
        session=session,
        payload=payload,
    )