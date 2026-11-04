
"""Cache expiration and validity classification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..schemas.enums import CacheValidity


def classify_cache_validity(
    expires_at: datetime,
    *,
    refresh_threshold_seconds: int,
    now: datetime | None = None,
) -> CacheValidity:
    """Classify a cached resource based on its expiration timestamp.

    Args:
        expires_at: The expiration timestamp returned by Gemini.
        refresh_threshold_seconds: How many seconds before expiration
            a cache should be considered EXPIRING.
        now: Optional reference time. Useful for deterministic
            verification and testing.

    Returns:
        CacheValidity.VALID:
            The cache has sufficient remaining lifetime.

        CacheValidity.EXPIRING:
            The cache is still usable but approaching expiration.

        CacheValidity.EXPIRED:
            The cache has already expired.

    Raises:
        ValueError: If timestamps are timezone-naive or the threshold
            is negative.
    """

    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise ValueError("expires_at must be timezone-aware.")

    if refresh_threshold_seconds < 0:
        raise ValueError(
            "refresh_threshold_seconds cannot be negative."
        )

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")

    threshold = timedelta(seconds=refresh_threshold_seconds)

    if expires_at <= now:
        return CacheValidity.EXPIRED

    if expires_at <= now + threshold:
        return CacheValidity.EXPIRING

    return CacheValidity.VALID
