
"""Persistence operations for distributed Prompt Cache worker leases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infra.postgres import PromptCacheLock
from shared.logging_setup import get_logger


_LOGGER = "prompt_cache_service.repositories.lock_repositories"
logger = get_logger(_LOGGER)


@dataclass(frozen=True)
class LeaseGrant:
    """Represent ownership granted to a Prompt Cache worker.

    Attributes:
        signature: Cache signature associated with the lease.
        owner_token: Unique identifier of the worker owning the lease.
        fencing_token: Monotonically increasing ownership version.
        lease_expires_at: Lease expiration timestamp returned by PostgreSQL.
    """

    signature: str
    owner_token: UUID
    fencing_token: int
    lease_expires_at: datetime


class LockRepository:
    """Manage PostgreSQL-backed distributed worker leases.

    Lease ownership is coordinated through the prompt_cache_locks table.

    Released rows are deliberately retained so that fencing tokens
    remain monotonically increasing across worker acquisitions.

    Transaction ownership belongs to the calling service.
    This repository never commits or rolls back a transaction.
    """

    @staticmethod
    def _lease_expiration(lease_seconds: int):
        """Calculate lease expiration using PostgreSQL database time."""

        return (
            func.clock_timestamp()
            + func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds)
        )

    @staticmethod
    def _validate_lease_seconds(lease_seconds: int) -> None:
        """Validate the requested lease duration."""

        if lease_seconds <= 0:
            raise ValueError(
                "lease_seconds must be greater than zero."
            )

    async def try_acquire(
        self,
        session: AsyncSession,
        *,
        signature: str,
        owner_token: UUID,
        lease_seconds: int,
    ) -> LeaseGrant | None:
        """Atomically acquire a lease for a cache signature.

        A lease can be acquired when:

        1. No lock row exists for the signature.
        2. The existing lease has been released.
        3. The existing lease has expired.

        Every successful acquisition increments the fencing token.

        Returns:
            LeaseGrant when ownership is acquired, otherwise None.
        """

        self._validate_lease_seconds(lease_seconds)

        insert_stmt = pg_insert(PromptCacheLock).values(
            signature=signature,
            owner_token=owner_token,
            fencing_token=1,
            lease_expires_at=self._lease_expiration(lease_seconds),
            heartbeat_at=func.clock_timestamp(),
        )

        stmt = (
            insert_stmt.on_conflict_do_update(
                index_elements=[PromptCacheLock.signature],
                set_={
                    "owner_token": owner_token,
                    "fencing_token": PromptCacheLock.fencing_token + 1,
                    "lease_expires_at": self._lease_expiration(
                        lease_seconds
                    ),
                    "heartbeat_at": func.clock_timestamp(),
                    "updated_at": func.clock_timestamp(),
                },
                where=or_(
                    PromptCacheLock.owner_token.is_(None),
                    PromptCacheLock.lease_expires_at.is_(None),
                    PromptCacheLock.lease_expires_at
                    <= func.clock_timestamp(),
                ),
            )
            .returning(
                PromptCacheLock.fencing_token,
                PromptCacheLock.lease_expires_at,
            )
        )

        result = await session.execute(stmt)
        row = result.one_or_none()

        if row is None:
            logger.debug(
                "Lease acquisition denied: signature=%s owner_token=%s",
                signature,
                owner_token,
            )
            return None

        fencing_token, lease_expires_at = row

        lease = LeaseGrant(
            signature=signature,
            owner_token=owner_token,
            fencing_token=fencing_token,
            lease_expires_at=lease_expires_at,
        )

        logger.info(
            "Worker lease acquired: signature=%s owner_token=%s "
            "fencing_token=%s expires_at=%s",
            signature,
            owner_token,
            fencing_token,
            lease_expires_at,
        )

        return lease

    async def renew(
        self,
        session: AsyncSession,
        *,
        signature: str,
        owner_token: UUID,
        fencing_token: int,
        lease_seconds: int,
    ) -> bool:
        """Renew an existing lease owned by the specified worker.

        Renewal succeeds only when:

        1. The signature matches.
        2. The owner token matches.
        3. The fencing token matches.
        4. The existing lease has not expired.

        Returns:
            True if the lease was renewed, otherwise False.
        """

        self._validate_lease_seconds(lease_seconds)

        stmt = (
            update(PromptCacheLock)
            .where(
                PromptCacheLock.signature == signature,
                PromptCacheLock.owner_token == owner_token,
                PromptCacheLock.fencing_token == fencing_token,
                PromptCacheLock.lease_expires_at
                > func.clock_timestamp(),
            )
            .values(
                lease_expires_at=self._lease_expiration(lease_seconds),
                heartbeat_at=func.clock_timestamp(),
                updated_at=func.clock_timestamp(),
            )
            .returning(PromptCacheLock.signature)
        )

        result = await session.execute(stmt)
        renewed_signature = result.scalar_one_or_none()

        if renewed_signature is None:
            logger.warning(
                "Worker lease renewal failed: signature=%s "
                "owner_token=%s fencing_token=%s",
                signature,
                owner_token,
                fencing_token,
            )
            return False

        logger.debug(
            "Worker lease renewed: signature=%s fencing_token=%s",
            signature,
            fencing_token,
        )

        return True

    async def release(
        self,
        session: AsyncSession,
        *,
        signature: str,
        owner_token: UUID,
        fencing_token: int,
    ) -> bool:
        """Release a lease if the supplied ownership details match.

        The lock row is never deleted. Its fencing token is preserved
        so that future acquisitions cannot reuse an older token.

        Returns:
            True if the lease was released, otherwise False.
        """

        stmt = (
            update(PromptCacheLock)
            .where(
                PromptCacheLock.signature == signature,
                PromptCacheLock.owner_token == owner_token,
                PromptCacheLock.fencing_token == fencing_token,
            )
            .values(
                owner_token=None,
                lease_expires_at=func.clock_timestamp(),
                updated_at=func.clock_timestamp(),
            )
            .returning(PromptCacheLock.signature)
        )

        result = await session.execute(stmt)
        released_signature = result.scalar_one_or_none()

        if released_signature is None:
            logger.warning(
                "Worker lease release skipped: ownership did not match. "
                "signature=%s owner_token=%s fencing_token=%s",
                signature,
                owner_token,
                fencing_token,
            )
            return False

        logger.info(
            "Worker lease released: signature=%s fencing_token=%s",
            signature,
            fencing_token,
        )

        return True

    async def invalidate(
        self,
        session: AsyncSession,
        *,
        signature: str,
    ) -> None:
        """Forcefully invalidate the current lease.

        This operation:

        1. Clears the current owner.
        2. Expires the lease immediately.
        3. Increments the fencing token.

        Incrementing the fencing token invalidates any worker holding
        the previous ownership version.
        """

        stmt = (
            update(PromptCacheLock)
            .where(
                PromptCacheLock.signature == signature
            )
            .values(
                owner_token=None,
                fencing_token=PromptCacheLock.fencing_token + 1,
                lease_expires_at=func.clock_timestamp(),
                updated_at=func.clock_timestamp(),
            )
            .returning(PromptCacheLock.fencing_token)
        )

        result = await session.execute(stmt)
        fencing_token = result.scalar_one_or_none()

        if fencing_token is None:
            logger.debug(
                "No lock row found during lease invalidation: signature=%s",
                signature,
            )
            return

        logger.warning(
            "Worker lease invalidated: signature=%s new_fencing_token=%s",
            signature,
            fencing_token,
        )

    async def is_owner(
        self,
        session: AsyncSession,
        *,
        signature: str,
        owner_token: UUID,
        fencing_token: int,
    ) -> bool:
        """Check whether the supplied worker still owns an active lease.

        The lease must match both ownership identifiers and must not
        have expired according to PostgreSQL database time.

        Returns:
            True when ownership is valid, otherwise False.
        """

        stmt = (
            select(PromptCacheLock.signature)
            .where(
                PromptCacheLock.signature == signature,
                PromptCacheLock.owner_token == owner_token,
                PromptCacheLock.fencing_token == fencing_token,
                PromptCacheLock.lease_expires_at
                > func.clock_timestamp(),
            )
        )

        result = await session.execute(stmt)
        existing_signature = result.scalar_one_or_none()

        return existing_signature is not None
