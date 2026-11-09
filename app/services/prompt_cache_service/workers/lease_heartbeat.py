
"""Background heartbeat for worker lease renewal."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from shared.logging_setup import get_logger

from ..repositories.lock_repository import (
    LeaseGrant,
    LockRepository,
)


_LOGGER = "prompt_cache_service.workers.lease_heartbeat"
logger = get_logger(_LOGGER)


class WorkerLeaseHeartbeat:
    """Keep a worker's PostgreSQL lease alive during execution."""

    STOP_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        lock_repository: LockRepository,
        lease: LeaseGrant,
        lease_seconds: int,
        heartbeat_seconds: int,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive.")

        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive.")

        if heartbeat_seconds >= lease_seconds:
            raise ValueError(
                "heartbeat_seconds must be shorter than lease_seconds."
            )

        self.session_factory = session_factory
        self.lock_repository = lock_repository
        self.lease = lease
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds

        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

        self._started = False
        self._lease_lost = False

    @property
    def lease_lost(self) -> bool:
        """Whether the worker has lost confirmed lease ownership."""
        return self._lease_lost

    async def start(self) -> None:
        """Start the background heartbeat task."""

        if self._started:
            raise RuntimeError(
                "Worker lease heartbeat has already been started."
            )

        self._started = True

        logger.info(
            "Starting lease heartbeat: signature=%s, "
            "owner_token=%s, fencing_token=%s",
            self.lease.signature,
            self.lease.owner_token,
            self.lease.fencing_token,
        )

        self._task = asyncio.create_task(
            self._run(),
            name=f"lease-heartbeat-{self.lease.signature[:12]}",
        )

    async def stop(self) -> None:
        """Stop the heartbeat task and wait for clean shutdown."""

        if self._task is None:
            return

        self._stop_event.set()

        try:
            await asyncio.wait_for(
                self._task,
                timeout=self.STOP_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "Lease heartbeat shutdown exceeded timeout; "
                "cancelling heartbeat task. signature=%s",
                self.lease.signature,
            )

            self._task.cancel()

            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._task,
                    timeout=self.STOP_TIMEOUT_SECONDS,
                )

    async def _run(self) -> None:
        """Periodically renew the lease using independent DB sessions."""

        while not self._stop_event.is_set():

            # Wait between renewals, but allow stop() to wake this task
            # immediately instead of waiting for the full interval.
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.heartbeat_seconds,
                )
                return

            except asyncio.TimeoutError:
                pass

            # Every renewal gets its own AsyncSession and transaction.
            # Never share the worker's session with this coroutine.
            try:
                async with self.session_factory() as session:
                    async with session.begin():
                        renewed = await self.lock_repository.renew(
                            session,
                            signature=self.lease.signature,
                            owner_token=self.lease.owner_token,
                            fencing_token=self.lease.fencing_token,
                            lease_seconds=self.lease_seconds,
                        )

                if not renewed:
                    self._lease_lost = True

                    logger.warning(
                        "Worker lease renewal rejected; "
                        "lease ownership has been lost. "
                        "signature=%s, fencing_token=%s",
                        self.lease.signature,
                        self.lease.fencing_token,
                    )
                    return

                logger.debug(
                    "Worker lease renewed successfully: "
                    "signature=%s, fencing_token=%s",
                    self.lease.signature,
                    self.lease.fencing_token,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                # Fail closed: if PostgreSQL cannot confirm renewal,
                # the worker must not assume it still owns the lease.
                self._lease_lost = True

                logger.exception(
                    "Lease heartbeat failed; treating lease ownership "
                    "as lost. signature=%s, fencing_token=%s",
                    self.lease.signature,
                    self.lease.fencing_token,
                )
                return
