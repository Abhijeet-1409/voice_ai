import uuid
from typing import Optional
from datetime import datetime

from sqlalchemy import String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from config.logger import get_logger

from infra.postgres.base import Base
from infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.postgres.call_log"


class CallLog(Base):
    __tablename__ = "calls"

    id:             Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid:     Mapped[str] = mapped_column(String, unique=True, nullable=False)
    caller_phone:   Mapped[str] = mapped_column(String, nullable=False)
    call_sid:       Mapped[Optional[str]] = mapped_column(String, nullable=True)
    started_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at:       Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_reason:     Mapped[Optional[str]] = mapped_column(String, nullable=True)
    duration_secs:  Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


async def save_call_log(
    stream_sid:   str,
    caller_phone: str,
    call_sid:     str,
    started_at:   datetime,
    ended_at:     Optional[datetime] = None,
    end_reason:   Optional[str] = None,
    duration_secs: Optional[int] = None
) -> None:
    """
    Save a completed call record to the PostgreSQL database.

    Args:
        stream_sid (str): Unique identifier for the call stream.
        caller_phone (str): Caller's phone number in E.164 format.
        call_sid (str): Exotel call SID.
        started_at (datetime): When the call started.
        ended_at (Optional[datetime], optional): When the call ended. Defaults to None.
        end_reason (Optional[str], optional): Reason the call ended (e.g., callended, stopped, timeout). Defaults to None.
        duration_secs (Optional[int], optional): Total call duration in seconds. Defaults to None.
    """

    async_session = get_async_sessionmaker()
    logger = get_logger(_LOGGER)

    try:
        # The async context manager automatically handles rollbacks on exceptions
        # and safely closes the session when done.
        async with async_session() as session:
            call = CallLog(
                stream_sid=stream_sid,
                caller_phone=caller_phone,
                call_sid=call_sid,
                started_at=started_at,
                ended_at=ended_at,
                end_reason=end_reason,
                duration_secs=duration_secs
            )
            session.add(call)
            await session.commit()
            logger.info(f"Saved call log — stream_sid={stream_sid}")

    except SQLAlchemyError as e:
        logger.error(f"Failed to save call log — stream_sid={stream_sid} error={e}")
        # Re-raise the exception if the upstream service needs to know the database insertion failed
        raise