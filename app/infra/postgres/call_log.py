import uuid
from typing import Optional
from datetime import datetime

from sqlalchemy import String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from config.logging import get_logger

from infra.postgres.base import Base
from infra.postgres.db import async_session

logger = get_logger("infra.postgres.call_log")


class CallLog(Base):
    __tablename__ = "calls"

    id:             Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid:     Mapped[str] = mapped_column(String, unique=True, nullable=False)
    caller_phone:   Mapped[str] = mapped_column(String, nullable=False)
    call_sid:       Mapped[Optional[str]] = mapped_column(String, nullable=True)
    started_at:     Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at:       Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
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
    Save a completed call record to the database.

    :param stream_sid: Unique identifier for the call stream.
    :param caller_phone: Caller's phone number in E.164 format.
    :param call_sid: Exotel call SID.
    :param started_at: When the call started.
    :param ended_at: When the call ended.
    :param end_reason: Reason the call ended (callended/stopped/timeout).
    :param duration_secs: Total call duration in seconds.
    """
    try:
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
        raise