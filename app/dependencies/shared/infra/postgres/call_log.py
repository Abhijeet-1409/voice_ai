import uuid
from typing import Optional
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from config.logger import get_logger
from config.constants import Channel, CallType

from infra.postgres.base import Base
from infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.postgres.call_log"


class CallLog(Base):
    """
    SQLAlchemy model representing a recorded call session.

    This table tracks the lifecycle metadata of every call across different channels 
    (like web or exotel), including user associations, caller information, timestamps, 
    duration, and termination reasons, serving as a historical record for billing, 
    debugging, and analytics.
    """

    __tablename__ = "calls"

    id:             Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid:     Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    caller_phone:   Mapped[Optional[str]] = mapped_column(String, nullable=True)
    user_id:        Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    
    channel:        Mapped[Channel]  = mapped_column(String, nullable=False)
    call_type:      Mapped[CallType] = mapped_column(String, nullable=False)
    
    call_sid:       Mapped[Optional[str]] = mapped_column(String, nullable=True)
    started_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at:       Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_reason:     Mapped[Optional[str]] = mapped_column(String, nullable=True)
    duration_secs:  Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


async def save_call_log(
    started_at:    datetime,
    channel:       Channel, 
    call_type:     CallType,
    stream_sid:    Optional[str] = None,
    caller_phone:  Optional[str] = None,
    call_sid:      Optional[str] = None,
    user_id:       Optional[str] = None,
    ended_at:      Optional[datetime] = None,
    end_reason:    Optional[str] = None,
    duration_secs: Optional[int] = None
) -> None:
    """
    Saves a completed call record to the PostgreSQL database.

    Uses an asynchronous database session to persist the call details. The async 
    context manager automatically handles rollbacks on exceptions and safely closes 
    the session upon completion.

    Args:
        started_at (datetime): Timestamp of when the call initiated.
        channel (Channel): The medium of the call (e.g., telephony, web).
        call_type (CallType): The classification of the call (e.g., inbound, outbound).
        stream_sid (Optional[str], optional): Unique identifier for the call stream. Defaults to None.
        caller_phone (Optional[str], optional): Caller's phone number in E.164 format. Defaults to None.
        call_sid (Optional[str], optional): External provider call SID (e.g., Exotel). Defaults to None.
        user_id (Optional[str], optional): Foreign key linking to the user associated with the call. Defaults to None.
        ended_at (Optional[datetime], optional): Timestamp of when the call ended. Defaults to None.
        end_reason (Optional[str], optional): Reason the call ended (e.g., 'callended', 'timeout'). Defaults to None.
        duration_secs (Optional[int], optional): Total call duration in seconds. Defaults to None.

    Raises:
        SQLAlchemyError: If a database constraint fails or the transaction cannot be committed.
    """

    async_session = get_async_sessionmaker()
    logger = get_logger(_LOGGER)

    try:
        async with async_session() as session:
            call = CallLog(
                stream_sid=stream_sid,
                caller_phone=caller_phone,
                call_sid=call_sid,
                user_id=user_id,
                channel=channel,
                call_type=call_type,
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