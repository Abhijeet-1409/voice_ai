import uuid
from typing import Optional
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from shared.config import Channel, CallType
from shared.infra.postgres import Base
from shared.infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.postgres.call_log"


class CallLog(Base):
    """
    SQLAlchemy model representing a recorded call session.

    This table tracks the lifecycle metadata of every call across different channels
    (like web or phone), serving as a historical record for billing, debugging, 
    and analytics.

    Attributes:
        id (str): Primary key, uniquely identifying the call record as a UUID string.
        stream_sid (str | None): Unique identifier for the LiveKit or SIP media stream.
        caller_phone (str | None): The caller's phone number, typically in E.164 format.
        user_id (str | None): Foreign key linking the call to a specific user account.
        channel (Channel): The medium through which the call took place (e.g., WEB, PHONE).
        call_type (CallType): The directionality of the call (e.g., INBOUND, OUTREACH).
        call_sid (str | None): The external telephony provider's routing/session ID.
        started_at (datetime): UTC timestamp marking when the call officially began.
        ended_at (datetime | None): UTC timestamp marking when the call concluded.
        end_reason (str | None): The operational reason the call terminated (e.g., user hung up, timeout).
        duration_secs (int | None): Total duration of the active call in seconds.
        created_at (datetime): UTC timestamp marking when this database record was created.
        qualification_summary (str | None): AI-generated text summary detailing the lead's 
            qualification status and key conversational insights gathered post-call.
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
    created_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    qualification_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


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
    duration_secs: Optional[int] = None,
    qualification_summary: Optional[str] = None
) -> None:
    """
    Saves a completed call record to the PostgreSQL database.

    Uses an asynchronous database session to persist the call details. The async
    context manager automatically handles rollbacks on exceptions and safely closes
    the session upon completion.

    Args:
        started_at (datetime): Timestamp of when the call initiated.
        channel (Channel): The medium of the call (e.g., WEB, PHONE).
        call_type (CallType): The classification of the call (e.g., INBOUND, OUTREACH).
        stream_sid (Optional[str], optional): Unique identifier for the LiveKit stream. Defaults to None.
        caller_phone (Optional[str], optional): Caller's phone number. Defaults to None.
        call_sid (Optional[str], optional): External provider call SID. Defaults to None.
        user_id (Optional[str], optional): Foreign key linking to the user. Defaults to None.
        ended_at (Optional[datetime], optional): Timestamp of when the call ended. Defaults to None.
        end_reason (Optional[str], optional): Reason the call ended (e.g., 'agent_transfer', 'timeout'). Defaults to None.
        duration_secs (Optional[int], optional): Total call duration in seconds. Defaults to None.
        qualification_summary (Optional[str], optional): AI-generated post-call summary. Defaults to None.

    Raises:
        SQLAlchemyError: If a database constraint fails, connection drops, or the transaction cannot be committed.
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
                duration_secs=duration_secs,
                qualification_summary=qualification_summary
            )
            session.add(call)
            await session.commit()
            await session.refresh(call)  # Fixed: passed 'call' instance to refresh
            logger.info(f"Saved call log successfully | stream_sid={stream_sid} | channel={channel.value}")

    except SQLAlchemyError as sql_err:
        logger.error(f"Failed to save call log | stream_sid={stream_sid} | error={sql_err}")
        raise