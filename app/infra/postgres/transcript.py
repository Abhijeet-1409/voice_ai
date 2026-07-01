import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from config.logging import get_logger

from infra.postgres.base import Base
from infra.postgres.db import async_session

logger = get_logger("infra.postgres.transcript")


class Transcript(Base):
    __tablename__ = "transcripts"

    id:         Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid: Mapped[str] = mapped_column(String, nullable=False)
    turns:      Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


async def save_transcript(
    stream_sid: str,
    turns: list[dict]
) -> None:
    """
    Save the full conversation transcript to the database.
    Called once at call end after flushing from Redis.

    :param stream_sid: Unique identifier for the call stream.
    :param turns: List of turn dicts from Redis transcript store.
    """
    try:
        async with async_session() as session:
            transcript = Transcript(
                stream_sid=stream_sid,
                turns=turns
            )
            session.add(transcript)
            await session.commit()
            logger.info(f"Saved transcript — stream_sid={stream_sid} turns={len(turns)}")
    except SQLAlchemyError as e:
        logger.error(f"Failed to save transcript — stream_sid={stream_sid} error={e}")
        raise