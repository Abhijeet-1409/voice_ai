import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from shared.infra.postgres import Base
from shared.infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.postgres.transcript"


class Transcript(Base):
    """
    SQLAlchemy model representing a historical record of a call transcript.

    Stores the accumulated conversational turns flushed out of volatile memory
    (Redis) into a structured JSONB array field for permanent analytics.
    """

    __tablename__ = "transcripts"

    id:         Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid: Mapped[str] = mapped_column(String, nullable=False)
    turns:      Mapped[list[dict]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


async def save_transcript(
    stream_sid: str,
    turns: list[dict]
) -> None:
    """
    Save the full conversation transcript to the PostgreSQL database.

    Typically called exactly once at call teardown after flushing active turns
    out of the temporary Redis storage mechanism.

    Args:
        stream_sid (str): Unique identifier for the call stream.
        turns (list[dict]): List of turn dictionaries fetched from the Redis transcript store.

    Raises:
        SQLAlchemyError: If the database transaction or commit sequence fails.
    """

    async_session = get_async_sessionmaker()
    logger = get_logger(_LOGGER)

    try:

        async with async_session() as session:
            transcript = Transcript(
                stream_sid=stream_sid,
                turns=turns
            )
            session.add(transcript)
            await session.commit()
            await session.refresh(transcript)
            logger.info(f"Saved transcript — stream_sid={stream_sid} turns={len(turns)}")

    except SQLAlchemyError as e:
        logger.error(f"Failed to save transcript — stream_sid={stream_sid} error={e}")
        raise