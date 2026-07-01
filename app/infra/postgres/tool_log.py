import uuid
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import String, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from config.logging import get_logger

from infra.postgres.base import Base
from infra.postgres.db import async_session

logger = get_logger("infra.postgres.tool_log")


class ToolLog(Base):
    __tablename__ = "tool_logs"

    id:         Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid: Mapped[str] = mapped_column(String, nullable=False)
    tool_name:  Mapped[str] = mapped_column(String, nullable=False)
    arguments:  Mapped[dict] = mapped_column(JSONB, nullable=False)
    result:     Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    called_at:  Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


async def save_tool_log(
    stream_sid: str,
    tool_name:  str,
    arguments:  dict,
    result:     Optional[str] = None,   
    latency_ms: Optional[int] = None,    
) -> None:
    """
    Save a tool invocation record to the database.
    Called every time a Gemini tool call completes.

    :param stream_sid: Unique identifier for the call stream.
    :param tool_name: Name of the tool that was called.
    :param arguments: Arguments passed to the tool.
    :param result: Result returned by the tool.
    :param latency_ms: Time taken for the tool call in milliseconds.
    """
    try:
        async with async_session() as session:
            log = ToolLog(
                stream_sid=stream_sid,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                latency_ms=latency_ms
            )
            session.add(log)
            await session.commit()
            logger.info(f"Saved tool log — stream_sid={stream_sid} tool={tool_name} latency={latency_ms}ms")
    except SQLAlchemyError as e:
        logger.error(f"Failed to save tool log — stream_sid={stream_sid} tool={tool_name} error={e}")
        raise