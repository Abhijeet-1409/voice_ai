import uuid
from typing import Optional
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from shared.infra.postgres import Base
from shared.infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.postgres.tool_log"
logger = get_logger(_LOGGER)
async_session = get_async_sessionmaker()


class ToolLog(Base):
    """
    SQLAlchemy model representing an invocation log for an AI tool call.

    This table tracks exactly which tools Gemini called, the inputs provided,
    the outcomes, and the latency, serving as an audit trail and analytics source.
    """

    __tablename__ = "tool_logs"

    id:            Mapped[str]           = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid:    Mapped[str]           = mapped_column(String, nullable=False)
    tool_name:     Mapped[str]           = mapped_column(String, nullable=False)
    arguments:     Mapped[dict]          = mapped_column(JSONB, nullable=False)
    result:        Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_error:      Mapped[bool]          = mapped_column(Boolean, nullable=False, default=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms:    Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    called_at:     Mapped[datetime]      = mapped_column(DateTime(timezone=True), nullable=False)


async def save_tool_log(
    stream_sid:    str,
    tool_name:     str,
    arguments:     dict,
    called_at:     datetime,
    result:        Optional[str] = None,
    is_error:      bool = False,
    error_message: Optional[str] = None,
    latency_ms:    Optional[int] = None,
) -> None:
    """
    Save a tool invocation record to the PostgreSQL database.

    Args:
        stream_sid (str): Unique identifier for the call stream.
        tool_name (str): Name of the specific tool that was executed.
        arguments (dict): Arguments provided to the tool as a dictionary.
        called_at (datetime): When the tool was invoked.
        result (Optional[str]): The raw output returned by the tool. Defaults to None.
        is_error (bool): Whether the tool call resulted in an error. Defaults to False.
        error_message (Optional[str]): Error message if is_error is True. Defaults to None.
        latency_ms (Optional[int]): Execution time in milliseconds. Defaults to None.

    Raises:
        SQLAlchemyError: If the database transaction fails.
    """
    try:
        async with async_session() as session:
            log = ToolLog(
                stream_sid=stream_sid,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                is_error=is_error,
                error_message=error_message,
                latency_ms=latency_ms,
                called_at=called_at
            )
            session.add(log)
            await session.commit()
            await session.refresh(log)
            logger.info(
                f"Saved tool log — stream_sid={stream_sid} "
                f"tool={tool_name} is_error={is_error} latency={latency_ms}ms"
            )
    except SQLAlchemyError as e:
        logger.error(
            f"Failed to save tool log — stream_sid={stream_sid} "
            f"tool={tool_name} error={e}"
        )
        raise