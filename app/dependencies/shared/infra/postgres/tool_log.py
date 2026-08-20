import uuid
import json
from json import JSONDecodeError
from typing import Optional
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from shared.infra.postgres import Base
from shared.infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.postgres.tool_log"


class ToolLog(Base):
    """
    SQLAlchemy model representing an invocation log for an AI tool call.

    This table tracks exactly which tools Gemini called, the inputs provided,
    the outcomes, and the latency, serving as an audit trail and analytics source.
    """

    __tablename__ = "tool_logs"

    id:         Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid: Mapped[str] = mapped_column(String, nullable=False)
    tool_name:  Mapped[str] = mapped_column(String, nullable=False)
    arguments:  Mapped[dict] = mapped_column(JSONB, nullable=False)
    result:     Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    called_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


async def save_tool_log(
    stream_sid: str,
    tool_name:  str,
    arguments:  str,
    result:     Optional[str] = None,
    latency_ms: Optional[int] = None,
) -> None:
    """
    Save a tool invocation record to the PostgreSQL database.

    Parses the stringified JSON arguments from the LLM before storing
    the data in a structured binary JSONB format.

    Args:
        stream_sid (str): Unique identifier for the call stream.
        tool_name (str): Name of the specific tool that was executed.
        arguments (str): Stringified JSON representing arguments provided to the tool.
        result (Optional[str], optional): The raw output or response returned by the tool. Defaults to None.
        latency_ms (Optional[int], optional): Execution time of the tool in milliseconds. Defaults to None.

    Raises:
        JSONDecodeError: If the argument string is not valid JSON.
        SQLAlchemyError: If the database transaction fails.
    """

    async_session = get_async_sessionmaker()
    logger = get_logger(_LOGGER)

    try:
        # Deserialize the stringified arguments explicitly into a dictionary
        arguments_dict = json.loads(arguments)

        async with async_session() as session:
            log = ToolLog(
                stream_sid=stream_sid,
                tool_name=tool_name,
                arguments=arguments_dict,
                result=result,
                latency_ms=latency_ms
            )
            session.add(log)
            await session.commit()
            await session.refresh()
            logger.info(f"Saved tool log — stream_sid={stream_sid} tool={tool_name} latency={latency_ms}ms")

    except JSONDecodeError as e:
        logger.error(f"Malformed tool argument JSON string — stream_sid={stream_sid} tool={tool_name} error={e}")
        raise
    except SQLAlchemyError as e:
        logger.error(f"Failed to save tool log — stream_sid={stream_sid} tool={tool_name} error={e}")
        raise