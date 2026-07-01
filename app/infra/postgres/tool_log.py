from datetime import datetime
import uuid

from sqlalchemy import String, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from infra.postgres.base import Base


class ToolLog(Base):
    __tablename__ = "tool_logs"

    id:         Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid: Mapped[str] = mapped_column(String, nullable=False)
    tool_name:  Mapped[str] = mapped_column(String, nullable=False)
    arguments:  Mapped[dict] = mapped_column(JSONB, nullable=False)
    result:     Mapped[str] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=True)
    called_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)