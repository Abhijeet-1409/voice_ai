from datetime import datetime
import uuid

from sqlalchemy import String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from infra.postgres.base import Base


class CallLog(Base):
    __tablename__ = "calls"

    id:             Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid:     Mapped[str] = mapped_column(String, unique=True, nullable=False)
    caller_phone:   Mapped[str] = mapped_column(String, nullable=False)
    call_sid:       Mapped[str] = mapped_column(String, nullable=True)
    started_at:     Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at:       Mapped[datetime] = mapped_column(DateTime, nullable=True)
    end_reason:     Mapped[str] = mapped_column(String, nullable=True)
    duration_secs:  Mapped[int] = mapped_column(Integer, nullable=True)