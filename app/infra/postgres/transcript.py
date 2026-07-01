from datetime import datetime
import uuid

from sqlalchemy import String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from infra.postgres.base import Base


class Transcript(Base):
    __tablename__ = "transcripts"

    id:         Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_sid: Mapped[str] = mapped_column(String, nullable=False)
    turns:      Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)