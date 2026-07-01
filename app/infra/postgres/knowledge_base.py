import uuid

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from infra.postgres.base import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id:        Mapped[str]        = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content:   Mapped[str]        = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)