import uuid

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from pgvector.sqlalchemy import Vector

from .base import Base


class KnowledgeChunk(Base):
    """
    SQLAlchemy model representing a single text chunk and its vector
    embedding, used for semantic similarity search and Retrieval-
    Augmented Generation (RAG).
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)