import uuid

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from pgvector.sqlalchemy import Vector

from shared.infra.postgres import Base


class KnowledgeBase(Base):
    """
    SQLAlchemy model representing a knowledge base document with vector embeddings.

    This table stores text content alongside its dense vector representation,
    enabling semantic similarity search and Retrieval-Augmented Generation (RAG).
    """

    __tablename__ = "knowledge_base"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)