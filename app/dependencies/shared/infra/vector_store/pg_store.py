from functools import cache

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from .base import BaseVectorStore, VectorStoreError
from shared.infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.vector_store.pg_store"


class PgVectorStore(BaseVectorStore):
    """
    Vector store implementation using PostgreSQL with the pgvector
    extension. Backs domain/tools/knowledge_base.py's search_knowledge_base
    tool. Implements BaseVectorStore, so this is a drop-in replacement
    for a different vector store backend later — same interface,
    different implementation.
    """

    def __init__(self):
        self.logger = get_logger(_LOGGER)
        self.async_session = get_async_sessionmaker()

    async def search(self, query_vector: list[float], top_k: int = 3) -> list[str]:

        try:
            async with self.async_session() as session:

                result = await session.execute(
                    text("""
                        SELECT content
                        FROM knowledge_base
                        ORDER BY embedding <-> :vector
                        LIMIT :top_k
                    """),
                    {"vector": query_vector, "top_k": top_k}
                )
                rows = result.fetchall()
                self.logger.debug(f"Vector search returned {len(rows)} results")
                return [row[0] for row in rows]

        except SQLAlchemyError as e:
            self.logger.error(f"Vector search failed: {e}")
            raise VectorStoreError("Failed to search knowledge base due to a database error.") from e

    async def insert(self, items: list[tuple[str, list[float]]]) -> None:
    
        if not items:
            self.logger.debug("insert called with an empty list — nothing to do.")
            return

        try:
            async with self.async_session() as session:
                await session.execute(
                    text("""
                        INSERT INTO knowledge_base (content, embedding)
                        VALUES (:content, :vector)
                    """),
                    [{"content": content, "vector": vector} for content, vector in items]
                )
                await session.commit()
                self.logger.debug(f"Inserted {len(items)} chunk(s) into knowledge base")

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to insert {len(items)} chunk(s): {e}")
            raise VectorStoreError(
                f"Failed to insert {len(items)} chunk(s) into knowledge base due to a database error."
            ) from e


@cache
def get_pgvectorstore() -> PgVectorStore:
    """
    Create and cache a thread-safe singleton instance of PgVectorStore.

    Returns:
        PgVectorStore: The cached vector storage manager instance.
    """
    return PgVectorStore()