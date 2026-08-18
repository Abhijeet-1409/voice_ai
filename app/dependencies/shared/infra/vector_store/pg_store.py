from functools import cache

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from shared.infra.vector_store import BaseVectorStore, VectorStoreError
from shared.infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.vector_store.pg_store"


class PgVectorStore(BaseVectorStore):
    """
    Vector store implementation using PostgreSQL with the pgvector extension.
    """

    def __init__(self):
        self.logger = get_logger(_LOGGER)
        self.async_session = get_async_sessionmaker()

    async def search(self, query_vector: list[float], top_k: int = 3) -> list[str]:
        """
        Search for semantically similar text chunks in PostgreSQL using pgvector.

        Args:
            query_vector (list[float]): The dense query embedding vector.
            top_k (int, optional): The number of near neighbors to return. Defaults to 3.

        Returns:
            list[str]: Plain text chunks ordered by cosine/L2 distance matching.
        """

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

    async def insert(self, content: str, vector: list[float]) -> None:
        """
        Insert a raw text chunk alongside its high-dimensional vector.

        Args:
            content (str): The plain text documentation/context chunk.
            vector (list[float]): Dense float list representing the vector embedding.

        Raises:
            SQLAlchemyError: If the transaction or commit step fails.
        """

        try:
            async with self.async_session() as session:

                await session.execute(
                    text("""
                        INSERT INTO knowledge_base (content, embedding)
                        VALUES (:content, :vector)
                    """),
                    {"content": content, "vector": vector}
                )
                await session.commit()
                self.logger.debug("Inserted chunk into knowledge base")

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to insert chunk: {e}")
            raise VectorStoreError("Failed to insert chunk into knowledge base due to a database error.") from e


@cache
def get_pgvectorstore() -> PgVectorStore:
    """
    Create and cache a thread-safe singleton instance of PgVectorStore.

    Returns:
        PgVectorStore: The cached vector storage manager instance.
    """
    return PgVectorStore()