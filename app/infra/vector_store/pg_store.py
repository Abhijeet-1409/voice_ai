from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from infra.postgres.db import async_session
from infra.vector_store.base import BaseVectorStore

from config.logging import get_logger


logger = get_logger("infra.vector_store.pg_store")


class PgVectorStore(BaseVectorStore):
    """
    A vector store implementation using Postgres with pgvector extension.
    """

    async def search(self, query_vector: list[float], top_k: int = 3) -> list[str]:
        """
        Search for similar vectors in Postgres using pgvector.

        :param query_vector: The query vector to search for.
        :param top_k: The number of results to return.
        :return: List of plain text chunks ordered by similarity.
        """
        try:
            async with async_session() as session:
                result = await session.execute(
                    text("""
                        SELECT content
                        FROM knowledge_base
                        ORDER BY embedding <-> :vector
                        LIMIT :top_k
                    """),
                    {"vector": str(query_vector), "top_k": top_k}
                )
                rows = result.fetchall()
                logger.debug(f"Vector search returned {len(rows)} results")
                return [row[0] for row in rows]
        except SQLAlchemyError as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def insert(self, content: str, vector: list[float]) -> None:
        """
        Insert a text chunk and its vector into Postgres.

        :param content: The plain text chunk.
        :param vector: The embedding vector for the text.
        :return: None
        """
        try:
            async with async_session() as session:
                await session.execute(
                    text("""
                        INSERT INTO knowledge_base (content, embedding)
                        VALUES (:content, :vector)
                    """),
                    {"content": content, "vector": str(vector)}
                )
                await session.commit()
                logger.debug(f"Inserted chunk into knowledge base")
        except SQLAlchemyError as e:
            logger.error(f"Failed to insert chunk: {e}")
            raise


vector_store = PgVectorStore()