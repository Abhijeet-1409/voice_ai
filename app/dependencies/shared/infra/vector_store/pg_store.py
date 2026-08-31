from functools import cache

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from shared.infra.postgres import KnowledgeChunk
from shared.infra.postgres.database import get_async_sessionmaker

from .base import BaseVectorStore, VectorStoreError


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

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
    ) -> list[str]:
        """
        Searches the knowledge base using vector similarity.

        Args:
            query_vector (list[float]): The embedded vector of the search query.
            top_k (int, optional): The maximum number of results to return. Defaults to 3.

        Returns:
            list[str]: A list of text chunks retrieved from the knowledge base.

        Raises:
            VectorStoreError: If a database error occurs during the search.
        """
        try:
            async with self.async_session() as session:
                stmt = (
                    select(KnowledgeChunk.content)
                    .order_by(KnowledgeChunk.embedding.l2_distance(query_vector))
                    .limit(top_k)
                )
                result = await session.execute(stmt)

                rows = result.scalars().all()
                self.logger.debug(f"Vector search returned {len(rows)} results")
                return list(rows)

        except SQLAlchemyError as sql_err:
            self.logger.error(f"Vector search failed: {sql_err}")
            raise VectorStoreError(
                "Failed to search knowledge base due to a database error."
            ) from sql_err

    async def insert(
        self,
        items: list[tuple[str, list[float]]],
    ) -> None:
        """
        Inserts new text chunks and their corresponding embeddings into the knowledge base.

        Args:
            items (list[tuple[str, list[float]]]): A list of tuples, where each tuple
                contains the text content and its vector embedding.

        Raises:
            VectorStoreError: If a database error occurs during insertion.
        """
        if not items:
            self.logger.debug("insert called with an empty list — nothing to do.")
            return

        try:
            async with self.async_session() as session:
                rows = [
                    KnowledgeChunk(content=content, embedding=vector)
                    for content, vector in items
                ]
                session.add_all(rows)
                await session.commit()
                self.logger.debug(f"Inserted {len(rows)} chunk(s) into knowledge base")

        except SQLAlchemyError as sql_err:
            self.logger.error(f"Failed to insert {len(items)} chunk(s): {sql_err}")
            raise VectorStoreError(
                f"Failed to insert {len(items)} chunk(s) "
                "into knowledge base due to a database error."
            ) from sql_err


def get_pgvectorstore() -> PgVectorStore:
    """
    Creates a new instance of PgVectorStore.

    Returns:
        PgVectorStore: The vector storage manager instance.
    """
    return PgVectorStore()