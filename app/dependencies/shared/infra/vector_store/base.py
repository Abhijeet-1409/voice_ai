from abc import ABC, abstractmethod


class VectorStoreError(Exception):
    """
    Base exception class for vector store errors — raised for any
    underlying storage failure (e.g. a database error during search or
    insert). Mirrors the same exception pattern used by BaseCRMClient
    and BaseCalendarClient: implementations should catch their own
    backend-specific errors and re-raise as VectorStoreError, so
    callers (search_knowledge_base and any future ingest tooling) only
    ever need to handle one exception type regardless of which vector
    store backend is active.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BaseVectorStore(ABC):
    """
    Abstract interface for vector store integrations (Postgres/pgvector
    today, potentially a dedicated vector DB later — same interface,
    swappable backend).

    Backs the knowledge base search used by domain/tools/knowledge_base.py's
    search_knowledge_base tool, as well as any future ingest script that
    populates the store with documentation/content chunks.
    """

    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int = 3) -> list[str]:
        """
        Search for the most semantically similar text chunks to a given
        query embedding.

        Args:
            query_vector: The dense embedding vector to search for,
                produced by the same embedding model used at insert time
                (mismatched embedding models would produce meaningless
                distances).
            top_k: The maximum number of results to return, ordered by
                similarity (closest first). Defaults to 3.

        Returns:
            A list of plain text chunks ordered by semantic similarity,
            most similar first. Empty list if nothing is found.

        Raises:
            VectorStoreError: If the underlying search operation fails.
        """
        ...

    @abstractmethod
    async def insert(self, items: list[tuple[str, list[float]]]) -> None:
        """
        Insert one or more text/vector pairs into the store.

        Always takes a list — for a single chunk, pass a one-item list:
        insert([(content, vector)]). Implementations should insert
        multiple items as a single batched operation rather than looping
        per item, so this is also the right method for bulk population
        (e.g. an ingest script loading many documentation chunks at once).

        Args:
            items: List of (content, vector) tuples to insert, where each
                vector is the embedding produced for the paired content
                by the same embedding model used at search time.

        Raises:
            VectorStoreError: If the underlying insert operation fails.
        """
        ...