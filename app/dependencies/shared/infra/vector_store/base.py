from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """
    Abstract base class defining the interface for vector store implementations.
    """

    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int = 3) -> list[str]:
        """
        Search for similar vectors in the store.

        Args:
            query_vector (list[float]): The dense embedding vector to search for.
            top_k (int, optional): The maximum number of results to return. Defaults to 3.

        Returns:
            list[str]: A list of plain text chunks ordered by semantic similarity.
        """
        pass

    @abstractmethod
    async def insert(self, content: str, vector: list[float]) -> None:
        """
        Insert a text chunk and its corresponding vector embedding into the store.

        Args:
            content (str): The plain text content chunk to store.
            vector (list[float]): The high-dimensional dense vector embedding.
        """
        pass