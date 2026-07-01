from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    """
    
    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int = 3) -> list[str]:
        """
        Search for similar vectors in the store.

        :param query_vector: The query vector to search for.
        :param top_k: The number of results to return.
        :return: List of plain text chunks ordered by similarity.
        """
        pass

    @abstractmethod
    async def insert(self, content: str, vector: list[float]) -> None:
        """
        Insert a text chunk and its vector into the store.

        :param content: The plain text chunk.
        :param vector: The embedding vector for the text.
        :return: None
        """
        pass
