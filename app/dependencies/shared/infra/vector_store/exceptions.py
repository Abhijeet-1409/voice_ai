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