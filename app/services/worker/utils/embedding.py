from functools import cache
from sentence_transformers import SentenceTransformer

from config.worker_settings import get_worker_settings
from shared.logging_setup.logger import get_logger


_LOGGER = "worker.domain.embedding"
logger = get_logger(_LOGGER)


@cache
def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer embedding model.
    Shared by both the Worker (for search) and the Ingest Script (for population).
    """
    settings = get_worker_settings()

    try:
        logger.debug(f"Loading embedding model '{settings.EMBEDDING_MODEL_NAME}' from path: {settings.EMBEDDING_MODEL_PATH}")
        model = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
        logger.debug(f"Embedding model '{settings.EMBEDDING_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}' from path='{settings.EMBEDDING_MODEL_PATH}' — error={e}")
        raise

    return model