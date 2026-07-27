from functools import cache

from shared.config.settings import AppBaseSettings


class WorkerSettings(AppBaseSettings):
    """
    Configuration settings specific to the Agent Worker service.
    
    Inherits foundational settings from AppBaseSettings and adds 
    configuration for the Gemini LLM, email notifications, and 
    knowledge base (RAG) parameters.
    """

    # ── LLM ───────────────────────────────────────────────────────────────────
    GEMINI_API_KEY_1: str 
    GEMINI_API_KEY_2: str 
    GEMINI_API_KEY_3: str
    GEMINI_MODEL: str = "gemini-3.1-flash-live-preview"
    GEMINI_TEMPERATURE: float = 0.4
    GEMINI_VOICE: str = "Aoede"
    GEMINI_TOKEN_LIMIT=80,

    # ── Email ─────────────────────────────────────────────────────────────────
    GMAIL_ADDRESS: str
    GMAIL_APP_PASSWORD: str
    NOTIFICATION_EMAIL: str
    
    # ── Rag ───────────────────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_MODEL_PATH: str
    RAG_TOP_K: int = 10


@cache
def get_worker_settings() -> WorkerSettings:
    """
    Retrieve the worker-specific application settings. 
    
    Uses caching to ensure the settings are instantiated only once during 
    the process lifecycle, preventing redundant environment variable lookups.

    Returns:
        WorkerSettings: The cached worker configuration object.
    """
    return WorkerSettings()