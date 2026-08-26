from functools import cache

from shared.config import AppBaseSettings


class WorkerSettings(AppBaseSettings):
    """
    Configuration settings specific to the Agent Worker service.

    Inherits foundational settings from AppBaseSettings and adds
    configuration for the Gemini LLM, email notifications, and
    knowledge base (RAG) parameters.
    """

    # ── Agent ────────────────────────────────────────────────────────────────────
    AGENT_NAME: str

    # ── LLM ────────────────────────────────────────────────────────────────────
    GEMINI_API_KEY_1: str
    GEMINI_API_KEY_2: str
    GEMINI_API_KEY_3: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.4
    GEMINI_VOICE: str = "Aoede"
    GEMINI_TOKEN_LIMIT: int = 80
    GEMINI_ACTIVE_KEY_INDEX: int = 1

    # ── Cartesia ───────────────────────────────────────────────────────────────
    CARTESIA_API_KEY: str
    CARTESIA_VOICE_ID: str
    CARTESIA_STT_MODEL: str = "ink-whisper"
    CARTESIA_TTS_MODEL: str = "sonic-multilingual"

    # ── Livekit ──────────────────────────────────────────────────────────────────
    LIVEKIT_URL: str
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str

    # ── Email ──────────────────────────────────────────────────────────────────
    GMAIL_ADDRESS: str
    GMAIL_APP_PASSWORD: str
    NOTIFICATION_EMAIL: str

    # ── Data ───────────────────────────────────────────────────────────────────
    DATA_DIR: str 

    # ── Rag ────────────────────────────────────────────────────────────────────
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