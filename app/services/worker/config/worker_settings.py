from functools import cache

from pydantic import Field

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
    GEMINI_THINKING_BUDGET: int = 0
    PROJECT: str | None
    LOCATION: str | None
    VERTEXAI: bool | None = False

    # ── Cartesia ───────────────────────────────────────────────────────────────
    CARTESIA_API_KEY: str
    CARTESIA_VOICE_ID: str
    CARTESIA_STT_MODEL: str = "ink-whisper"
    CARTESIA_TTS_MODEL: str = "sonic-multilingual"
    CARTESIA_TTS_LANGUAGE: str = "en"

    # ── Deepgram ───────────────────────────────────────────────────────────────
    DEEPGRAM_API_KEY: str
    DEEPGRAM_STT_MODEL: str = "nova-3"
    DEEPGRAM_STT_LANGUAGE: str = "multi"
    DEEPGRAM_STT_ENDPOINTING_MS: int = 25
    DEEPGRAM_STT_KEYTERMS: list[str] = Field(default_factory=list)

    # ── Selero VAD ───────────────────────────────────────────────────────────────
    VAD_MIN_SPEECH_DURATION: float = 0.05
    VAD_MIN_SILENCE_DURATION: float = 0.25
    VAD_ACTIVATION_THRESHOLD: float = 0.5

     # ── Turn handling ───────────────────────────────────────────────────────────
    TURNHANDLING_ENDPOINTING_MODE: str = "fixed"
    TURNHANDLING_ENDPOINTING_MIN_DELAY: float = 0.25
    TURNHANDLING_ENDPOINTING_MAX_DELAY: float = 0.8
    TURNHANDLING_PREEMPTIVE_GENERATION_ENABLED: bool = True
    TURNHANDLING_PREEMPTIVE_GENERATION_PREEMPTIVE_TTS: bool =False   

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