from functools import cache

from livekit.plugins import google
from livekit.agents import AgentSession

from shared.config.logger import get_logger

from config.worker_settings import get_worker_settings

_LOGGER = "worker.agent.session"
logger = get_logger(_LOGGER)


@cache
def create_agent_session(api_key: str) -> AgentSession:
    """
    Creates and caches a reusable AgentSession maker for the session pool using 
    Google's Gemini Realtime model.
    
    The @cache decorator ensures that we instantiate a unique session maker 
    for each distinct API key, allowing the worker to efficiently reuse them 
    for concurrent calls without constant re-initialization.

    Args:
        api_key (str): The Google API key to authenticate the Gemini model.

    Returns:
        AgentSession: The configured LiveKit AgentSession object.
    """
    settings = get_worker_settings()

    logger.debug(
        f"Initializing Gemini Realtime session maker — "
        f"model='{settings.GEMINI_MODEL}', voice='{settings.GEMINI_VOICE}'"
    )

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model=settings.GEMINI_MODEL,
            temperature=settings.GEMINI_TEMPERATURE,
            voice=settings.GEMINI_VOICE,
            max_output_tokens=settings.GEMINI_TOKEN_LIMIT,
            api_key=api_key
        )
    )

    logger.debug("Successfully initialized and cached the AgentSession maker.")
    
    return session
