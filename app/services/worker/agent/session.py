from livekit.plugins import google
from livekit.agents import AgentSession
from livekit.plugins import silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from shared.logging_setup.logger import get_logger

from app.services.worker.schemas.session_data import UserData
from config.worker_settings import get_worker_settings


_LOGGER = "worker.agent.session"
logger = get_logger(_LOGGER)


def create_agent_session(api_key: str, userdata: UserData) -> AgentSession:
    """
    Creates and configures a new LiveKit AgentSession for an incoming voice call.

    This function assembles a complete multilingual voice pipeline, combining local
    Voice Activity Detection (Silero), multilingual turn detection, Cartesia STT/TTS,
    and a Google Gemini LLM. The provided user metadata is serialized and attached
    to the session for downstream tracking and context routing.

    Args:
        api_key (str): The Google API key to authenticate the Gemini LLM.
        user_data (UserData): The dataclass containing call metadata (e.g.,
            stream_sid, customer_id, channel) to be bound to this session.

    Returns:
        AgentSession: The fully initialized voice agent session ready to be
            started within a LiveKit room context.
    """
    settings = get_worker_settings()

    logger.info(
        f"Initializing AgentSession for stream [{user_data.stream_sid}] — "
        f"LLM: '{settings.GEMINI_MODEL}', Cartesia TTS: '{settings.CARTESIA_TTS_MODEL}'"
    )

    session = AgentSession(
        userdata=userdata,
        vad=silero.VAD.load(),
        stt=cartesia.STT(
            api_key=settings.CARTESIA_API_KEY,
            model=settings.CARTESIA_STT_MODEL,
        ),
        llm=google.LLM(
            model=settings.GEMINI_MODEL,
            temperature=settings.GEMINI_TEMPERATURE,
            voice=settings.GEMINI_VOICE,
            max_output_tokens=settings.GEMINI_TOKEN_LIMIT,
            api_key=api_key
        ),
        tts=cartesia.TTS(
            api_key=settings.CARTESIA_API_KEY,
            voice_id=settings.CARTESIA_VOICE_ID,
            model=settings.CARTESIA_TTS_MODEL
        ),
        turn_detection=MultilingualModel()
    )

    logger.debug(f"AgentSession successfully constructed and bound to stream [{userdata.stream_sid}].")

    return session