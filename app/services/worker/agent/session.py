from livekit.plugins import google
from livekit.agents import AgentSession
from livekit.plugins import silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from shared.logging_setup import get_logger

from schemas import UserData
from config import get_worker_settings


_LOGGER = "worker.agent.session"
logger = get_logger(_LOGGER)


def create_agent_session(api_key: str, user_data: UserData) -> AgentSession[UserData]:
    """
    Creates and configures a new LiveKit AgentSession for an incoming voice call.

    This function assembles a complete multilingual voice pipeline, combining local
    Voice Activity Detection (Silero), multilingual turn detection, Cartesia STT/TTS,
    and a Google Gemini LLM (cascade pipeline — text in, text out; not the Live/
    realtime API). The UserData instance is bound directly to the session — not
    converted to a dict — since every tool and Task in this codebase reads/writes
    it via attribute access (e.g. ctx.userdata.customer_id) and relies on Pydantic
    validation on mutation (validate_assignment=True on UserData).

    Args:
        api_key (str): The Google API key to authenticate the Gemini LLM.
        user_data (UserData): This call's UserData instance (already
            constructed and populated by job_entrypoint.py) to be bound
            to this session.

    Returns:
        AgentSession[UserData]: The fully initialized voice agent session
            ready to be started within a LiveKit room context.
    """
    settings = get_worker_settings()

    logger.info(
        f"Initializing AgentSession for stream [{user_data.stream_sid}] — "
        f"LLM: '{settings.GEMINI_MODEL}', Cartesia TTS: '{settings.CARTESIA_TTS_MODEL}'"
    )

    session = AgentSession[UserData](
        userdata=user_data,
        vad=silero.VAD.load(),
        stt=cartesia.STT(
            api_key=settings.CARTESIA_API_KEY,
            model=settings.CARTESIA_STT_MODEL,
        ),
        llm=google.LLM(
            model=settings.GEMINI_MODEL,
            temperature=settings.GEMINI_TEMPERATURE,
            max_output_tokens=settings.GEMINI_TOKEN_LIMIT,
            api_key=api_key,
        ),
        tts=cartesia.TTS(
            api_key=settings.CARTESIA_API_KEY,
            voice_id=settings.CARTESIA_VOICE_ID,
            model=settings.CARTESIA_TTS_MODEL,
        ),
        turn_detection=MultilingualModel(),
    )

    logger.debug(f"AgentSession successfully constructed and bound to stream [{user_data.stream_sid}].")

    return session