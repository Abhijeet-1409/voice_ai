import asyncio
from typing import Optional

from livekit.plugins import google
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import silero, cartesia, deepgram 

from google.genai.client import Client
from google.genai.types import CachedContent

from shared.logging_setup import get_logger

from schemas import UserData
from config import get_worker_settings
from .agent_factory import build_instruction


_LOGGER = "worker.agent.session"
logger = get_logger(_LOGGER)


def create_prompt_cache(user_data: UserData) -> Optional[str]:
    """
    Creates a context cache on Vertex AI for the system instructions.

    Constructs the system prompt based on call type metadata and registers it 
    with Vertex AI under a dedicated display name and configured TTL. 
    Caching system prompts reduces latency and token processing costs for subsequent requests.

    Args:
        user_data (UserData): The session context containing call type and user metadata.

    Returns:
        Optional[str]: The resource name of the created Vertex AI cache (e.g., 'cachedContents/1234567890'), 
        or None if cache creation fails.
    """
    settings = get_worker_settings()

    try:
        client = Client(
            enterprise=True, 
            project=settings.PROJECT, 
            location=settings.LOCATION
        )

        system_instruction = build_instruction(user_data)
        logger.debug(f"Cache system_instruction length: {len(system_instruction)} chars")
        cache_display_name = f"{user_data.call_type.value}_system_prompt"

        cached_content: CachedContent = client.caches.create(
            model=settings.GEMINI_MODEL,
            config={
                "display_name": cache_display_name,
                "system_instruction": system_instruction,
                "ttl": f"{settings.CACHED_CONTENT_TTL}s",
            }
        )

        logger.info(
            f"Successfully created Vertex AI prompt cache '{cached_content.name}' "
            f"[display_name={cache_display_name}, stream_sid={user_data.stream_sid}]"
        )
        return cached_content.name

    except Exception as err:
        logger.error(
            f"Failed to create Vertex AI prompt cache for call_type='{user_data.call_type}' "
            f"[stream_sid={user_data.stream_sid}]: {err}",
        )
        return None


async def _warm_llm(user_data: UserData, llm: google.LLM) -> None:
    """
    Executes a lightweight, dummy generation request to pre-warm the Vertex AI LLM client.

    Reduces first-turn latency by forcing the internal client and connection pools 
    to initialize prior to active caller interaction.

    Args:
        user_data (UserData): Session context used for contextual log references.
        llm (google.LLM): The LiveKit Google LLM plugin instance to pre-warm.
    """
    settings = get_worker_settings()

    try:
        client: Client = llm._client  # private attribute access on LiveKit google.LLM wrapper
        await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents="hi",
            config={"max_output_tokens": 1},
        )
        logger.debug(f"LLM warm-up completed successfully [stream_sid={user_data.stream_sid}]")
    except Exception as err:
        logger.warning(
            f"LLM warm-up failed (non-fatal) [stream_sid={user_data.stream_sid}]: {err}",
            exc_info=True
        )


async def _warm_tts(user_data: UserData, tts: cartesia.TTS) -> None:
    """
    Executes the pre-warm routine on the Cartesia TTS plugin instance.

    Offloads socket pre-connection and web-socket handshakes to a background thread 
    to optimize initial speech synthesis speed.

    Args:
        user_data (UserData): Session context used for contextual log references.
        tts (cartesia.TTS): The Cartesia TTS plugin instance to pre-warm.
    """
    try:
        await asyncio.to_thread(tts.prewarm)
        logger.debug(f"TTS warm-up completed successfully [stream_sid={user_data.stream_sid}]")
    except Exception as err:
        logger.warning(f"TTS warm-up failed (non-fatal) [stream_sid={user_data.stream_sid}]: {err}")


async def warm_up_pipeline(user_data: UserData, llm: google.LLM, tts: cartesia.TTS) -> None:
    """
    Concurrently executes pre-warming routines for pipeline components (LLM and TTS).

    Runs model warm-ups in parallel to minimize overall startup overhead before starting 
    the active session. Captures all step errors safely as non-fatal warnings.

    Args:
        user_data (UserData): Session context for logging and call identification.
        llm (google.LLM): The Google Gemini LLM instance to warm up.
        tts (cartesia.TTS): The Cartesia TTS instance to warm up.
    """
    results = await asyncio.gather(
        _warm_llm(user_data, llm),
        _warm_tts(user_data, tts),
        return_exceptions=True,
    )
    for res in results:
        if isinstance(res, Exception):
            logger.warning(f"Warm-up pipeline step raised unexpectedly [stream_sid={user_data.stream_sid}]: {res}")


async def create_agent_session(user_data: UserData) -> AgentSession[UserData]:
    """
    Creates and configures a new LiveKit AgentSession for an incoming voice call.

    Assembles a complete voice processing pipeline combining local Voice Activity 
    Detection (Silero), STT (Deepgram v2), turn handling options, Gemini LLM via 
    Vertex AI with context caching, and Cartesia TTS. Binds the provided UserData instance 
    directly to the session lifecycle.

    Args:
        user_data (UserData): The session context instance containing caller profile 
            and call metadata.

    Returns:
        AgentSession[UserData]: The initialized voice agent session ready for live room execution.
    """
    settings = get_worker_settings()

    logger.info(f"Initializing AgentSession for stream [{user_data.stream_sid}]")

    cache_name = create_prompt_cache(user_data)

    vad = silero.VAD.load(
        min_speech_duration=settings.VAD_MIN_SPEECH_DURATION,
        min_silence_duration=settings.VAD_MIN_SILENCE_DURATION,
        activation_threshold=settings.VAD_ACTIVATION_THRESHOLD,
    )

    stt = deepgram.STTv2(
        model=settings.DEEPGRAM_STT_MODEL,             
        language_hint=["hi", "en", "mr", "bn", "ta", "te", "gu", "kn", "ml", "ur"],
        keyterm=settings.DEEPGRAM_STT_KEYTERMS,
        eot_threshold=settings.DEEPGRAM_STT_EOU_THRESHOLD,
        eot_timeout_ms=settings.DEEPGRAM_STT_EOU_TIMEOUT_MS,
    )

    turn_handling = TurnHandlingOptions(
        turn_detection="stt",
        endpointing={
            "mode": settings.TURNHANDLING_ENDPOINTING_MODE,
            "min_delay": settings.TURNHANDLING_ENDPOINTING_MIN_DELAY,
            "max_delay": settings.TURNHANDLING_ENDPOINTING_MAX_DELAY,
        },
        preemptive_generation={
            "enabled": settings.TURNHANDLING_PREEMPTIVE_GENERATION_ENABLED,
            "preemptive_tts": settings.TURNHANDLING_PREEMPTIVE_GENERATION_PREEMPTIVE_TTS,
            "max_retries": settings.TURNHANDLING_PREEMPTIVE_GENERATION_PREEMPTIVE_TTS_MAX_RETRIES_PER_TURN,
            "max_speech_duration": settings.TURNHANDLING_PREEMPTIVE_GENERATION_PREEMPTIVE_TTS_MAX_SPEECH_DURATION,
        },
    )

    llm = google.LLM(
        model=settings.GEMINI_MODEL,
        temperature=settings.GEMINI_TEMPERATURE,
        max_output_tokens=settings.GEMINI_TOKEN_LIMIT,
        thinking_config={"thinking_budget": settings.GEMINI_THINKING_BUDGET},
        vertexai=settings.VERTEXAI,
        project=settings.PROJECT,
        location=settings.LOCATION,
        cached_content=cache_name,
    )

    tts = cartesia.TTS(
        api_key=settings.CARTESIA_API_KEY,
        voice=settings.CARTESIA_VOICE_ID,
        model=settings.CARTESIA_TTS_MODEL,
        language=settings.CARTESIA_TTS_LANGUAGE
    )

    logger.info(f"llm cachce conent: {llm._opts.cached_content}")
    await warm_up_pipeline(user_data, llm, tts)


    session = AgentSession[UserData](
        userdata=user_data,
        vad=vad,
        stt=stt,
        turn_handling=turn_handling,
        llm=llm,
        tts=tts,
    )

    logger.debug(f"AgentSession successfully constructed and bound to stream [{user_data.stream_sid}].")

    return session