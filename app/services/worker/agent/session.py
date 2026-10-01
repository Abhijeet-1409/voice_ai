import asyncio
from typing import Optional

import livekit.plugins.google as livekit_google_plugin
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import silero, cartesia, deepgram, sarvam

from google.genai.client import Client

from shared.logging_setup import get_logger

from schemas import UserData
from config import get_worker_settings


_LOGGER = "worker.agent.session"
logger = get_logger(_LOGGER)


async def _warm_llm(user_data: UserData, llm: livekit_google_plugin.LLM) -> None:
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


async def warm_up_pipeline(user_data: UserData, llm: livekit_google_plugin.LLM, tts: cartesia.TTS) -> None:
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


async def create_agent_session(user_data: UserData, cached_content: Optional[str] = None) -> AgentSession[UserData]:
    """
    Creates and configures a new LiveKit AgentSession for an incoming voice call.

    Assembles a complete voice processing pipeline combining local Voice Activity 
    Detection (Silero), STT (Deepgram v2), turn handling options, Gemini LLM via 
    Vertex AI with optional context caching, and Cartesia TTS. Binds the provided 
    UserData instance directly to the session lifecycle.

    Args:
        user_data (UserData): The session context instance containing caller profile 
            and call metadata.
        cached_content (Optional[str], optional): The resource name of a pre-created Vertex AI 
            prompt cache to attach to the LLM configuration. Defaults to None.

    Returns:
        AgentSession[UserData]: The initialized voice agent session ready for live room execution.
    """
    settings = get_worker_settings()

    logger.info(f"Initializing AgentSession for stream [{user_data.stream_sid}]")

    vad = silero.VAD.load(
        min_speech_duration=settings.VAD_MIN_SPEECH_DURATION,
        min_silence_duration=settings.VAD_MIN_SILENCE_DURATION,
        activation_threshold=settings.VAD_ACTIVATION_THRESHOLD,
    )

    # stt = deepgram.STTv2(
    #     model=settings.DEEPGRAM_STT_MODEL,             
    #     language_hint=["hi", "en", "mr", "bn", "ta", "te", "gu", "kn", "ml", "ur"],
    #     keyterm=settings.DEEPGRAM_STT_KEYTERMS,
    #     eot_threshold=settings.DEEPGRAM_STT_EOU_THRESHOLD,
    #     eot_timeout_ms=settings.DEEPGRAM_STT_EOU_TIMEOUT_MS,
    # )

    stt = cartesia.STT(
        model=settings.CARTESIA_STT_MODEL,
        api_key=settings.CARTESIA_API_KEY,
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

    llm = livekit_google_plugin.LLM(
        model=settings.GEMINI_MODEL,
        temperature=settings.GEMINI_TEMPERATURE,
        max_output_tokens=settings.GEMINI_TOKEN_LIMIT,
        thinking_config={"thinking_budget": settings.GEMINI_THINKING_BUDGET},
        vertexai=settings.VERTEXAI,
        project=settings.PROJECT,
        location=settings.LOCATION,
        cached_content=cached_content,
    )

    tts = cartesia.TTS(
        api_key=settings.CARTESIA_API_KEY,
        voice=settings.CARTESIA_VOICE_ID,
        model=settings.CARTESIA_TTS_MODEL,
        language=settings.CARTESIA_TTS_LANGUAGE
    )

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