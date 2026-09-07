import json
from functools import partial

from livekit.plugins import noise_cancellation
from livekit.agents import room_io, AutoSubscribe, JobContext
from livekit.agents.llm import RealtimeModel

from shared.call_context import stream_sid_var
from shared.logging_setup import get_logger
from shared.config import CallType, Channel
from shared.infra.postgres import db_init
from shared.infra.redis import ping_redis

from schemas import UserData
from .agent_factory import build_agent 
from .session import create_agent_session
from .key_selector import select_gemini_key
from .event_handlers import on_close, on_error, on_conversation_item_added, on_function_tools_executed, on_user_input_transcribed
from utils import lookup_customer
from config import get_worker_settings
from speech import OutputLanguage
from speech.adapters import CartesiaOutputLanguage


_LOGGER = "worker.agent.entrypoint"
logger = get_logger(_LOGGER)


async def entrypoint(ctx: JobContext) -> None:
    """
    Primary execution entrypoint for a LiveKit agent job (per-call).

    This function is triggered every time a new call routes to this worker. 
    It is responsible for bootstrapping the call lifecycle, which includes:
    1. Initializing network-bound singletons (Database, Redis) post-fork.
    2. Extracting call metadata and looking up the caller's profile.
    3. Constructing the unified UserData context object.
    4. Building the AgentSession and configuring the assistant.
    5. Registering event handlers for call lifecycle events.
    6. Starting the active voice session with noise cancellation.

    Args:
        ctx (JobContext): The context for the current LiveKit job, providing 
                          access to the room, connections, and metadata.
                          
    Raises:
        Exception: If call bootstrapping fails, the error is logged and re-raised 
                   to let the framework cleanly terminate the job.
    """
    try:
        settings = get_worker_settings()

        await db_init()  # ensure the database engine is initialized before any DB operations
        await ping_redis()  # ensure Redis is reachable

        # connect to the room first — metadata isn't reliably readable
        # until the room connection is established
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # extract call metadata
        metadata: dict = json.loads(ctx.room.metadata or "{}")
        stream_sid: str = ctx.room.name
        channel: Channel = Channel(metadata.get("channel", Channel.PHONE))
        call_type: CallType = CallType(metadata.get("call_type", CallType.INBOUND))

        # set stream_sid in logging context as early as possible, so every
        # subsequent log line in this call (including from key selection)
        # carries the right stream_sid
        stream_sid_var.set(stream_sid)
        logger.debug(f"Room metadata: {metadata}")
        
        # select a key — sync, no await
        gemini_key = select_gemini_key()

        # look up the caller by phone — pure lookup, never creates
        phone = metadata.get("phone")
        contact = await lookup_customer(phone) if phone else None

        # build user_data from the found contact if one exists, otherwise
        # from metadata alone — customer_id stays None in that case, and a
        # new contact is only created at call end (see
        # event_handlers._end_of_call_writes)
        if contact is not None:
            user_data = UserData(
                channel=channel,
                call_type=call_type,
                stream_sid=stream_sid,
                clerk_id=metadata.get("clerk_id"),
                phone=phone,
                email=contact.get("email") or metadata.get("email"),
                customer_id=contact.get("id"),
                name=contact.get("name"),
            )
        else:
            user_data = UserData(
                channel=channel,
                call_type=call_type,
                stream_sid=stream_sid,
                clerk_id=metadata.get("clerk_id"),
                phone=phone,
                email=metadata.get("email"),
            )

        # create the agent session and the configured assistant
        session = create_agent_session(gemini_key, user_data)
        user_data.is_realtime_model = isinstance(session.llm, RealtimeModel) 
        agent = build_agent(user_data)

        # determine the output language controller for TTS, if TTS is enabled
        language = settings.CARTESIA_TTS_LANGUAGE
        output_language: OutputLanguage | None = CartesiaOutputLanguage(language,session.tts) if session.tts is not None else None

        # register event handlers
        session.on(
            "conversation_item_added",
            partial(on_conversation_item_added, stream_sid=stream_sid),
        )
        session.on(
            "user_input_transcribed",
            partial(on_user_input_transcribed, stream_sid=stream_sid, output_language=output_language)
        )
        session.on(
            "close",
            partial(on_close, stream_sid=stream_sid, userdata=user_data),
        )
        session.on(
            "function_tools_executed",
            partial(on_function_tools_executed, userdata=user_data),
        )
        session.on(
            "error",
            partial(on_error, stream_sid=stream_sid),
        )

        # start the session with noise cancellation enabled
        await session.start(
            agent=agent,
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),
                ),
            ),
        )

    except Exception as e:
        logger.error("Error occurred in entrypoint", exc_info=True)
        raise