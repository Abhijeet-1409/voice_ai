import json
from functools import partial

from livekit.plugins import noise_cancellation
from livekit.agents import room_io, AutoSubscribe, JobContext


from shared.logging_setup.logger import get_logger, stream_sid_var
from shared.config.constants import CallType, Channel


from app.services.worker.schemas.session_data import UserData
from agent.agent_factory import build_agent
from agent.session import create_agent_session
from agent.key_selector import select_gemini_key
from agent.handlers import conversation_item_added_event_handler, close_event_handler, error_event_handler


_LOGGER = "worker.agent.entrypoint"
logger = get_logger(_LOGGER)


async def _entrypoint(ctx: JobContext):
    """
    """
    try:

        # select gemini_key
        gemini_key = await select_gemini_key()

        # connect to the room
        ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # extracting channel and call_type from metadata
        metadata: dict = json.load(ctx.room.metadata)
        stream_sid: str = metadata.get("stream_sid")
        channel: Channel = metadata.get("channel")
        call_type: CallType = metadata.get("call_type")

        # create a user_data object
        user_data = UserData(
            channel=channel,
            call_type=call_type,
            stream_sid=stream_sid,
            clerk_id=metadata.get("clerk_id"),
            phone=metadata.get("phone"),
            email=metadata.get("email"),
        )

        # setting stream_sid context var
        stream_sid_var.set(stream_sid)

        # create a agent session
        session = create_agent_session(gemini_key, user_data)

        # register conversation_item_added_event_handler func for conversation_item_added event on session object
        session.on("conversation_item_added", partial(conversation_item_added_event_handler, stream_sid = stream_sid))

        # register close_event_handler func for close event on session object
        session.on("close", partial(close_event_handler, stream_sid = stream_sid))

        # register close_event_handler func for error event on session object
        session.on("error", partial(error_event_handler, stream_sid = stream_sid))

        # create a agent object
        agent = build_agent(user_data)

        # Start the session with noise cancellation enabled
        await session.start(
            agent=agent,
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
                ),
            ),
        )

        ctx.add_shutdown_callback()

    except Exception as e:
        logger.error()
        raise





