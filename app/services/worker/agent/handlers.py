import asyncio

from livekit.agents.voice import ConversationItemAddedEvent, CloseEvent, ErrorEvent, CloseReason
from livekit.agents.llm.chat_context import ChatMessage, ChatRole

from shared.infra.redis.session_store import append_turn
from shared.infra.postgres import save_call_log, save_tool_log, save_transcript


def conversation_item_added_event_handler(event: ConversationItemAddedEvent, stream_sid: str):
    """
    """
    if not isinstance(event.item, ChatMessage):
        return

    transcription = event.item.text_content()
    if not transcription:
        return

    role: ChatRole = event.item.role
    asyncio.create_task(append_turn(stream_sid=stream_sid, speaker=role, text=transcription))

        
def close_event_handler(event: CloseEvent):
    """
    """
    pass


def error_event_handler(event: ErrorEvent):
    """
    """
    pass


async def shutdown_callback():
    """
    """
    pass