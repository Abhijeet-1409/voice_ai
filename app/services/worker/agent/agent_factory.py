from livekit.agents import ChatContext
from livekit.agents.llm import FunctionTool

from shared.config import CallType
from shared.logging_setup import get_logger

from .assistant import Assistant
from config import get_worker_settings
from schemas.session_data import UserData
from domain.tools import search_knowledge_base, update_caller_info, create_ticket, get_tickets, qualify_lead


_LOGGER = "worker.agent.agent_factory"
logger = get_logger(_LOGGER)


COMMON_TOOLS: list[FunctionTool] = [search_knowledge_base, update_caller_info]
SUPPORT_TOOLS: list[FunctionTool] = [create_ticket, get_tickets]
SALES_TOOLS: list[FunctionTool] = [qualify_lead]


def build_agent(user_data: UserData, instructions: str, chat_ctx: ChatContext | None = None) -> Assistant:
    """
    Constructs and configures an Assistant instance tailored to the given call's context.

    Dynamically selects the available FunctionTool set based on the call type:
      - OUTREACH: Configured with COMMON_TOOLS + SALES_TOOLS.
      - INBOUND: Configured with COMMON_TOOLS + SUPPORT_TOOLS + SALES_TOOLS.
      - Fallback: Defaults to COMMON_TOOLS only.

    Note: Bound tools on the Assistant instance itself (such as meeting scheduling) 
    are appended automatically during Assistant initialization.

    Args:
        user_data (UserData): The session context for the active call.
        instructions (str): The system prompt instructions to configure on the assistant.
        chat_ctx (ChatContext | None, optional): An existing conversation history context 
            to restore or initialize the assistant with. Defaults to None.

    Returns:
        Assistant: A fully configured Assistant instance ready for the voice session.
    """
    settings = get_worker_settings()
    call_type = user_data.call_type

    match call_type:
        case CallType.OUTREACH:
            tool_list = list(COMMON_TOOLS) + list(SALES_TOOLS)

        case CallType.INBOUND:
            tool_list = list(COMMON_TOOLS) + list(SUPPORT_TOOLS) + list(SALES_TOOLS)

        case _:
            tool_list = list(COMMON_TOOLS)

    assistant = Assistant(
        instructions=instructions, 
        tools=tool_list,  
        name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
        user_data=user_data,
        chat_ctx=chat_ctx
    )

    logger.debug(
        f"Built Assistant for call_type='{call_type}' with {len(tool_list)} "
        f"configured tools (plus Assistant's own bound tools)."
    )

    return assistant