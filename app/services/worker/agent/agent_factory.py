from livekit.agents.llm import FunctionTool

from shared.logging_setup import get_logger
from shared.config import CallType, Track, TicketPriority, TicketStatus

from .assistant import Assistant
from config import get_worker_settings
from schemas.session_data import UserData
from utils import describe_all
from domain import INBOUND_SYSTEM_PROMPT, OUTREACH_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
from domain.tools import search_knowledge_base, update_caller_info, create_ticket, get_tickets, qualify_lead


_LOGGER = "worker.agent.agent_factory"
logger = get_logger(_LOGGER)


COMMON_TOOLS: list[FunctionTool] = [search_knowledge_base, update_caller_info]
SUPPORT_TOOLS: list[FunctionTool] = [create_ticket, get_tickets]
SALES_TOOLS: list[FunctionTool] = [qualify_lead]


def build_instruction(user_data: UserData) -> str:
    """
    Constructs the system prompt instruction string based on the call type.

    Renders the appropriate prompt template (`OUTREACH_PROMPT`, `INBOUND_PROMPT`, or `DEFAULT_PROMPT`) 
    by populating global application settings and dynamically formatting valid enum references required 
    by the active call flow.

    Args:
        user_data (UserData): The session context containing metadata and call specifications.

    Returns:
        str: The fully formatted system instruction prompt.
    """
    settings = get_worker_settings()
    call_type = user_data.call_type
    
    match call_type:
        case CallType.OUTREACH:
            prompt_template = OUTREACH_SYSTEM_PROMPT
            enum_reference = describe_all(Track)

        case CallType.INBOUND:
            prompt_template = INBOUND_SYSTEM_PROMPT
            enum_reference = describe_all(Track, TicketPriority, TicketStatus)

        case _:
            logger.warning(f"Unrecognized call_type='{call_type}' — falling back to DEFAULT_PROMPT.")
            prompt_template = DEFAULT_SYSTEM_PROMPT
            enum_reference = ""

    instructions = prompt_template.format(
        enum_reference=enum_reference,
        agent_name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
    )

    return instructions


def build_agent(user_data: UserData) -> Assistant:
    """
    Constructs and configures an Assistant instance tailored to the given call's context.

    Dynamically selects both the prompt instructions and available FunctionTool set 
    based on the call type:
      - OUTREACH: Configured with COMMON_TOOLS + SALES_TOOLS.
      - INBOUND: Configured with COMMON_TOOLS + SUPPORT_TOOLS + SALES_TOOLS.
      - Fallback: Defaults to COMMON_TOOLS only.

    Note: Bound tools on the Assistant instance itself (such as meeting scheduling) 
    are appended automatically during Assistant initialization.

    Args:
        user_data (UserData): The session context for the active call.

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

    instructions = build_instruction(user_data=user_data)

    assistant = Assistant(
        instructions=instructions, 
        tools=tool_list,  
        name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
        user_data=user_data
    )

    logger.debug(
        f"Built Assistant for call_type='{call_type}' with {len(tool_list)} "
        f"configured tools (plus Assistant's own bound tools)."
    )

    return assistant