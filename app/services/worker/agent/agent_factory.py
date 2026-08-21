from livekit.agents.llm import FunctionTool

from shared.logging_setup import get_logger
from shared.config import CallType, Track, TicketPriority, TicketStatus

from agent import Assistant
from schemas.session_data import UserData
from utils import build_user_context_block, describe_all
from domain import OUTREACH_PROMPT, INBOUND_PROMPT, DEFAULT_PROMPT
from domain.tools import search_knowledge_base, update_caller_info, create_ticket, get_tickets, qualify_lead


_LOGGER = "worker.agent.agent_factory"
logger = get_logger(_LOGGER)


COMMON_TOOLS: list[FunctionTool] = [search_knowledge_base, update_caller_info]
SUPPORT_TOOLS: list[FunctionTool] = [create_ticket, get_tickets]
SALES_TOOLS: list[FunctionTool] = [qualify_lead]


def build_agent(user_data: UserData) -> Assistant:
    """
    Constructs an Assistant configured for the given call's UserData.

    Both the system prompt and the available tool list are composed
    based on call_type:
      - OUTREACH: COMMON_TOOLS + SALES_TOOLS (no ticketing — outreach
        calls are agent-initiated, sales-only by nature).
      - INBOUND: COMMON_TOOLS + SUPPORT_TOOLS + SALES_TOOLS (branches
        internally between Support Flow and Qualification Flow).
      - unrecognized call_type: falls back to DEFAULT_PROMPT with
        COMMON_TOOLS only.

    The chosen prompt template is rendered with two injections:
      - {user_context}: this call's UserData, via build_user_context_block.
      - {enum_reference}: valid values for the enums this flow's tools
        actually use, via describe_all — Track only for outreach,
        Track + TicketPriority + TicketStatus for inbound, omitted
        for the default fallback (no enum-typed tool params there).

    schedule_meeting is NOT part of the tool lists here — it's a bound
    method on Assistant itself (needs self.chat_ctx to hand conversation
    history into ConfirmEmailTask/ChooseSlotTask, which a free function
    can't access), always appended by Assistant.__init__ regardless of
    call_type.

    Args:
        user_data: The current call's UserData.

    Returns:
        Assistant: A configured Assistant instance.
    """
    call_type = user_data.call_type

    match call_type:
        case CallType.OUTREACH:
            prompt_template = OUTREACH_PROMPT
            tool_list = list(COMMON_TOOLS) + list(SALES_TOOLS)
            enum_reference = describe_all(Track)

        case CallType.INBOUND:
            prompt_template = INBOUND_PROMPT
            tool_list = list(COMMON_TOOLS) + list(SUPPORT_TOOLS) + list(SALES_TOOLS)
            enum_reference = describe_all(Track, TicketPriority, TicketStatus)

        case _:
            logger.warning(f"Unrecognized call_type='{call_type}' — falling back to DEFAULT_PROMPT.")
            prompt_template = DEFAULT_PROMPT
            tool_list = list(COMMON_TOOLS)
            enum_reference = ""

    instructions = prompt_template.format(
        user_context=build_user_context_block(user_data),
        enum_reference=enum_reference,
    )

    assistant = Assistant(instructions=instructions, tools=tool_list)

    logger.debug(
        f"Built Assistant for call_type='{call_type}' with {len(tool_list)} "
        f"configured tools (plus Assistant's own bound tools)."
    )

    return assistant