from livekit.agents.llm import FunctionTool

from shared.config import CallType

from schemas import ToolDependencies
from domain.tools import (
    search_knowledge_base,
    update_caller_info,
    create_ticket,
    get_tickets,
    qualify_lead,
    create_get_contact_info_tool,
    create_schedule_meeting_tool,
)


COMMON_TOOLS: list[FunctionTool] = [
    search_knowledge_base,
    update_caller_info,
]

SUPPORT_TOOLS: list[FunctionTool] = [
    create_ticket,
    get_tickets,
]

SALES_TOOLS: list[FunctionTool] = [
    qualify_lead,
]


def select_tools(
    call_type: CallType,
    tool_dependencies: ToolDependencies,
) -> list[FunctionTool]:
    """
    Select and assemble the function tools available for a call.

    Standard tools are selected according to the call type.
    Contact-confirmation and meeting-scheduling tools are created
    with the supplied runtime dependencies and appended for every
    call type.

    Args:
        call_type: Type of call determining the standard tool set.
        tool_dependencies: Runtime dependencies for creating the
            contact-confirmation and scheduling tools.

    Returns:
        list[FunctionTool]: The complete tool list for the assistant.
    """
    get_contact_info = create_get_contact_info_tool(tool_dependencies)
    schedule_meeting = create_schedule_meeting_tool(tool_dependencies)

    match call_type:
        case CallType.OUTREACH:
            tool_list = [*COMMON_TOOLS, *SALES_TOOLS]

        case CallType.INBOUND:
            tool_list = [
                *COMMON_TOOLS,
                *SUPPORT_TOOLS,
                *SALES_TOOLS,
            ]

        case _:
            tool_list = list(COMMON_TOOLS)

    tool_list.extend([
        get_contact_info,
        schedule_meeting,
    ])

    return tool_list
