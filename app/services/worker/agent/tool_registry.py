from typing import Optional

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
    tool_dependencies: ToolDependencies,
    call_type: Optional[CallType] = None,
) -> list[FunctionTool]:
    """
    Select and assemble the function tools available for a call.

    Selects standard tools based on the specified call type. If the call
    type is OUTREACH, common and sales tools are selected. If the call
    type is INBOUND, common, support, and sales tools are selected. If
    the call type is None or does not match a supported type, only common
    tools are selected.

    Contact-confirmation and meeting-scheduling tools are created using
    the supplied runtime dependencies and appended to the selected tools
    for every call type.

    Args:
        tool_dependencies (ToolDependencies): Runtime dependencies used
            to create the contact-confirmation and meeting-scheduling tools.
        call_type (Optional[CallType]): Type of call determining the
            standard tool set. Defaults to None, which selects only common
            tools before the additional tools are appended.

    Returns:
        list[FunctionTool]: The complete list of function tools available
            to the assistant for the specified call type.
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
