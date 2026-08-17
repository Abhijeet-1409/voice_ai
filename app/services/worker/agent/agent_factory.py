from livekit.agents import Agent
from livekit.agents.llm import FunctionTool

from shared.logging_setup.logger import get_logger

from app.services.worker.schemas.session_data import UserData
from domain.tools.knowledge_base import search_knowledge_base
from domain.tools.crm import get_customer_profile, create_ticket
from domain.tools.sales_qualification import qualify_lead, schedule_meeting, send_followup_email
from app.services.worker.domain.system_prompt import SALES_QUALIFICATION_PROMPT, UNIFIED_INBOUND_PROMPT, DEFAULT_PROMPT


_LOGGER = "worker.agent.agent_factory"
logger = get_logger(_LOGGER)


def build_agent(user_data: UserData) -> Agent:
    """
    Constructs and caches a LiveKit Agent tailored to a specific call type.

    The @cache decorator ensures that identical Agent configurations are
    reused for the same call types, saving memory and initialization overhead.
    Both the system prompt and the available tools are dynamically composed
    based on whether the call is 'inbound' or 'outreach'.

    Args:
        call_type (CallType): The direction or type of the call (e.g., 'inbound' or 'outreach').
                              This dictates the agent's persona, instructions, and toolset.

    Returns:
        Agent: A configured LiveKit Agent equipped with the appropriate prompt and tools.
    """

    prompt: str = DEFAULT_PROMPT
    tool_list: list[FunctionTool] = [search_knowledge_base, send_followup_email]

    call_type = user_data.call_type

    match call_type:

        case "inbound":
            prompt = UNIFIED_INBOUND_PROMPT
            tool_list.extend([get_customer_profile, create_ticket, qualify_lead, schedule_meeting])

        case "outreach":
            prompt = SALES_QUALIFICATION_PROMPT
            tool_list.extend([qualify_lead, schedule_meeting])

        case _:
            logger.warning(f"Unrecognized call_type='{call_type}' — falling back to DEFAULT_PROMPT.")

    agent = Agent(
        instructions=prompt,
        tools=tool_list
    )

    logger.debug(f"Successfully built and cached the '{call_type}' Agent with {len(tool_list)} configured tools.")

    return agent