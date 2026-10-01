from shared.config import CallType
from shared.logging_setup import get_logger

from schemas import UserData
from config import get_worker_settings
from domain import OUTREACH_SYSTEM_PROMPT, INBOUND_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT


_LOGGER = "worker.utils.prompt_context"
logger = get_logger(_LOGGER)


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

        case CallType.INBOUND:
            prompt_template = INBOUND_SYSTEM_PROMPT

        case _:
            logger.warning(f"Unrecognized call_type='{call_type}' — falling back to DEFAULT_PROMPT.")
            prompt_template = DEFAULT_SYSTEM_PROMPT

    instructions = prompt_template.format(
        agent_name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
    )

    return instructions


def build_user_context_block(userdata: UserData) -> str:
    """
    Renders UserData into a fixed-shape, LLM-friendly text block for
    injection into the {user_context} placeholder in domain/system_prompt.py's
    prompt templates.

    Always includes every field (using "Unknown"/"Not yet determined" for
    unset values) rather than conditionally omitting empty fields — see
    design decision: a fixed, predictable shape every call is preferred
    over variable-length output, and explicit placeholders avoid raw
    "None" strings leaking into the prompt.

    Args:
        userdata: The current call's UserData.

    Returns:
        A formatted multi-line string ready to fill {user_context}.
    """
    return (
        f"Customer_id: {userdata.customer_id or 'Unknown'}\n"
        f"Contact name: {userdata.name or 'Unknown'}\n"
        f"Phone number: {userdata.phone or 'Unknown'}\n"
        f"Email: {userdata.email or 'Unknown'}\n"
        f"Channel: {userdata.channel}\n"
        f"Call type: {userdata.call_type}\n"
        f"Track: {userdata.track or 'Not yet determined'}\n"
        f"Previously qualified: {'Yes' if userdata.qualified else 'No'}\n"
        f"Lifecycle stage: {userdata.lifecyclestage}"
    )