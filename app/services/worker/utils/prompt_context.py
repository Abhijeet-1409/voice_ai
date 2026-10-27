from typing import Literal, Optional

from shared.config import CallType
from shared.logging_setup import get_logger

from schemas import UserData
from config import get_worker_settings
from domain import (
        OUTREACH_SYSTEM_PROMPT,
        INBOUND_SYSTEM_PROMPT,
        DEFAULT_SYSTEM_PROMPT,
        TASK_IDENTITY_TEMPLATE,
        SUPERVISOR_CONTEXT_TEMPLATE,
        WORKFLOW_POSITION_TEMPLATE,
        CONFIRM_EMAIL_TASK_PROMPT,
        CHOOSE_SLOT_TASK_PROMPT,
        OUTREACH_SYSTEM_PROMPT_V2,
        INBOUND_SYSTEM_PROMPT_V2,
        DEFAULT_SYSTEM_PROMPT_V2,
        VOICE_AGENT_GUIDE,
        VOICE_AGENT_GUIDE_V2
    )


_LOGGER = "worker.utils.prompt_context"
logger = get_logger(_LOGGER)


def build_instruction(call_type: CallType, version: Optional[Literal[1, 2]] = None) -> str:
    """
    Constructs the system prompt instruction string based on the call type and prompt version.

    Selects the appropriate prompt template (`OUTREACH_PROMPT`, `INBOUND_PROMPT`, or
    `DEFAULT_PROMPT`) based on the call type and requested version, then populates it
    with global application settings and dynamically formatted enum references required
    by the active call flow.

    Args:
        call_type (CallType): The type of call used to select the appropriate prompt template.
        version (Optional[Literal[1, 2]]): The system prompt version to use. Version 1
            uses the original prompts, while version 2 uses the updated prompts. Defaults
            to None when no specific version is requested.

    Returns:
        str: The fully formatted system instruction prompt.
    """
    settings = get_worker_settings()

    if version is None:
        # Default to version 1 if not specified
        version = 1

    if version not in (1, 2):
        logger.warning(f"Invalid version '{version}' specified — defaulting to version 1.")
        version = 1  # fallback to version 1 if invalid

    match call_type:
        case CallType.OUTREACH:
            if version == 1:
                prompt_template = OUTREACH_SYSTEM_PROMPT
            else:
                prompt_template = OUTREACH_SYSTEM_PROMPT_V2

        case CallType.INBOUND:
            if version == 1:
                prompt_template = INBOUND_SYSTEM_PROMPT
            else:
                prompt_template = INBOUND_SYSTEM_PROMPT_V2

        case _:
            logger.warning(f"Unrecognized call_type='{call_type}' — falling back to DEFAULT_PROMPT.")
            if version == 1:
                prompt_template = DEFAULT_SYSTEM_PROMPT
            else:
                prompt_template = DEFAULT_SYSTEM_PROMPT_V2

    instructions = prompt_template.format(
        agent_name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
    )

    return instructions


def build_voice_agent_guide(enum_reference: str, version: Optional[Literal[1, 2]] = None) -> str:
    """Builds and formats the voice agent instruction guide for a specific version.

    Args:
        enum_reference: The string representation of the enum reference value
            to be interpolated into the prompt template.
        version: The guide template version to use. Options are 1 or 2.
            If None or invalid, defaults to version 1.

    Returns:
        The formatted voice agent guide string containing company settings
        and enum references.
    """
    settings = get_worker_settings()

    if version is None:
        # Default to version 1 if not specified
        version = 1

    if version not in (1, 2):
        logger.warning(f"Invalid version '{version}' specified — defaulting to version 1.")
        version = 1  # fallback to version 1 if invalid

    if version == 1:
        guide = VOICE_AGENT_GUIDE
    else:
        guide = VOICE_AGENT_GUIDE_V2

    return guide.format(
        company_name=settings.COMPANY_NAME,
        enum_reference=enum_reference,
    )

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


def build_task_identity(
    *,
    task_name: str,
    supervisor_name: str,
) -> str:
    """Build the identity section for a bounded AgentTask."""
    return TASK_IDENTITY_TEMPLATE.format(
        task_name=task_name,
        supervisor_name=supervisor_name,
    )


def build_supervisor_context(
    *,
    supervisor_name: str,
    supervisor_role: str,
    invocation_reason: str,
    known_context: str,
    expected_result: str,
    already_completed: str,
) -> str:
    """Build the Supervisor context section for a bounded AgentTask."""
    return SUPERVISOR_CONTEXT_TEMPLATE.format(
        supervisor_name=supervisor_name,
        supervisor_role=supervisor_role,
        invocation_reason=invocation_reason,
        known_context=known_context,
        expected_result=expected_result,
        already_completed=already_completed,
    )


def build_workflow_position(
    *,
    previous_step: str,
    current_step: str,
    next_step: str,
) -> str:
    """Build the workflow-position section for a bounded AgentTask."""
    return WORKFLOW_POSITION_TEMPLATE.format(
        previous_step=previous_step,
        current_step=current_step,
        next_step=next_step,
    )


def build_confirm_email_task_prompt(
    *,
    supervisor_name: str,
    supervisor_role: str,
    invocation_reason: str,
    known_context: str,
    expected_result: str,
    already_completed: str,
    previous_step: str,
    current_step: str,
    next_step: str,
) -> str:
    """Render the complete ConfirmEmailTask prompt."""
    return (
        build_task_identity(
            task_name="ConfirmEmailTask",
            supervisor_name=supervisor_name,
        )
        + build_supervisor_context(
            supervisor_name=supervisor_name,
            supervisor_role=supervisor_role,
            invocation_reason=invocation_reason,
            known_context=known_context,
            expected_result=expected_result,
            already_completed=already_completed,
        )
        + build_workflow_position(
            previous_step=previous_step,
            current_step=current_step,
            next_step=next_step,
        )
        + CONFIRM_EMAIL_TASK_PROMPT
    )


def build_choose_slot_task_prompt(
    *,
    supervisor_name: str,
    supervisor_role: str,
    invocation_reason: str,
    known_context: str,
    expected_result: str,
    already_completed: str,
    previous_step: str,
    current_step: str,
    next_step: str,
    meeting_slot_description: str,
) -> str:
    """
    Render the complete ChooseSlotTask prompt.

    The actual available meeting slots are intentionally NOT injected here.
    They are runtime state and should be fetched and injected by
    ChooseSlotTask.on_enter().
    """
    return (
        build_task_identity(
            task_name="ChooseSlotTask",
            supervisor_name=supervisor_name,
        )
        + build_supervisor_context(
            supervisor_name=supervisor_name,
            supervisor_role=supervisor_role,
            invocation_reason=invocation_reason,
            known_context=known_context,
            expected_result=expected_result,
            already_completed=already_completed,
        )
        + build_workflow_position(
            previous_step=previous_step,
            current_step=current_step,
            next_step=next_step,
        )
        + CHOOSE_SLOT_TASK_PROMPT.format(
            meeting_slot_description=meeting_slot_description,
        )
    )