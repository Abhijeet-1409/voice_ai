from typing import Literal, Optional

from shared.config import CallType
from shared.logging_setup import get_logger

from schemas import UserData
from config import get_worker_settings
from domain import (
    OUTREACH_SYSTEM_PROMPT,
    INBOUND_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    OUTREACH_SYSTEM_PROMPT_V2,
    INBOUND_SYSTEM_PROMPT_V2,
    DEFAULT_SYSTEM_PROMPT_V2,
    OUTREACH_SYSTEM_PROMPT_V3,
    INBOUND_SYSTEM_PROMPT_V3,
    DEFAULT_SYSTEM_PROMPT_V3,
    TASK_IDENTITY_TEMPLATE,
    SUPERVISOR_CONTEXT_TEMPLATE,
    WORKFLOW_POSITION_TEMPLATE,
    CONFIRM_EMAIL_TASK_PROMPT,
    CHOOSE_SLOT_TASK_PROMPT,
    VOICE_AGENT_GUIDE,
    VOICE_AGENT_GUIDE_V2,
    AWS_PROGRAM_REFERENCE,
    INBOUND_VOICE_AGENT_GUIDE,
    OUTREACH_VOICE_AGENT_GUIDE,
    DEFAULT_VOICE_AGENT_GUIDE,
)

_LOGGER = "worker.utils.prompt_context"
logger = get_logger(_LOGGER)


def build_instruction(
    call_type: Optional[CallType] = None,
    version: Optional[Literal[1, 2, 3]] = None,
) -> str:
    """Build the system instruction for the requested call type and version.

    Version 3 is the default. Versions 1 and 2 remain available for rollback.
    """
    settings = get_worker_settings()
    version = 3 if version is None else version

    if version not in (1, 2, 3):
        logger.warning(
            "Invalid system instruction version %r; defaulting to version 3.",
            version,
        )
        version = 3

    templates = {
        1: {
            CallType.OUTREACH: OUTREACH_SYSTEM_PROMPT,
            CallType.INBOUND: INBOUND_SYSTEM_PROMPT,
            None: DEFAULT_SYSTEM_PROMPT,
        },
        2: {
            CallType.OUTREACH: OUTREACH_SYSTEM_PROMPT_V2,
            CallType.INBOUND: INBOUND_SYSTEM_PROMPT_V2,
            None: DEFAULT_SYSTEM_PROMPT_V2,
        },
        3: {
            CallType.OUTREACH: OUTREACH_SYSTEM_PROMPT_V3,
            CallType.INBOUND: INBOUND_SYSTEM_PROMPT_V3,
            None: DEFAULT_SYSTEM_PROMPT_V3,
        },
    }

    prompt_template = templates[version].get(call_type)

    if prompt_template is None:
        logger.warning(
            "Unrecognized call_type=%r; using the DEFAULT system prompt.",
            call_type,
        )
        prompt_template = templates[version][None]

    return prompt_template.format(
        agent_name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
    )


def build_voice_agent_guide(
    enum_reference: str,
    version: Optional[Literal[1, 2, 3]] = None,
    call_type: Optional[CallType] = None,
) -> str:
    """Build the business guide for one call type.

    Versions 1 and 2 preserve their historical single-guide behaviour.
    Version 3 selects exactly one call-type-specific guide and injects
    the runtime enum reference into that guide.
    """
    settings = get_worker_settings()
    version = 3 if version is None else version

    if version not in (1, 2, 3):
        logger.warning(
            "Invalid voice guide version %r; defaulting to version 3.",
            version,
        )
        version = 3

    if version == 1:
        guide = VOICE_AGENT_GUIDE
        return guide.format(
            company_name=settings.COMPANY_NAME,
            enum_reference=enum_reference,
        )

    if version == 2:
        guide = VOICE_AGENT_GUIDE_V2
        return guide.format(
            company_name=settings.COMPANY_NAME,
            enum_reference=enum_reference,
        )

    match call_type:
        case CallType.INBOUND:
            guide = INBOUND_VOICE_AGENT_GUIDE
        case CallType.OUTREACH:
            guide = OUTREACH_VOICE_AGENT_GUIDE
        case _:
            logger.warning(
                "Unrecognized call_type=%r; using DEFAULT V3 guide.",
                call_type,
            )
            guide = DEFAULT_VOICE_AGENT_GUIDE

    return guide.format(
        company_name=settings.COMPANY_NAME,
        enum_reference=enum_reference,
        aws_program_reference=AWS_PROGRAM_REFERENCE,
    )


def build_user_context_block(userdata: UserData) -> str:
    """
    Renders UserData into a fixed-shape, LLM-friendly text block for
    injection into the {user_context} placeholder in domain/system_prompt.py's
    prompt templates.

    Always includes every field, using "Unknown" or "Not yet determined"
    for unset values.
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

    Actual available meeting slots are runtime state and should be fetched
    and injected by ChooseSlotTask.on_enter().
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
