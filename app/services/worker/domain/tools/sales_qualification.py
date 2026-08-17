from livekit.agents import function_tool, RunContext

from shared.logging_setup.logger import get_logger
from shared.infra.crm.mock import get_mockcrmclient


_LOGGER = "worker.domain.tools.sales_qualification"
logger = get_logger(_LOGGER)


@function_tool
async def qualify_lead(
        ctx: RunContext,
        track: str,
        contact: str,
        interest_summary: str
    ) -> str:
    """
    Log a qualified lead against one of the three AWS partner tracks.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        track (str): The specific AWS partner track the contact is interested in.
        contact (str): The contact's unique identifier (e.g., phone number or user ID).
        interest_summary (str): A brief summary of the contact's needs or interests.

    Returns:
        str: A confirmation message containing the generated lead ID.
    """
    try:
        crm_client = get_mockcrmclient()
        interest = f"[{track}] {interest_summary}"
        lead_id = await crm_client.create_lead(contact, interest)

        logger.debug(f"Created qualified lead — contact={contact} track={track} lead_id={lead_id}")
        return f"Successfully logged the qualified lead. The reference lead ID is {lead_id}."
    except Exception as e:
        logger.error(f"Failed to create qualified lead — contact={contact} error={e}")
        return "System error: unable to log the qualified lead at this time. Please inform the caller to try again later."


@function_tool
async def schedule_meeting(
        ctx: RunContext,
        contact: str,
        proposed_time: str
    ) -> str:
    """
    Book a Deep-Dive Assessment Meeting with a Solutions Architect.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        contact (str): The contact's unique identifier.
        proposed_time (str): The agreed-upon date or time for the meeting.

    Returns:
        str: A confirmation message indicating the meeting was successfully scheduled.
    """
    try:
        schedule_date_time = f"{proposed_time} 10:00 AM"

        logger.debug(f"Scheduled assessment meeting — contact={contact} proposed_time={proposed_time}")
        return f"Successfully scheduled the assessment meeting for {schedule_date_time}."
    except Exception as e:
        logger.error(f"Failed to schedule meeting — contact={contact} error={e}")
        return "System error: unable to schedule the meeting at this time. Please inform the caller to try again later."


@function_tool
async def send_followup_email(
        ctx: RunContext,
        contact: str,
        track: str
    ) -> str:
    """
    Send a short, non-technical follow-up email for the given track.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        contact (str): The contact's unique identifier.
        track (str): The AWS partner track discussed, to dictate the email template.

    Returns:
        str: A confirmation message indicating the email was sent.
    """
    try:
        logger.debug(f"Sent follow-up email — contact={contact} track={track}")
        return "Successfully sent the follow-up email for the discussed track."
    except Exception as e:
        logger.error(f"Failed to send follow-up email — contact={contact} error={e}")
        return "System error: unable to send the follow-up email at this time."