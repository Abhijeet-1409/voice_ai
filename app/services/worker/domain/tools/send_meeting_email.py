import asyncio

from livekit.agents import RunContext, function_tool

from shared.config import Track
from shared.logging_setup import get_logger
from utils import send_meeting_confirmation_email


_LOGGER = "worker.domain.tools.send_meeting_email"
logger = get_logger(_LOGGER)


@function_tool()
async def send_meeting_email(ctx: RunContext) -> str:
    """
    Send a confirmation email after a meeting has been successfully booked.

    The caller's email must already be confirmed, and the meeting booking
    must have populated both meeting_scheduled and meeting_slot.
    """
    userdata = ctx.userdata

    # Prevent duplicate email sends within the current session.
    if userdata.meeting_email_sent:
        return (
            "The meeting confirmation email has already been sent. "
            "Do not send it again."
        )

    if userdata.meeting_email_sending:
        return (
            "The meeting confirmation email is currently being sent. "
            "Do not start another send."
        )

    # Ensure that the caller's email has been confirmed.
    recipient_email = (userdata.email or "").strip()

    if not recipient_email:
        logger.warning(
            "Meeting email requested without a confirmed caller email"
        )
        return (
            "The caller's email address has not been confirmed. "
            "Confirm the email before sending the meeting confirmation."
        )

    # Ensure that the meeting was successfully booked.
    meeting_slot = (userdata.meeting_slot or "").strip()

    if userdata.meeting_scheduled is not True or not meeting_slot:
        logger.warning(
            "Meeting email requested before a successful booking"
        )
        return (
            "No successfully booked meeting was found. Schedule and confirm "
            "the meeting before sending a confirmation email."
        )

    userdata.meeting_email_sending = True

    try:
        # Keep the shared Track enum instead of converting it to a string.
        track: Track | None = userdata.track

        # smtplib is blocking, so run it in a separate thread.
        sent = await asyncio.to_thread(
            send_meeting_confirmation_email,
            recipient_email=recipient_email,
            customer_name=userdata.name,
            meeting_slot=meeting_slot,
            track=track,
        )

    except Exception:
        logger.exception(
            "Unexpected error while sending meeting email"
        )
        sent = False

    finally:
        userdata.meeting_email_sending = False

    if not sent:
        return (
            "The meeting was booked, but the confirmation email could not "
            "be sent. Do not tell the caller that the email was sent."
        )

    userdata.meeting_email_sent = True

    return (
        f"The meeting confirmation email was successfully sent "
        f"to {recipient_email}."
    )