import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import get_worker_settings
from shared.config import Track
from shared.logging_setup import get_logger


_LOGGER = "worker.utils.email_utils"
logger = get_logger(_LOGGER)


TRACK_DISPLAY_NAMES: dict[Track, str] = {
    Track.BILLING_TRANSFER: "AWS Billing Transfer",
    Track.GREEN_FIELD_MIGRATION: "AWS Green Field Migration",
    Track.VMWARE_WORKLOAD_MIGRATION: "AWS VMware Workload Migration",
}


def _build_meeting_email_body(
    *,
    customer_name: str | None,
    meeting_slot: str,
    track: Track | None,
    company_name: str,
) -> str:
    """Build the plain-text meeting confirmation email."""
    greeting_name = customer_name.strip() if customer_name else "there"

    lines = [
        f"Hello {greeting_name},",
        "",
        "Your meeting has been successfully scheduled.",
        "",
        "MEETING DETAILS",
        "------------------------------",
        f"Company        : {company_name}",
        f"Meeting time   : {meeting_slot}",
    ]

    if track is not None:
        track_name = TRACK_DISPLAY_NAMES.get(track, track.value)
        lines.append(f"Discussion area: {track_name}")

    lines.extend(
        [
            "",
            "We look forward to speaking with you.",
            "",
            "If you need to make any changes, please reply to this email.",
            "",
            f"Regards,\n{company_name} Team",
        ]
    )

    return "\n".join(lines)


def send_meeting_confirmation_email(
    *,
    recipient_email: str,
    customer_name: str | None,
    meeting_slot: str,
    track: Track | None = None,
) -> bool:
    """
    Send a meeting confirmation email through Gmail SMTP over SSL.

    Returns True when Gmail accepts the message for submission, otherwise
    returns False. It does not raise SMTP exceptions to the caller.
    """
    settings = get_worker_settings()
    recipient = recipient_email.strip()

    try:
        if not recipient:
            logger.warning("Cannot send meeting confirmation: recipient is empty")
            return False

        subject_name = customer_name.strip() if customer_name else "there"

        message = MIMEMultipart()
        message["From"] = settings.GMAIL_ADDRESS
        message["To"] = recipient
        message["Subject"] = f"Meeting confirmation — {subject_name}"

        message.attach(
            MIMEText(
                _build_meeting_email_body(
                    customer_name=customer_name,
                    meeting_slot=meeting_slot,
                    track=track,
                    company_name=settings.COMPANY_NAME,
                ),
                "plain",
                "utf-8",
            )
        )

        logger.info("Sending meeting confirmation email to %s", recipient)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(
                settings.GMAIL_ADDRESS,
                settings.GMAIL_APP_PASSWORD,
            )
            server.send_message(message)

        logger.info("Meeting confirmation email sent to %s", recipient)
        return True

    except Exception:
        # Do not log credentials or the complete message body.
        logger.exception(
            "Failed to send meeting confirmation email to %s",
            recipient,
        )
        return False
