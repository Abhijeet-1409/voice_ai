import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config.settings import settings


def send_email_notification(session: dict) -> bool:
    """
    Send call summary email to sales team after call ends.
    Returns True if sent successfully, False if failed.
    Never raises an exception — email failure should not crash the app.
    """
    try:
        # ── Build email content ────────────────────────────────────────────
        subject = f"New Intelics Call — {session.get('caller_name') or 'Unknown Caller'}"

        body = _build_email_body(session)

        # ── Compose message ────────────────────────────────────────────────
        msg = MIMEMultipart()
        msg["From"]    = settings.gmail_address
        msg["To"]      = settings.notification_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # ── Send via Gmail SMTP SSL ────────────────────────────────────────
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(settings.gmail_address, settings.gmail_app_password)
            server.sendmail(
                settings.gmail_address,
                settings.notification_email,
                msg.as_string(),
            )

        return True

    except Exception as e:
        print(f"[email_utils] Failed to send email: {e}")
        return False


# ── Email body builder ────────────────────────────────────────────────────────

def _build_email_body(session: dict) -> str:
    lines = []

    lines.append("=" * 60)
    lines.append("INTELICS VOICE AI — CALL SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # ── Caller info ────────────────────────────────────────────────────────
    lines.append("CALLER DETAILS")
    lines.append("-" * 40)
    lines.append(f"Session ID     : {session.get('session_id', 'N/A')}")
    lines.append(f"Name           : {session.get('caller_name') or 'Not captured'}")
    lines.append(f"Phone          : {session.get('caller_phone') or 'Not captured'}")
    lines.append(f"Email          : {session.get('caller_email') or 'Not captured'}")
    lines.append(f"Need           : {session.get('caller_need') or 'Not captured'}")
    lines.append(f"Interest Level : {session.get('interest_level') or 'Not assessed'}")
    lines.append("")

    # ── Call stats ─────────────────────────────────────────────────────────
    lines.append("CALL STATS")
    lines.append("-" * 40)
    lines.append(f"Start Time     : {session.get('start_time', 'N/A')}")
    lines.append(f"End Time       : {session.get('end_time', 'N/A')}")
    lines.append(f"Exchange Count : {session.get('exchange_count', 0)}")
    lines.append("")

    # ── Full conversation ──────────────────────────────────────────────────
    lines.append("FULL CONVERSATION")
    lines.append("-" * 40)

    exchanges = session.get("exchanges", [])
    if not exchanges:
        lines.append("No exchanges recorded.")
    else:
        for ex in exchanges:
            lines.append(f"[{ex.get('timestamp', '')}]")
            lines.append(f"Customer : {ex.get('caller_message', '')}")
            lines.append(f"Agent    : {ex.get('agent_reply', '')}")
            lines.append("")

    lines.append("=" * 60)
    lines.append("End of call summary")
    lines.append("=" * 60)

    return "\n".join(lines)