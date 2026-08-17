from shared.infra.calendar.base import CalendarClientError
from shared.infra.calendar.mock import MockCalendarClient
from shared.logging_setup.logger import get_logger


_LOGGER = "worker.utils.meeting_scheduler"
logger = get_logger(_LOGGER)


# Swap to the real Cal.com client here once shared/infra/calendar/calcom.py
# exists, same BaseCalendarClient interface — no call-site changes needed.
_calendar_client = MockCalendarClient()


async def get_slots(track: str | None = None) -> list[str]:
    """
    Fetch available meeting slots, optionally scoped to a track
    (e.g. "aws_partner"). Returns [] on any failure — callers (schedule_meeting,
    ChooseSlotTask) treat an empty list as "no slots available" and fall back
    to an offer-to-follow-up-by-email response, never a raised exception.
    """
    try:
        return await _calendar_client.get_available_slots(track=track)
    except CalendarClientError as e:
        logger.warning("Failed to fetch available slots for track=%s: %s", track, e)
        return []


async def confirm_booking(
    slot: str,
    contact_email: str,
    track: str | None = None,
) -> bool:
    """
    Book the given slot for contact_email. Returns True/False rather than
    raising — schedule_meeting calls this AFTER ctx.disallow_interruptions()
    during the booking write, and needs a plain boolean to decide which of
    two spoken confirmations to give (booked vs. apologize-and-offer-follow-up).
    """
    try:
        return await _calendar_client.book_slot(
            slot=slot,
            contact_email=contact_email,
            track=track,
        )
    except CalendarClientError as e:
        logger.error(
            "Booking failed for slot=%s track=%s: %s", slot, track, e
        )
        return False
