from typing import Optional

from shared.logging_setup import get_logger
from shared.config import Track
from shared.infra.calendar import get_mockcalendarclient, CalendarClientError


_LOGGER = "worker.utils.meeting_scheduler"
logger = get_logger(_LOGGER)


async def get_slots(track: Optional[Track] = None) -> list[str]:
    """
    Fetches available meeting slots for the given track.

    Thin orchestration over the calendar client factory — swapping to a
    real Cal.com implementation later means changing the one import
    below, not every caller (Assistant.schedule_meeting, ChooseSlotTask).

    Args:
        track: Optional context (e.g. which offering track), passed
            through to the calendar client. Unused by the current
            stateless mock implementation.

    Returns:
        List of human-readable slot strings. Empty list if none
        available or if the calendar client fails (fails soft — callers
        should treat an empty list as "offer to follow up instead",
        not crash the call).
    """
    calendar_client = get_mockcalendarclient()

    try:
        slots = await calendar_client.get_available_slots(track=track)
        logger.debug(f"Retrieved {len(slots)} available slots (track={track})")
        return slots
    except CalendarClientError as e:
        logger.error(f"Failed to fetch available slots (track={track}): {e}")
        # Fail soft: an empty list lets the calling tool degrade
        # gracefully ("no slots available, we'll follow up by email")
        # rather than crashing the whole tool call over a calendar
        # backend hiccup.
        return []


async def confirm_booking(slot: str, contact_email: str, track: Optional[Track] = None) -> bool:
    """
    Books a previously offered slot.

    Args:
        slot: One of the strings previously returned by get_slots.
        contact_email: Confirmed email to send the meeting invite to.
        track: Optional context, passed through to the calendar client.

    Returns:
        True if booking succeeded, False otherwise (either a normal
        booking failure or a calendar client error — both treated the
        same way by the caller: apologize, offer to follow up by email).
    """
    calendar_client = get_mockcalendarclient()

    try:
        success = await calendar_client.book_slot(slot, contact_email, track=track)
        if success:
            logger.info(f"Booked slot={slot} contact_email={contact_email} track={track}")
        else:
            logger.warning(f"Booking returned failure — slot={slot} contact_email={contact_email}")
        return success
    except CalendarClientError as e:
        logger.error(f"Failed to book slot={slot} contact_email={contact_email}: {e}")
        return False