from typing import Optional

from shared.logging_setup import get_logger
from shared.config import Track
from shared.infra.calendar import (
    get_mockcalendarclient,
    CalendarClientError,
)
from shared.infra.calendar.mock import MeetingSlot


_LOGGER = "worker.utils.meeting_scheduler"
logger = get_logger(_LOGGER)


async def get_slots(track: Optional[Track] = None) -> list[MeetingSlot]:
    """
    Fetches available meeting slots for the given track.

    Thin orchestration over the calendar client factory — swapping to a
    real Cal.com implementation later means changing the one import
    below, not every caller (Assistant.schedule_meeting, ChooseSlotTask).

    Args:
        track: Optional context (e.g. which offering track), passed
            through to the calendar client.

    Returns:
        List of available MeetingSlot objects. Empty list if none
        available or if the calendar client fails.
    """
    calendar_client = get_mockcalendarclient()

    try:
        slots = await calendar_client.get_available_slots(track=track)
        logger.debug(
            "Retrieved %d available slots (track=%s)",
            len(slots),
            track,
        )
        return slots
    except CalendarClientError as e:
        logger.error(
            "Failed to fetch available slots (track=%s): %s",
            track,
            e,
        )
        return []


async def confirm_booking(
    slot: MeetingSlot,
    contact_email: str,
    track: Optional[Track] = None,
) -> bool:
    """
    Books a previously offered meeting slot.

    Args:
        slot: One of the MeetingSlot objects previously returned by get_slots.
        contact_email: Confirmed email to send the meeting invite to.
        track: Optional context, passed through to the calendar client.

    Returns:
        True if booking succeeded, False otherwise.
    """
    calendar_client = get_mockcalendarclient()

    try:
        success = await calendar_client.book_slot(
            slot,
            contact_email,
            track=track,
        )

        if success:
            logger.info(
                "Booked slot_id=%s slot=%s contact_email=%s track=%s",
                slot.id,
                slot.value,
                contact_email,
                track,
            )
        else:
            logger.warning(
                "Booking returned failure — slot_id=%s slot=%s "
                "contact_email=%s track=%s",
                slot.id,
                slot.value,
                contact_email,
                track,
            )

        return success

    except CalendarClientError as e:
        logger.error(
            "Failed to book slot_id=%s slot=%s contact_email=%s track=%s: %s",
            slot.id,
            slot.value,
            contact_email,
            track,
            e,
        )
        return False