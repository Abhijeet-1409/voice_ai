from typing import Any, Optional, Sequence

from shared.logging_setup import get_logger
from shared.config import Track
from shared.infra.calendar import (
    get_mockcalendarclient,
    CalendarClientError,
)
from shared.infra.calendar.mock import MeetingSlot


_LOGGER = "worker.utils.meeting_scheduler"
logger = get_logger(_LOGGER)


def _normalize_slots(raw_slots: Sequence[Any]) -> list[MeetingSlot]:
    """Normalize provider results to the structured slot contract.

    The current mock provider returns display strings, so IDs are assigned in
    the order returned. A provider that already returns objects with ``id``
    and ``value`` keeps those IDs and values. Invalid or duplicate records
    raise rather than becoming misleading availability.
    """
    if isinstance(raw_slots, (str, bytes)):
        raise TypeError(
            "Calendar get_available_slots must return a sequence of slots, "
            "not one string"
        )

    normalized: list[MeetingSlot] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw_slots, start=1):
        if isinstance(item, str):
            slot = MeetingSlot(id=f"slot_{index}", value=item)
        else:
            slot_id = getattr(item, "id", None)
            slot_value = getattr(item, "value", None)
            if not isinstance(slot_id, str) or not isinstance(slot_value, str):
                raise TypeError(
                    "Calendar slots must be strings or objects with string "
                    f"'id' and 'value'; received {type(item).__name__} "
                    f"at index {index - 1}."
                )
            if not slot_id.strip() or not slot_value.strip():
                raise ValueError(
                    f"Calendar slot at index {index - 1} has an empty "
                    "'id' or 'value'."
                )
            slot = MeetingSlot(id=slot_id, value=slot_value)

        if slot.id in seen_ids:
            raise ValueError(
                f"Duplicate meeting slot ID returned by calendar: {slot.id!r}"
            )
        seen_ids.add(slot.id)
        normalized.append(slot)

    return normalized


async def get_slots(track: Optional[Track] = None) -> list[MeetingSlot]:
    """Fetch calendar availability and normalize it for ChooseSlotTask."""
    calendar_client = get_mockcalendarclient()

    try:
        raw_slots = await calendar_client.get_available_slots(track=track)
    except CalendarClientError as exc:
        logger.error(
            "Failed to fetch available slots (track=%s): %s", track, exc
        )
        return []

    slots = _normalize_slots(raw_slots)
    logger.debug(
        "Retrieved %d available slot(s) (track=%s): %s",
        len(slots),
        track,
        [(slot.id, slot.value) for slot in slots],
    )
    return slots


async def confirm_booking(
    slot: MeetingSlot,
    contact_email: str,
    track: Optional[Track] = None,
) -> bool:
    """Book the selected MeetingSlot through the calendar client."""
    if not isinstance(slot, MeetingSlot):
        raise TypeError(f"slot must be MeetingSlot, got {type(slot).__name__}")

    calendar_client = get_mockcalendarclient()

    try:
        # Keep the scheduler and mock calendar interfaces aligned: the mock
        # book_slot() expects a MeetingSlot object, not its display string.
        success = await calendar_client.book_slot(
            slot=slot,
            contact_email=contact_email,
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

        return bool(success)

    except CalendarClientError as exc:
        logger.error(
            "Failed to book slot_id=%s slot=%s contact_email=%s track=%s: %s",
            slot.id,
            slot.value,
            contact_email,
            track,
            exc,
        )
        return False
