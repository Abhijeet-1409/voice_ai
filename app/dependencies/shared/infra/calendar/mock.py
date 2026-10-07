from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from shared.logging_setup import get_logger
from shared.config import Track
from .base import BaseCalendarClient


_LOGGER = "infra.calendar.mock"


@dataclass(frozen=True)
class MeetingSlot:
    id: str
    value: str


_SLOT_OFFSETS = [
    (1, 14, 0),
    (3, 11, 0),
    (4, 16, 0),
]


def _generate_dummy_slots(now: Optional[datetime] = None) -> list[MeetingSlot]:
    """
    Builds human-readable meeting slots relative to the current date.

    Each slot contains:
    - id: stable identifier used internally for slot selection
    - value: human-readable representation shown to the caller
    """
    base = now or datetime.now()
    slots = []

    for index, (day_offset, hour, minute) in enumerate(_SLOT_OFFSETS, start=1):
        slot_dt = (
            base.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
            + timedelta(days=day_offset)
        )

        if day_offset == 1:
            label = (
                f"Tomorrow ({slot_dt.strftime('%d %b %Y')}) "
                f"{slot_dt.strftime('%-I:%M %p')} IST"
            )
        else:
            label = (
                f"{slot_dt.strftime('%A')} ({slot_dt.strftime('%d %b %Y')}) "
                f"{slot_dt.strftime('%-I:%M %p')} IST"
            )

        slots.append(
            MeetingSlot(
                id=f"slot_{index}",
                value=label,
            )
        )

    return slots


class MockCalendarClient(BaseCalendarClient):

    def __init__(self):
        self.logger = get_logger(_LOGGER)
        self.slots = _generate_dummy_slots()

    async def get_available_slots(
        self,
        track: Optional[Track] = None,
    ) -> list[MeetingSlot]:
        self.logger.debug(
            f"MockCalendar get_available_slots — track={track}"
        )
        return self.slots

    async def book_slot(
        self,
        slot: MeetingSlot,
        contact_email: str,
        track: Optional[Track] = None,
    ) -> bool:
        self.logger.info(
            f"MockCalendar book_slot — "
            f"slot_id={slot.id} "
            f"slot={slot.value} "
            f"contact_email={contact_email} "
            f"track={track}"
        )

        # Stateless: always succeeds, no persistence, no removal
        # from future get_available_slots() results.
        return True


def get_mockcalendarclient() -> MockCalendarClient:
    """
    Creates a new instance of the MockCalendarClient.

    Returns:
        MockCalendarClient: A new mock calendar client instance.
    """
    return MockCalendarClient()