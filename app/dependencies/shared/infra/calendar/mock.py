from typing import Optional
from functools import cache
from datetime import datetime, timedelta

from shared.logging_setup import get_logger
from shared.config import Track
from shared.infra.calendar.base import BaseCalendarClient


_LOGGER = "infra.calendar.mock"

# Offsets (days from now) and times used to generate dummy slots. Kept
# as plain data so it's easy to tweak without touching the generation
# logic. No persistence: booking a slot does NOT remove it from future
# get_available_slots() results. Deliberate simplification while the
# real Cal.com backend is deferred — see base.py docstring.
_SLOT_OFFSETS = [
    (1, 14, 0),   # tomorrow, 2:00 PM
    (3, 11, 0),   # +3 days, 11:00 AM
    (4, 16, 0),   # +4 days, 4:00 PM
]


def _generate_dummy_slots(now: Optional[datetime] = None) -> list[str]:
    """
    Builds human-readable slot strings relative to the current date,
    so dummy data stays realistic (e.g. "Tomorrow" is actually
    tomorrow) instead of a hardcoded string that goes stale.
    """
    base = now or datetime.now()
    slots = []
    for day_offset, hour, minute in _SLOT_OFFSETS:
        slot_dt = base.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=day_offset)
        if day_offset == 1:
            label = f"Tomorrow {slot_dt.strftime('%-I:%M %p')} IST"
        else:
            label = f"{slot_dt.strftime('%A')} {slot_dt.strftime('%-I:%M %p')} IST"
        slots.append(label)
    return slots


class MockCalendarClient(BaseCalendarClient):

    def __init__(self):
        self.logger = get_logger(_LOGGER)

    async def get_available_slots(self, track: Optional[Track] = None) -> list[str]:
        self.logger.debug(f"MockCalendar get_available_slots — track={track}")
        return _generate_dummy_slots()

    async def book_slot(
        self,
        slot: str,
        contact_email: str,
        track: Optional[Track] = None,
    ) -> bool:
        self.logger.info(
            f"MockCalendar book_slot — slot={slot} contact_email={contact_email} track={track}"
        )
        # Stateless: always succeeds, no persistence, no removal from
        # future get_available_slots() results.
        return True


@cache
def get_mockcalendarclient() -> MockCalendarClient:
    """
    Create and cache a thread-safe singleton instance of the MockCalendarClient.

    Returns:
        MockCalendarClient: The cached mock calendar client instance.
    """
    return MockCalendarClient()