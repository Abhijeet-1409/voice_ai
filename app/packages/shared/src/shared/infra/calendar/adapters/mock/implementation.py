from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from shared.logging_setup import get_logger
from shared.config import Track
from ...base import BaseCalendarClient, MeetingSlot


_LOGGER = "infra.calendar.adapters.mock.implementation"
IST = ZoneInfo("Asia/Kolkata")


_SLOT_OFFSETS = [
    (1, 14, 0),
    (3, 11, 0),
    (4, 16, 0),
]


def _format_hour(dt: datetime) -> str:
    """Format the hour without a leading zero on Windows and Unix."""
    return f"{dt.strftime('%I').lstrip('0') or '12'}:{dt.strftime('%M %p')}"


def _generate_dummy_slots(now: Optional[datetime] = None) -> list[MeetingSlot]:
    """Build mock meeting slots relative to the current date in India time.

    A naive ``now`` passed by a test is interpreted as IST. An aware datetime
    is converted to IST. This keeps the displayed date and time consistent
    with the explicit ``IST`` suffix, regardless of the host machine timezone.
    """
    if now is None:
        base = datetime.now(IST)
    elif now.tzinfo is None:
        base = now.replace(tzinfo=IST)
    else:
        base = now.astimezone(IST)

    slots: list[MeetingSlot] = []

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

        date_label = slot_dt.strftime("%d %b %Y")
        time_label = _format_hour(slot_dt)
        if day_offset == 1:
            label = f"Tomorrow ({date_label}) {time_label} IST"
        else:
            label = f"{slot_dt.strftime('%A')} ({date_label}) {time_label} IST"

        slots.append(MeetingSlot(id=f"slot_{index}", value=label))

    return slots


class MockCalendarClient(BaseCalendarClient):
    """Calendar mock for development and testing only; it creates no real event."""

    def __init__(self):
        self.logger = get_logger(_LOGGER)
        self.slots = _generate_dummy_slots()

    async def get_available_slots(
        self,
        track: Optional[Track] = None,
    ) -> list[MeetingSlot]:
        self.logger.debug("MockCalendar get_available_slots — track=%s", track)
        return list(self.slots)

    async def book_slot(
        self,
        slot: MeetingSlot,
        contact_email: str,
        track: Optional[Track] = None,
    ) -> bool:
        """Simulate booking a slot; no real calendar event or invite is created."""
        if not isinstance(slot, MeetingSlot):
            self.logger.error(
                "Mock booking rejected: expected MeetingSlot, got %s",
                type(slot).__name__,
            )
            return False
        if not isinstance(contact_email, str) or not contact_email.strip():
            self.logger.error("Mock booking rejected: contact email is empty or invalid")
            return False

        self.logger.info(
            "MOCK ONLY — simulated booking accepted; no real calendar event/invite created. "
            "slot_id=%s slot=%s contact_email=%s track=%s",
            slot.id,
            slot.value,
            contact_email,
            track,
        )
        # This mock intentionally remains stateless: bookings are not persisted,
        # and slots are not removed from subsequent availability responses.
        return True


def get_mockcalendarclient() -> MockCalendarClient:
    """Create a new mock calendar client instance."""
    return MockCalendarClient()
