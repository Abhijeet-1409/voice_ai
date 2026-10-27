from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from shared.config import Track


@dataclass(frozen=True)
class MeetingSlot:
    """An available or selected meeting slot.

    Attributes:
        id: Stable identifier used internally for selection.
        value: Human-readable time label shown to the caller.
    """

    id: str
    value: str


class BaseCalendarClient(ABC):
    """
    Abstract interface for calendar/scheduling integrations
    (dummy data today, real Cal.com API later — same interface).
    """

    @abstractmethod
    async def get_available_slots(self, track: Optional[Track] = None) -> list[MeetingSlot]:
        """
        Return a list of human-readable available meeting slots.

        Args:
            track: Optional context (e.g. which offering track) in case
                different tracks route to different calendars/SAs later.

        Returns:
            List of slot strings, e.g. ["Tomorrow 2 PM IST", ...].
            Empty list if nothing is available.
        """
        ...

    @abstractmethod
    async def book_slot(self, slot: MeetingSlot, contact_email: str, track: Optional[Track] = None) -> bool:
        """
        Book a previously offered slot.

        Args:
            slot: The selected meeting slot, containing its stable identifier and human-readable time label.
            contact_email: Confirmed email to send the meeting invite to.
            track: Optional context, same as above.

        Returns:
            True if booking succeeded, False otherwise.
        """
        ...