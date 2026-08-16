from abc import ABC, abstractmethod


class CalendarClientError(Exception):
    """Base exception for calendar client errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BaseCalendarClient(ABC):
    """
    Abstract interface for calendar/scheduling integrations
    (dummy data today, real Cal.com API later — same interface).
    """

    @abstractmethod
    async def get_available_slots(self, track: str | None = None) -> list[str]:
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
    async def book_slot(self, slot: str, contact_email: str, track: str | None = None) -> bool:
        """
        Book a previously offered slot.

        Args:
            slot: One of the strings previously returned by get_available_slots.
            contact_email: Confirmed email to send the meeting invite to.
            track: Optional context, same as above.

        Returns:
            True if booking succeeded, False otherwise.
        """
        ...