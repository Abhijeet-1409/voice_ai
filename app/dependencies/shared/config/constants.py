from enum import Enum


class TicketPriority(Enum):
    """
    Priority levels that can be assigned to a ticket, from low to high urgency.
    """

    LOW    = "low"
    NORMAL = "normal"
    HIGH   = "high"


CUSTOMER_CACHE_TTL = 900          # 15 minutes in seconds
CALL_SESSION_TTL = 21600          # 6 hours — safety TTL for Redis transcript
GEMINI_KEY_COOLDOWN_SECONDS = 3600  # 1 hour — cooldown period for API key usage