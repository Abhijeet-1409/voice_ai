from enum import Enum
from typing import Literal


class TicketPriority(Enum):
    """
    Priority levels that can be assigned to a ticket, from low to high urgency.
    """

    LOW    = "low"
    NORMAL = "normal"
    HIGH   = "high"

Channel = Literal["exotel", "web"]      # source of the call (phone or browser)
CallType = Literal["inbound", "outreach"]  # direction of the call

CUSTOMER_CACHE_TTL = 900          # 15 minutes in seconds
CALL_SESSION_TTL = 21600          # 6 hours — safety TTL for Redis transcript
GEMINI_KEY_COOLDOWN_SECONDS = 3600  # 1 hour — cooldown period for API key 
WEB_TOKEN_RATE_LIMIT = 10           # max token requests per user per window
WEB_TOKEN_RATE_WINDOW_SECONDS = 60  # sliding window size in seconds