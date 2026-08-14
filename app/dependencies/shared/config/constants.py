from enum import StrEnum
from typing import Literal

class LifecycleStage(StrEnum):
    """
    Lifecycle stages representing a customer's progression through the sales and conversion funnel.

    Attributes:
        LEAD (str): An initial contact or inquiry that has not yet been fully vetted.
        SALES_QUALIFIED_LEAD (str): A vetted prospect deemed ready for direct sales engagement.
        OPPORTUNITY (str): An active, potential deal currently moving through the sales pipeline.
        CUSTOMER (str): A closed-won account that has successfully completed a purchase or onboarding.
    """
    LEAD = "lead"
    SALES_QUALIFIED_LEAD = "sales_qualified_lead"
    OPPORTUNITY = "opportunity"
    CUSTOMER = "customer"

class TicketPriority(StrEnum):
    """
    Priority levels indicating the urgency and required response time for a support ticket.

    Attributes:
        LOW (str): Non-critical issues, feature requests, or general inquiries.
        NORMAL (str): Standard operational issues requiring attention within normal SLAs.
        HIGH (str): Critical, time-sensitive, or blocking issues requiring immediate action.
    """
    LOW    = "low"
    NORMAL = "normal"
    HIGH   = "high"

class TicketStatus(StrEnum):
    """
    Statuses indicating a ticket's current state in the resolution workflow.

    Attributes:
        OPEN (str): The ticket is active, unresolved, and awaiting agent or system action.
        CLOSED (str): The ticket has been successfully resolved, canceled, or otherwise finalized.
    """
    OPEN = "open"
    CLOSED = "closed"

class Track(StrEnum):
    """
    Enumeration of available AWS partner qualification tracks.
    
    These tracks categorize the primary objective or migration strategy 
    assigned to a contact/lead in the sales funnel.

    Attributes:
        BILLING_TRANSFER: Represents a transition of existing AWS billing to the partner.
        GREEN_FIELD_MIGRATION: Represents building or migrating into a brand new, empty AWS environment.
        VMWARE_WORKLOAD_MIGRATION: Represents the migration of existing on-premises VMware workloads to AWS.
    """
    BILLING_TRANSFER = "billing_transfer"
    GREEN_FIELD_MIGRATION = "green_field_migration"
    VMWARE_WORKLOAD_MIGRATION = "vmware_workload_migration"

Channel = Literal["phone", "web"]      # source of the call (phone or browser)
CallType = Literal["inbound", "outreach"]  # direction of the call

CUSTOMER_CACHE_TTL = 900          # 15 minutes in seconds
CALL_SESSION_TTL = 21600          # 6 hours — safety TTL for Redis transcript
GEMINI_KEY_COOLDOWN_SECONDS = 3600  # 1 hour — cooldown period for API key 
WEB_TOKEN_RATE_LIMIT = 10           # max token requests per user per window
WEB_TOKEN_RATE_WINDOW_SECONDS = 60  # sliding window size in seconds