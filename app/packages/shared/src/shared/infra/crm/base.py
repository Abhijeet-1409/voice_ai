from typing import Optional
from abc import ABC, abstractmethod

from shared.config import TicketPriority, TicketStatus, LifecycleStage, Track


class BaseCRMClient(ABC):
    """
    Abstract interface for CRM integrations (mock Postgres today, real
    HubSpot API later — same interface, swappable backend).

    `phone_number` is the sole identifier used to look up or create a
    contact — the single lookup key used for both phone and web
    channels. Once a contact exists, `contact_id` is used to reference
    it in every other method.

    Implementations MUST raise ContactNotFoundError from any method
    that takes contact_id if no matching contact exists, and MUST
    raise ContactAlreadyExistsError from create_contact if phone_number
    is already in use.
    """

    @abstractmethod
    async def get_contact(self, phone_number: str) -> Optional[dict]:
        """
        Fetch a contact record by phone number.

        Args:
            phone_number: Normalized phone number (E.164), the sole
                lookup key across both channels.

        Returns:
            A dict of contact fields if found, else None. Unlike the
            contact_id-based methods below, a missing contact here is a
            normal, expected outcome — not an error — since this is
            exactly how callers check "does this contact exist yet."
        """
        ...

    @abstractmethod
    async def create_contact(
        self,
        phone_number: str,
        lifecyclestage: LifecycleStage = LifecycleStage.LEAD,
        qualified: bool = False,
        name: Optional[str] = None,
        email: Optional[str] = None,
        track: Optional[Track] = None,
    ) -> str:
        """
        Create a new contact in the CRM.

        Args:
            phone_number: Normalized phone number, the sole primary identifier.
            lifecyclestage: Initial lifecycle stage of the contact.
            qualified: Whether the lead has passed initial qualification criteria. Defaults to False.
            name: Caller's name, if known at creation time.
            email: Caller's email, if known at creation time.
            track: Specific focus track (e.g., AWS partner track), if assigned at creation time.

        Returns:
            The new contact's unique ID.

        Raises:
            ContactAlreadyExistsError: If phone_number is already in use.
            CRMClientError: If the creation fails due to underlying storage errors.
        """
        ...

    @abstractmethod
    async def update_contact(self, contact_id: str, data: dict) -> None:
        """
        Update fields on an existing contact — e.g. name, email, track,
        qualified, lifecyclestage.

        Args:
            contact_id: The contact's unique ID (from get_contact/create_contact).
            data: Field-value pairs to update.

        Raises:
            ContactNotFoundError: If no contact exists with contact_id.
        """
        ...

    @abstractmethod
    async def create_ticket(
        self,
        contact_id: str,
        description: str,
        priority: TicketPriority = TicketPriority.NORMAL,
    ) -> str:
        """
        Create a support ticket linked to a contact.

        Args:
            contact_id: The contact's unique ID.
            description: Description of the issue.
            priority: Urgency level, defaults to NORMAL.

        Returns:
            The new ticket's unique ID.

        Raises:
            ContactNotFoundError: If no contact exists with contact_id.
        """
        ...

    @abstractmethod
    async def get_tickets(
        self,
        contact_id: str,
        status: Optional[TicketStatus] = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Fetch a contact's most recent tickets, optionally filtered by
        status. Results are ordered most-recent-first, so the first
        item is the contact's "last ticket".

        Used by Support Flow to check a caller's existing issues (open
        or otherwise) before logging a new one, or when the caller asks
        about a previous ticket's status.

        Args:
            contact_id: The contact's unique ID.
            status: If provided, only tickets matching this status are
                returned (e.g. TicketStatus.OPEN). If None, all tickets
                for the contact are returned regardless of status.
            limit: Maximum number of tickets to return, most recent
                first. Defaults to 5 — deliberately capped so callers
                (voice agent tools) aren't overwhelmed with a caller's
                full ticket history; older tickets are expected to be
                checked via a separate portal, not read aloud.

        Returns:
            A list of dicts, each representing one matching ticket,
            ordered most-recent-first, at most `limit` items.

        Raises:
            ContactNotFoundError: If no contact exists with contact_id.
        """
        ...