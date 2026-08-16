from abc import ABC, abstractmethod
from typing import Optional

from shared.config import TicketPriority, TicketStatus


class CRMClientError(Exception):
    """
    Base exception class for CRM client errors.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ContactNotFoundError(CRMClientError):
    """
    Raised when a contact_id passed to a CRM method does not correspond
    to any existing contact. Applies to any method that takes contact_id
    as a parameter (update_contact, create_ticket, get_tickets).
    """
    def __init__(self, contact_id: str):
        super().__init__(f"No contact found with id: {contact_id}")
        self.contact_id = contact_id


class ContactAlreadyExistsError(CRMClientError):
    """
    Raised when create_contact is called with a phone_number that
    already belongs to an existing contact. phone_number is the sole
    identifier/unique key across the CRM, so duplicates are not allowed.
    """
    def __init__(self, phone_number: str):
        super().__init__(f"A contact already exists with phone_number: {phone_number}")
        self.phone_number = phone_number


class TicketNotFoundError(CRMClientError):
    """
    Raised when a ticket_id passed to a CRM method does not correspond
    to any existing ticket.
    """
    def __init__(self, ticket_id: str):
        super().__init__(f"No ticket found with id: {ticket_id}")
        self.ticket_id = ticket_id


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
        name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> str:
        """
        Create a new contact.

        Args:
            phone_number: Normalized phone number, the sole identifier.
            name: Caller's name, if known at creation time.
            email: Caller's email, if known at creation time.

        Returns:
            The new contact's unique ID.

        Raises:
            ContactAlreadyExistsError: If phone_number is already in use.
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
    async def get_tickets(self, contact_id: str, status: Optional[TicketStatus] = None) -> list[dict]:
        """
        Fetch tickets for a contact, optionally filtered by status.

        Used by Support Flow to check a caller's existing issues (open or
        otherwise) before logging a new one.

        Args:
            contact_id: The contact's unique ID.
            status: If provided, only tickets matching this status are
                returned (e.g. TicketStatus.OPEN). If None, all tickets
                for the contact are returned regardless of status.

        Returns:
            A list of dicts, each representing one matching ticket.

        Raises:
            ContactNotFoundError: If no contact exists with contact_id.
        """
        ...