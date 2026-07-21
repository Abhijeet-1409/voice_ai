from abc import ABC, abstractmethod
from typing import Optional

from config.constants import TicketPriority


class BaseCRMClient(ABC):
    """
    Abstract base class defining the interface for CRM client integrations.
    """

    @abstractmethod
    async def get_customer(self, identifier: str) -> Optional[dict]:
        """
        Fetch a customer record by their unique identifier.

        Args:
            identifier (str): The unique identifier for the customer (e.g., phone number or Clerk ID).

        Returns:
            Optional[dict]: A dictionary containing customer data if found, or None if the customer does not exist.
        """
        pass

    @abstractmethod
    async def create_lead(self, identifier: str, interest: str) -> str:
        """
        Create a new lead in the CRM.

        Args:
            identifier (str): The unique identifier for the lead (e.g., phone number or Clerk ID).
            interest (str): The specific plan or service the lead is interested in.

        Returns:
            str: The unique Lead ID created in the CRM.
        """
        pass

    @abstractmethod
    async def create_ticket(self, customer_id: str, description: str, priority: TicketPriority = TicketPriority.NORMAL) -> str:
        """
        Create a customer support ticket in the CRM.

        Args:
            customer_id (str): The unique Customer ID (typically retrieved via get_customer).
            description (str): A detailed description of the customer's issue.
            priority (TicketPriority, optional): The urgency level of the ticket. Defaults to TicketPriority.NORMAL.

        Returns:
            str: The unique Ticket ID created in the CRM.
        """
        pass

    @abstractmethod
    async def update_customer(self, customer_id: str, data: dict) -> None:
        """
        Update an existing customer record in the CRM.

        Typically used when a customer upgrades their plan or changes details during an interaction.

        Args:
            customer_id (str): The unique Customer ID (typically retrieved via get_customer).
            data (dict): A dictionary of fields and values to update in the CRM.
        """
        pass