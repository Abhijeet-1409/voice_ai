from abc import ABC, abstractmethod
from typing import Optional

from config.constants import TicketPriority


class BaseCRMClient(ABC):
    """
    Abstract base class defining the interface for CRM client integrations.
    """

    @abstractmethod
    async def get_by_phone(self, phone: str) -> Optional[dict]:
        """
        Fetch a customer record by their phone number.

        Args:
            phone (str): Customer phone number in E.164 format.

        Returns:
            Optional[dict]: A dictionary containing customer data if found, or None if it is a new caller.
        """
        pass

    @abstractmethod
    async def create_lead(self, phone: str, interest: str) -> str:
        """
        Create a new lead in the CRM for a new caller.

        Args:
            phone (str): Caller phone number in E.164 format.
            interest (str): The specific plan or service the caller is interested in.

        Returns:
            str: The unique Lead ID created in the CRM.
        """
        pass

    @abstractmethod
    async def create_ticket(self, customer_id: str, description: str, priority: TicketPriority = TicketPriority.NORMAL) -> str:
        """
        Create a customer support ticket in the CRM.

        Args:
            customer_id (str): The unique Customer ID retrieved from get_by_phone.
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

        Typically used when a customer upgrades their plan or changes details during a call.

        Args:
            customer_id (str): The unique Customer ID retrieved from get_by_phone.
            data (dict): A dictionary of fields and values to update in the CRM.
        """
        pass