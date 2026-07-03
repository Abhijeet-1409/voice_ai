from abc import ABC, abstractmethod
from typing import Optional

from config.constants import TicketPriority


class BaseCRMClient(ABC):
    """
    Abstract base class for CRM clients.
    """

    @abstractmethod
    async def get_by_phone(self, phone: str) -> Optional[dict]:
        """
        Fetch customer record by phone number.

        :param phone: Customer phone number in E.164 format.
        :return: Customer dict if found, None if new caller.
        """
        pass

    @abstractmethod
    async def create_lead(self, phone: str, interest: str) -> str:
        """
        Create a new lead in the CRM for a new caller.

        :param phone: Caller phone number in E.164 format.
        :param interest: What plan or service they are interested in.
        :return: Lead ID created in CRM.
        """
        pass

    @abstractmethod
    async def create_ticket(self, customer_id: str, description: str, priority: TicketPriority = TicketPriority.NORMAL) -> str:
        """
        Create a support ticket in the CRM.

        :param customer_id: Customer ID from get_by_phone.
        :param description: Description of the issue.
        :param priority: Ticket priority — low, normal, high.
        :return: Ticket ID created in CRM.
        """
        pass

    @abstractmethod
    async def update_customer(self, customer_id: str, data: dict) -> None:
        """
        Update customer record in the CRM.
        Used when customer upgrades plan during call.

        :param customer_id: Customer ID from get_by_phone.
        :param data: Fields to update.
        :return: None
        """
        pass