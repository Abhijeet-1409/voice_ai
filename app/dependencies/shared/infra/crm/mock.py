from functools import cache
from typing import Optional

from config.logger import get_logger
from config.constants import TicketPriority

from infra.crm.base import BaseCRMClient


_LOGGER = "infra.crm.mock"


class MockCRMClient(BaseCRMClient):
    """
    Mock CRM client implementation for local development and testing.

    Returns static dummy data so the AI bot can be tested without requiring
    a live connection to a real CRM provider.
    """

    def __init__(self):
        self.logger = get_logger(_LOGGER)

    async def get_customer(self, identifier: str) -> Optional[dict]:
        """
        Mock implementation of fetching a customer by identifier.
        """
        self.logger.debug(f"Mock CRM get_customer — identifier={identifier}")
        # Returning a dictionary to simulate an existing customer.
        return {
            "customer_id":    "MOCK-001",
            "name":           "Joy Sharma",
            "phone":          "9132467843",
            "email":          "joy@example.com",
            "current_plan":   "Basic",
            "plan_since":     "2023-06-01",
            "renewal_date":   "2024-06-01",
            "monthly_spend":  499,
            "usage_percent":  87,
            "open_tickets":   0
        } 

    async def create_lead(self, identifier: str, interest: str) -> str:
        """
        Mock implementation of creating a sales lead.
        """
        self.logger.debug(f"Mock CRM create_lead — identifier={identifier} interest={interest}")
        return "MOCK-LEAD-001"

    async def create_ticket(self, customer_id: str, description: str, priority: TicketPriority = TicketPriority.NORMAL) -> str:
        """
        Mock implementation of creating a support ticket.
        """
        self.logger.debug(f"Mock CRM create_ticket — customer_id={customer_id} priority={priority.value}")
        return "MOCK-TICKET-001"

    async def update_customer(self, customer_id: str, data: dict) -> None:
        """
        Mock implementation of updating a customer profile.
        """
        self.logger.debug(f"Mock CRM update_customer — customer_id={customer_id} data={data}")


@cache
def get_mockcrmclient() -> MockCRMClient:
    """
    Create and cache a thread-safe singleton instance of the MockCRMClient.

    Returns:
        MockCRMClient: The cached mock CRM client instance.
    """
    return MockCRMClient()