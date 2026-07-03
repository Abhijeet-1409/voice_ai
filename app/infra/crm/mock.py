from typing import Optional

from config.logging import get_logger
from config.constants import TicketPriority

from infra.crm.base import BaseCRMClient


logger = get_logger("infra.crm.mock")


class MockCRMClient(BaseCRMClient):
    """
    Mock CRM client for development and testing.
    Returns dummy data so the bot can be tested without a real CRM.
    Replace with real implementation when CRM is decided.
    """

    async def get_by_phone(self, phone: str) -> Optional[dict]:
        logger.debug(f"Mock CRM get_by_phone — phone={phone}")
        # return None to simulate new caller
        # return dict to simulate existing customer
        return {
            "customer_id":    "MOCK-001",
            "name":           "Joy Sharma",
            "phone":          phone,
            "email":          "joy@example.com",
            "current_plan":   "Basic",
            "plan_since":     "2023-06-01",
            "renewal_date":   "2024-06-01",
            "monthly_spend":  499,
            "usage_percent":  87,
            "open_tickets":   0
        }

    async def create_lead(self, phone: str, interest: str) -> str:
        logger.debug(f"Mock CRM create_lead — phone={phone} interest={interest}")
        return "MOCK-LEAD-001"

    async def create_ticket(self, customer_id: str, description: str, priority: TicketPriority = TicketPriority.NORMAL) -> str:
        logger.debug(f"Mock CRM create_ticket — customer_id={customer_id} priority={priority.value}")
        return "MOCK-TICKET-001"

    async def update_customer(self, customer_id: str, data: dict) -> None:
        logger.debug(f"Mock CRM update_customer — customer_id={customer_id} data={data}")


crm_client = MockCRMClient()