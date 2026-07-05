from livekit.agents import function_tool, RunContext

from infra.crm.mock import crm_client
from infra.redis.customer_cache import get_cached_customer, cache_customer

from config.constants import TicketPriority
from config.logging import get_logger


logger = get_logger("domain.tools.crm")


def _format_customer(data: dict) -> str:
    """Format raw CRM dict into voice-friendly prose."""
    return (
        f"Customer {data['name']} is on the {data['current_plan']} plan "
        f"since {data['since']} with {data['usage_percent']}% usage. "
        f"They have {data['open_tickets']} open tickets."
    )


@function_tool
async def get_customer_profile(ctx: RunContext, phone: str) -> str:
    """
    Fetch customer profile from CRM using the caller's phone number in E.164
    format. Call this immediately at the very start of every call before
    saying anything else to the caller.
    """
    logger.debug(f"get_customer_profile called — phone={phone}")
    try:
        cached = await get_cached_customer(phone)
        if cached:
            logger.debug(f"Cache hit for phone={phone}")
            return _format_customer(cached)

        data = await crm_client.get_by_phone(phone)
        if data is None:
            return "This is a new caller with no existing account on file."

        await cache_customer(phone, data)
        return _format_customer(data)
    except Exception as e:
        logger.error(f"get_customer_profile failed — phone={phone} error={e}")
        return "I was unable to retrieve the customer profile at this time."


@function_tool
async def create_lead(ctx: RunContext, phone: str, interest: str) -> str:
    """
    Create a new lead in the CRM when a new caller expresses interest in
    Intelics Cloud Services. Call this before ending the call with any
    new caller who has shown interest in our products or plans.
    """
    logger.debug(f"create_lead called — phone={phone} interest={interest}")
    try:
        lead_id = await crm_client.create_lead(phone, interest)
        return f"I have logged your interest. Our sales team will follow up with you shortly. Your reference is {lead_id}."
    except Exception as e:
        logger.error(f"create_lead failed — phone={phone} error={e}")
        return "I was unable to log your interest at this time. Our team will still follow up with you."


@function_tool
async def create_ticket(
    ctx: RunContext,
    customer_id: str,
    description: str,
    priority: str = "normal"
) -> str:
    """
    Create a support ticket in the CRM when a caller reports a problem,
    raises a complaint, or requests follow-up from the technical team.
    Call this after collecting enough detail about the issue from the caller.
    Priority must be one of: low, normal, high.
    """
    logger.debug(f"create_ticket called — customer_id={customer_id} priority={priority}")
    try:
        priority_map = {
            "low":    TicketPriority.LOW,
            "normal": TicketPriority.NORMAL,
            "high":   TicketPriority.HIGH,
        }
        ticket_priority = priority_map.get(priority.lower(), TicketPriority.NORMAL)
        ticket_id = await crm_client.create_ticket(customer_id, description, ticket_priority)
        return f"I have created support ticket {ticket_id} for you. Our team will get back to you within 24 hours."
    except Exception as e:
        logger.error(f"create_ticket failed — customer_id={customer_id} error={e}")
        return "I was unable to create a ticket at this time. Please call back and we will assist you."


@function_tool
async def update_customer(
    ctx: RunContext,
    customer_id: str,
    data: dict
) -> str:
    """
    Update a customer record in the CRM. Call this when a customer agrees
    to upgrade their plan, changes their contact details, or when any
    account information needs to be updated during the call.
    """
    logger.debug(f"update_customer called — customer_id={customer_id} data={data}")
    try:
        await crm_client.update_customer(customer_id, data)
        return "I have updated your account details successfully."
    except Exception as e:
        logger.error(f"update_customer failed — customer_id={customer_id} error={e}")
        return "I was unable to update your account at this time. Please try again later."