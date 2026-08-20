from typing import Optional

from livekit.agents import ToolError, function_tool, RunContext

from shared.logging_setup import get_logger
from shared.config import TicketPriority, TicketStatus
from shared.infra.crm import get_mockcrmclient, CRMClientError, ContactNotFoundError

from schemas import UserData


_LOGGER = "worker.domain.tools.crm"
logger = get_logger(_LOGGER)


@function_tool
async def create_ticket(
    ctx: RunContext,
    description: str,
    priority: TicketPriority = TicketPriority.NORMAL,
) -> str:
    """
    Creates a new customer support ticket in the CRM for the current
    caller. Assumes ctx.userdata.customer_id is already set — on the
    current SIP-only phone channel, identity is always resolved by
    job_entrypoint.py before the agent joins, so no separate check is
    needed here.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        description (str): A detailed description of the customer's
            issue or request.
        priority (TicketPriority, optional): Urgency level. Defaults to
            NORMAL — only use HIGH for outages or urgent billing issues.

    Returns:
        str: A confirmation message including the unique Ticket ID.
    """
    userdata: UserData = ctx.userdata

    try:
        crm_client = get_mockcrmclient()
        ticket_id = await crm_client.create_ticket(
            contact_id=userdata.customer_id, 
            description=description, 
            priority=priority
        )

        logger.info(
            f"Support ticket created successfully | customer_id={userdata.customer_id} | "
            f"ticket_id={ticket_id} | priority={priority.value}"
        )
        return f"Successfully created support ticket with ID {ticket_id} at {priority.value} priority."

    except ContactNotFoundError as not_found_err:
        logger.error(
            f"Failed to create ticket: Contact not found in CRM | "
            f"customer_id={userdata.customer_id} | error={not_found_err}"
        )
        raise ToolError(
            "The caller's account couldn't be found. Let them know a "
            "specialist will follow up to resolve this."
        ) from not_found_err

    except CRMClientError as crm_err:
        logger.error(
            f"CRM operation failed during ticket creation | "
            f"customer_id={userdata.customer_id} | error={crm_err}"
        )
        raise ToolError(
            "Unable to create a support ticket right now. Let the caller "
            "know a specialist will follow up shortly."
        ) from crm_err


@function_tool
async def get_tickets(ctx: RunContext, status: Optional[TicketStatus] = None) -> str:
    """
    Retrieves the caller's most recent support tickets (up to 5),
    optionally filtered by status. Use this when the caller asks about
    an existing ticket — e.g. whether a previous issue was resolved,
    or what's currently open.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        status (TicketStatus, optional): Filter to OPEN or CLOSED only.
            Omit to return tickets regardless of status.

    Returns:
        str: A natural-language summary of the caller's tickets.
    """
    try:
        crm_client = get_mockcrmclient()
        tickets = await crm_client.get_tickets(ctx.userdata.customer_id, status=status)

        if not tickets:
            return "No tickets found for this caller."

        lines = [
            f"- {t['description']} (priority: {t['priority'].value}, status: {t['status'].value})"
            for t in tickets
        ]
        summary = "\n".join(lines)

        return (
            f"Here are the most recent tickets:\n{summary}\n"
            f"For older tickets, let the caller know they can check the customer portal."
        )

    except ContactNotFoundError as not_found_err:
        raise ToolError(
            "The caller's account couldn't be found. Let them know a "
            "specialist will follow up to resolve this."
        ) from not_found_err

    except CRMClientError as crm_err:
        raise ToolError(
            "Unable to retrieve ticket information right now. Let the "
            "caller know a specialist will follow up shortly."
        ) from crm_err
