from typing import Optional

from shared.logging_setup import get_logger
from shared.config import LifecycleStage
from shared.infra.crm import get_mockcrmclient, ContactAlreadyExistsError

from schemas.session_data import UserData


_LOGGER = "worker.utils.customer_lookup"
logger = get_logger(_LOGGER)


async def lookup_or_create_customer(phone_number: str, name: Optional[str] = None) -> dict:
    """
    Find-or-create orchestration: looks up a contact by phone number,
    creating one if none exists. This is the single entry point for
    resolving a caller's CRM identity — used by:
      - job_entrypoint.py (phone channel): called before the agent joins,
        since the phone number is already known from room metadata.
      - domain/tools/crm.py's get_customer_profile (web channel): called
        mid-call, once the LLM has collected a phone number from the caller.

    Uses the CRM client factory directly (get_mockcrmclient) rather than
    accepting a client parameter — swapping to real HubSpot later means
    changing this one import, not threading a client through every caller.

    Args:
        phone_number: Normalized phone number, the sole CRM lookup key.
        name: Caller's name, if already known (e.g. stated in conversation
            before the lookup happens). Only used if creating a new contact.

    Returns:
        A dict of contact fields (id, phone_number, name, email,
        lifecyclestage, track, qualified) — same shape whether the
        contact was found or newly created.

    Raises:
        CRMClientError: If the underlying CRM operation fails for reasons
            other than a duplicate phone_number (which is handled
            internally via re-fetch, not raised). Not caught here beyond
            that — callers (job_entrypoint.py, get_customer_profile) are
            responsible for deciding how to handle/report this failure,
            since the right fallback differs by context (phone channel
            setup failure vs. a mid-call tool error the LLM should relay).
    """
    crm_client = get_mockcrmclient()

    existing = await crm_client.get_contact(phone_number)
    if existing is not None:
        logger.debug(f"Existing contact found for phone_number={phone_number}: id={existing['id']}")
        return existing

    try:
        new_id = await crm_client.create_contact(phone_number, name=name)
    except ContactAlreadyExistsError:
        # Race condition: another concurrent call created this contact
        # between our get_contact check above and this create_contact
        # call. Not a real error from the caller's perspective — just
        # re-fetch the now-existing contact instead of crashing.
        logger.warning(
            f"Race condition on create_contact for phone_number={phone_number} — re-fetching."
        )
        existing = await crm_client.get_contact(phone_number)
        if existing is not None:
            return existing
        # Genuinely unexpected: duplicate error fired but the contact
        # still isn't findable. Re-raise rather than silently returning
        # something wrong.
        raise

    logger.info(f"Created new contact for phone_number={phone_number}: id={new_id}")

    return {
        "id": new_id,
        "phone_number": phone_number,
        "name": name,
        "email": None,
        "lifecyclestage": LifecycleStage.LEAD,
        "track": None,
        "qualified": False,
    }


def apply_contact_to_userdata(userdata: UserData, contact: dict) -> None:
    """
    Copies a contact dict (as returned by lookup_or_create_customer) onto
    an existing UserData instance in place.

    Kept separate from lookup_or_create_customer so callers can decide
    when/whether to apply the result — e.g. job_entrypoint.py applies it
    before UserData is even handed to the session, while
    get_customer_profile applies it to an already-live ctx.userdata
    mid-call.

    Args:
        userdata: The UserData instance to update in place.
        contact: A contact dict as returned by lookup_or_create_customer.
    """
    userdata.customer_id = str(contact["id"])
    userdata.phone = contact["phone_number"]
    userdata.name = contact.get("name")
    userdata.email = contact.get("email")
    userdata.track = contact.get("track")
    userdata.qualified = contact.get("qualified", False)
    userdata.lifecyclestage = contact.get("lifecyclestage", LifecycleStage.LEAD)