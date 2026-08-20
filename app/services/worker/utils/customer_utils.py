from typing import Optional

from shared.logging_setup import get_logger
from shared.config import LifecycleStage
from shared.infra.crm import get_mockcrmclient, ContactAlreadyExistsError, CRMClientError, ContactNotFoundError

from schemas.session_data import UserData


_LOGGER = "worker.utils.customer_utils"
logger = get_logger(_LOGGER)


async def lookup_customer(phone_number: str) -> Optional[dict]:
    """
    Queries the CRM to find an existing contact associated with the provided phone number.

    Deliberately non-raising: this function is designed to be safe for pre-call 
    routing or mid-call lookups. If the CRM is unreachable or throws an error, 
    it logs the failure and gracefully returns None. This ensures the voice agent 
    can still proceed with the conversation (treating the caller as a new or 
    unidentified lead) rather than crashing the active call.

    Args:
        phone_number (str): The user's phone number to search for in the CRM.

    Returns:
        Optional[dict]: A dictionary containing the contact's CRM fields if found. 
            Returns None if the contact does not exist or if the CRM request fails.
    """
    crm_client = get_mockcrmclient()

    try:
        existing = await crm_client.get_contact(phone_number)

        if existing is None:
            logger.debug(f"No existing contact found for phone_number={phone_number}")
            return None 

        logger.info(f"Existing contact retrieved successfully | phone_number={phone_number} | id={existing['id']}")
        return existing
        
    except CRMClientError as crm_err:
        logger.error(f"CRM operation failed during customer lookup | phone_number={phone_number} | error={crm_err}")
        return None


async def create_customer(userdata: UserData) -> None:
    """
    Creates a new contact from the accumulated UserData at call end, and
    applies the resulting contact_id back onto userdata in place.

    Deliberately non-raising: this runs post-call (event_handlers.py's
    on_close), where an exception could interfere with cleanup/shutdown
    logic. On any failure, this logs and returns without crashing —
    losing one contact record is preferable to crashing post-call
    handling.

    Args:
        userdata: The call's UserData, used both as the source of
            contact fields (name, email, track, qualified,
            lifecyclestage, phone) and as the target for the resulting
            customer_id.
    """
    crm_client = get_mockcrmclient()

    try:
        contact_id: str = await crm_client.create_contact(
            phone_number=userdata.phone,
            lifecyclestage=userdata.lifecyclestage,
            qualified=userdata.qualified,
            name=userdata.name,
            email=userdata.email,
            track=userdata.track,
        )
        userdata.customer_id = contact_id
        logger.info(f"Successfully created new contact post-call | phone={userdata.phone} | id={contact_id}")

    except ContactAlreadyExistsError as exist_err:
        logger.warning(
            f"Conflict: Contact already exists | phone={userdata.phone} | Re-fetching to bind ID | error={exist_err}"
        )
        
        # Reuse lookup_customer which handles its own errors and safely returns None on failure
        existing = await lookup_customer(userdata.phone)
        
        if existing is not None:
            userdata.customer_id = existing["id"]
            logger.info(f"Successfully bound existing contact ID post-call | phone={userdata.phone} | id={existing['id']}")
        else:
            # Extremely unlikely — nothing more we can do here without
            # raising, so leave userdata.customer_id unset.
            logger.error(
                f"ContactAlreadyExistsError fired for phone={userdata.phone} but "
                f"lookup_customer returned None. Leaving customer_id unset."
            )

    except CRMClientError as crm_err:
        logger.error(f"CRM client failed to create contact post-call | phone={userdata.phone} | error={crm_err}")

async def update_customer(userdata: UserData) -> None:
    """
    Updates an existing contact from the accumulated UserData at call
    end. Counterpart to create_customer — used when
    userdata.customer_id is already set (contact was found earlier in
    the call via lookup_customer), rather than newly created.
 
    Deliberately non-raising, same reasoning as create_customer: runs
    post-call (event_handlers.py's on_close), where an exception could
    interfere with cleanup/shutdown logic.
 
    Relies on MockCRMClient.update_contact's own regression guards
    (qualified only ever advances False->True, lifecyclestage only ever
    moves forward per LIFECYCLE_STAGE_ORDER) — this function does not
    duplicate that logic, it just passes through whatever this call's
    userdata currently holds.
 
    Args:
        userdata: The call's UserData. userdata.customer_id must already
            be set — this function does nothing if it's None.
    """
    if userdata.customer_id is None:
        logger.debug("update_customer called with no customer_id set — nothing to update.")
        return
 
    crm_client = get_mockcrmclient()
 
    try:
        await crm_client.update_contact(
            userdata.customer_id,
            {
                "name": userdata.name,
                "email": userdata.email,
                "track": userdata.track,
                "qualified": userdata.qualified,
                "lifecyclestage": userdata.lifecyclestage,
            },
        )
        logger.info(f"Updated contact at call end — id={userdata.customer_id}")
 
    except ContactNotFoundError as err:
        # Contact was deleted/removed between this call resolving it and
        # call end — nothing more we can do without raising.
        logger.error(
            f"update_customer: contact not found — id={userdata.customer_id} error={err}"
        )
 
    except CRMClientError as err:
        logger.error(
            f"Failed to update contact at call end — id={userdata.customer_id} error={err}"
        )

def apply_contact_to_userdata(userdata: UserData, contact: dict) -> None:
    """
    Copies a contact dictionary (as returned by CRM lookup) onto
    an existing UserData instance in place.

    Kept separate from lookup logic so callers can decide
    when/whether to apply the result — e.g. job_entrypoint.py applies it
    before UserData is even handed to the session, while
    get_customer_profile applies it to an already-live ctx.userdata
    mid-call.

    Args:
        userdata (UserData): The UserData instance to update in place.
        contact (dict): A contact dictionary containing CRM fields.
    """
    userdata.customer_id = str(contact["id"])
    userdata.phone = contact["phone_number"]
    userdata.name = contact.get("name")
    userdata.email = contact.get("email")
    userdata.track = contact.get("track")
    userdata.qualified = contact.get("qualified", False)
    userdata.lifecyclestage = contact.get("lifecyclestage", LifecycleStage.LEAD)

    logger.debug(f"Applied CRM contact data to UserData | customer_id={userdata.customer_id}")