from functools import cache
from typing import Optional

from pydantic import BaseModel
from pydantic_core import ValidationError

from sqlalchemy import Result, select, update
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from shared.logging_setup import get_logger
from shared.config import TicketPriority, LifecycleStage, TicketStatus, Track, LIFECYCLE_STAGE_ORDER
from shared.infra.postgres import Contact, Ticket
from shared.infra.postgres.database import get_async_sessionmaker
from .base import BaseCRMClient, CRMClientError, ContactNotFoundError, ContactAlreadyExistsError


_LOGGER = "infra.crm.mock"


class ContactUpdateSchema(BaseModel):
    """
    Pydantic schema for validating contact update payloads.

    All fields are optional to support partial updates. When converting
    this schema to a dictionary, unprovided fields default to None and
    should be filtered out to avoid overwriting existing database values.

    Attributes:
        name (Optional[str]): The contact's full name.
        email (Optional[str]): The contact's email address.
        track (Optional[Track]): The specific track or segment the contact belongs to.
        qualified (Optional[bool]): Indicates whether the contact is a qualified lead.
        lifecyclestage (Optional[LifecycleStage]): The current lifecycle stage of the contact.
    """
    name: Optional[str] = None
    email: Optional[str] = None
    track: Optional[Track] = None
    qualified: Optional[bool] = None
    lifecyclestage: Optional[LifecycleStage] = None


class MockCRMClient(BaseCRMClient):

    def __init__(self):
        self.logger = get_logger(_LOGGER)
        self.async_session = get_async_sessionmaker()

    async def get_contact(self, phone_number: str) -> Optional[dict]:
        try:
            async with self.async_session() as session:
                stmt = (
                    select(Contact)
                    .where(Contact.phone_number == phone_number)
                )
                result: Result[Contact] = await session.execute(stmt)
                contact: Optional[Contact] = result.scalar_one_or_none()

                if contact is None:
                    return None

                return {
                    "id": contact.id,
                    "phone_number": contact.phone_number,
                    "name": contact.name,
                    "email": contact.email,
                    "lifecyclestage": contact.lifecyclestage,
                    "qualified": contact.qualified,
                    "track": contact.track
                }
        except SQLAlchemyError as sql_err:
            self.logger.error(f"Database error while fetching contact by phone {phone_number}: {sql_err}")
            raise CRMClientError("Failed to fetch contact due to a database error.") from sql_err

    async def create_contact(self, 
            phone_number: str,
            lifecyclestage: LifecycleStage = LifecycleStage.LEAD,
            qualified: bool = False,
            name: Optional[str] = None, 
            email: Optional[str] = None,
            track: Optional[Track] = None,
        ) -> str:
        try:
            async with self.async_session() as session:
                contact = Contact(
                    phone_number=phone_number,
                    lifecyclestage=lifecyclestage,
                    qualified=qualified,
                    name=name,
                    email=email,
                    track=track
                )
                session.add(contact)
                await session.commit()
                await session.refresh(contact)

                self.logger.info(f"Successfully created contact with ID: {contact.id}")
                return contact.id

        except IntegrityError as integ_err:
            self.logger.error(f"Integrity error: Contact with phone number {phone_number} already exists. Details: {integ_err}")
            raise ContactAlreadyExistsError(phone_number) from integ_err
        except SQLAlchemyError as sql_err:
            self.logger.error(f"Database error while creating contact for phone {phone_number}: {sql_err}")
            raise CRMClientError("Database error while creating contact with given details") from sql_err

    async def update_contact(self, contact_id: str, data: dict) -> None:
        """
        Updates an existing contact's information in the database.

        This method accepts a dictionary of data, validates it against the
        ContactUpdateSchema, and applies partial updates to the contact.
        Only the fields explicitly provided in the dictionary will be
        modified.

        Includes two regression guards, since both fields represent one-way
        progress rather than freely overwritable values:
            - `qualified` can only move from False to True, never back.
            - `lifecyclestage` can only move forward per LIFECYCLE_STAGE_ORDER
            (lead -> sales_qualified_lead -> opportunity -> customer), never
            backward. A contact already at 'customer' stays there even if a
            later, unrelated call's data would otherwise set it back to
            'lead'.

        Args:
            contact_id (str): The unique identifier of the contact to update.
            data (dict): A dictionary containing the fields to update.

        Raises:
            CRMClientError: If the provided data is empty, validation fails,
                or a database error occurs.
            ContactNotFoundError: If no contact matching the provided ID exists.
        """
        try:
            if not data:
                raise CRMClientError(f"No data provided for updating contact {contact_id}")

            contact_update = ContactUpdateSchema(**data)
            clean_data = {key: value for key, value in contact_update.model_dump().items() if value is not None}

            if not clean_data:
                raise CRMClientError(f"No valid fields provided to update for contact {contact_id}.")

            async with self.async_session() as session:
                existing = await session.get(Contact, contact_id)

                if existing is None:
                    raise ContactNotFoundError(contact_id)

                # Regression guard: never let qualified go True -> False.
                if "qualified" in clean_data and existing.qualified and not clean_data["qualified"]:
                    self.logger.debug(
                        f"Dropping qualified=False update for contact {contact_id} — "
                        f"already qualified, refusing to regress."
                    )
                    del clean_data["qualified"]

                # Regression guard: never let lifecyclestage move backward.
                if "lifecyclestage" in clean_data:
                    existing_rank = LIFECYCLE_STAGE_ORDER[existing.lifecyclestage]
                    new_rank = LIFECYCLE_STAGE_ORDER[clean_data["lifecyclestage"]]
                    if new_rank < existing_rank:
                        self.logger.debug(
                            f"Dropping lifecyclestage regression for contact {contact_id} — "
                            f"{existing.lifecyclestage} -> {clean_data['lifecyclestage']} not allowed."
                        )
                        del clean_data["lifecyclestage"]

                if not clean_data:
                    self.logger.info(
                        f"No changes to apply for contact {contact_id} after regression guards."
                    )
                    return

                stmt = (
                    update(Contact)
                    .where(Contact.id == contact_id)
                    .values(**clean_data)
                )
                await session.execute(stmt)
                await session.commit()

                self.logger.info(
                    f"Contact {contact_id} updated successfully with fields: {list(clean_data.keys())}"
                )

        except ValidationError as val_err:
            self.logger.error(f"Validation error while updating contact {contact_id}: {val_err}")
            raise CRMClientError(f"Validation error while updating contact {contact_id}") from val_err
        except SQLAlchemyError as sql_err:
            self.logger.error(f"Database error while updating contact {contact_id}: {sql_err}")
            raise CRMClientError(f"Database error while updating contact {contact_id}") from sql_err

    async def create_ticket(
        self,
        contact_id: str,
        description: str,
        priority: TicketPriority = TicketPriority.NORMAL
    ) -> str:
        try:
            async with self.async_session() as session:
                contact = await session.get(Contact, contact_id)
                if contact is None:
                    raise ContactNotFoundError(contact_id)

                ticket = Ticket(
                    contact_id=contact_id,
                    description=description,
                    priority=priority
                )

                session.add(ticket)
                await session.commit()
                await session.refresh(ticket)

                self.logger.info(f"Ticket {ticket.id} created successfully for contact {contact_id}")
                return ticket.id

        except SQLAlchemyError as sql_err:
            self.logger.error(f"Database error while creating ticket for contact {contact_id}: {sql_err}")
            raise CRMClientError(f"Failed to create ticket for contact {contact_id} due to a database error.") from sql_err

    async def get_tickets(
        self,
        contact_id: str,
        status: Optional[TicketStatus] = None,
        limit: int = 5,
    ) -> list[dict]:
        try:
            async with self.async_session() as session:
                existing = await session.get(Contact, contact_id)
                if existing is None:
                    raise ContactNotFoundError(contact_id)

                stmt = select(Ticket).where(Ticket.contact_id == contact_id)

                if status is not None:
                    stmt = stmt.where(Ticket.status == status)

                stmt = stmt.order_by(Ticket.created_at.desc()).limit(limit)

                result: Result[Ticket] = await session.execute(stmt)
                tickets = result.scalars().all()

                formatted = [
                    {
                        "id": t.id,
                        "description": t.description,
                        "priority": t.priority,
                        "status": t.status,
                    }
                    for t in tickets
                ]

                self.logger.info(
                    f"Retrieved {len(formatted)} tickets for contact {contact_id} "
                    f"(status={status.value if status else 'any'}, limit={limit})"
                )
                return formatted

        except SQLAlchemyError as sql_err:
            self.logger.error(f"Database error while retrieving tickets for contact {contact_id}: {sql_err}")
            raise CRMClientError(f"Database error retrieving tickets for contact {contact_id}") from sql_err


def get_mockcrmclient() -> MockCRMClient:
    """
    Creates a new instance of the MockCRMClient.

    Returns:
        MockCRMClient: The mock CRM client instance.
    """
    return MockCRMClient()