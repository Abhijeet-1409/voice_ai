from functools import cache
from typing import Optional

from pydantic import BaseModel
from pydantic_core import ValidationError

from sqlalchemy import CursorResult, Result, select, update
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from shared.logging_setup import get_logger
from shared.config import TicketPriority, LifecycleStage, TicketStatus, Track
from shared.infra.postgres import Contact, Ticket
from shared.infra.postgres.database import get_async_sessionmaker
from shared.infra.crm import BaseCRMClient, CRMClientError, ContactNotFoundError, ContactAlreadyExistsError


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
        Only the fields explicitly provided in the dictionary will be modified.

        The `data` dictionary can include the following optional fields:
            - name (str): The contact's full name.
            - email (str): The contact's email address.
            - track (Track): The specific track or segment the contact belongs to.
            - qualified (bool): Indicates whether the contact is a qualified lead.
            - lifecyclestage (LifecycleStage): The current lifecycle stage of the contact.

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
                stmt = (
                    update(Contact)
                    .where(Contact.id == contact_id)
                    .values(**clean_data)
                )
                result: CursorResult = await session.execute(stmt)

                if result.rowcount == 0:
                    raise ContactNotFoundError(contact_id)

                await session.commit()
                self.logger.info(f"Contact {contact_id} updated successfully with fields: {list(clean_data.keys())}")

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

    async def get_tickets(self, contact_id: str, status: Optional[TicketStatus] = None) -> list[dict]:
        try:
            async with self.async_session() as session:
                stmt = (
                    select(Contact)
                    .where(Contact.id == contact_id)
                    .options(selectinload(Contact.tickets))
                )
                result: Result[Contact] = await session.execute(stmt)
                contact: Optional[Contact] = result.scalar_one_or_none()

                if contact is None:
                    raise ContactNotFoundError(contact_id)

                all_tickets = [
                    {
                        "id": ticket.id,
                        "description": ticket.description,
                        "priority": ticket.priority,
                        "status": ticket.status,
                    }
                    for ticket in contact.tickets
                ]

                if status is None:
                    self.logger.info(f"Retrieved {len(all_tickets)} total tickets for contact {contact_id}")
                    return all_tickets

                filtered_tickets = [ticket for ticket in all_tickets if ticket["status"] == status]
                self.logger.info(f"Retrieved {len(filtered_tickets)} {status.value} tickets for contact {contact_id}")
                return filtered_tickets

        except SQLAlchemyError as sql_err:
            self.logger.error(f"Database error while retrieving tickets for contact {contact_id}: {sql_err}")
            raise CRMClientError(f"Database error retrieving tickets for contact {contact_id}") from sql_err


@cache
def get_mockcrmclient() -> MockCRMClient:
    """
    Create and cache a thread-safe singleton instance of the MockCRMClient.

    Returns:
        MockCRMClient: The cached mock CRM client instance.
    """
    return MockCRMClient()