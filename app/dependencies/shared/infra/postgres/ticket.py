import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.config.constants import TicketPriority, TicketStatus

from shared.infra.postgres.database import Base


class Ticket(Base):
    """
    SQLAlchemy model representing a customer support ticket or inquiry tied to a specific contact.

    Attributes:
        id (str): Primary key, uniquely identifying the ticket record as a UUID string.
        contact_id (str): Foreign key linking the ticket to its associated contact record in the 'contacts' table.
        description (str): Detailed text description of the issue, request, or inquiry.
        priority (TicketPriority): Assigned priority level indicating urgency (defaults to NORMAL).
        status (TicketStatus): Current operational state of the ticket in the workflow (defaults to OPEN).
        created_at (datetime): UTC timestamp recording when the ticket was created.
        contact (Contact): Many-to-one relationship mapping back to the parent Contact entity.
    """

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    contact_id: Mapped[str] = mapped_column(ForeignKey("contacts.id"), nullable=False)

    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[TicketPriority] = mapped_column(default=TicketPriority.NORMAL)
    status: Mapped[TicketStatus] = mapped_column(default=TicketStatus.OPEN)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    contact: Mapped["Contact"] = relationship(back_populates="tickets")  # type: ignore