import uuid
from typing import Optional
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.config import LifecycleStage, Track
from shared.infra.postgres import Base


class Contact(Base):
    """
    SQLAlchemy model representing a customer contact record and its associated sales metadata.

    Attributes:
        id (str): Primary key, uniquely identifying the contact record as a UUID string.
        phone_number (str): Unique, indexed phone number used for contact identification and call routing.
        name (str | None): Full name of the contact, if available.
        email (str | None): Primary email address of the contact, if available.
        lifecyclestage (LifecycleStage): Current stage of the contact in the sales and conversion
            funnel (defaults to LEAD).
        track (str | None): Assigned AWS partner qualification track (e.g., Billing Transfer,
            Green Field Migration, VMware Workload Migration).
        qualified (bool): Flag indicating whether the lead/contact has met qualification criteria.
        created_at (datetime): UTC timestamp recording when the contact record was created.
        updated_at (datetime): UTC timestamp recording when the contact record was last modified.
        tickets (list[Ticket]): One-to-many relationship linking this contact to their associated support tickets.
    """

    __tablename__ = "contacts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    phone_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    lifecyclestage: Mapped[LifecycleStage] = mapped_column(String,
        default=LifecycleStage.LEAD, nullable=False
    )
    track: Mapped[Optional[Track]] = mapped_column(String, default=None, nullable=True)
    qualified: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    tickets: Mapped[list["Ticket"]] = relationship(back_populates="contact")  # type: ignore