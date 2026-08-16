import uuid
from typing import Optional
from datetime import datetime

from sqlalchemy import String, DateTime, select, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError

from shared.logging_setup import get_logger
from shared.infra.postgres import Base
from shared.infra.postgres.database import get_async_sessionmaker


_LOGGER = "infra.postgres.users"


class User(Base):
    """
    SQLAlchemy model representing a registered application user.

    This table maps external authentication identities (like Clerk) to internal
    database records, serving as the core entity for linking call logs, billing,
    and other user-specific data.
    """

    __tablename__ = "users"

    id:             Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    clerk_user_id:  Mapped[str] = mapped_column(String, unique=True, nullable=False)

    # FIX: Added Optional[] because nullable=True
    email:          Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at:     Mapped[datetime] =  mapped_column(DateTime(timezone=True), server_default=func.now())


async def create_user(
        clerk_user_id: str,
        email: Optional[str] = None
    ) -> None:
    """
    Creates a new user record in the database.

    Uses an asynchronous database session to persist the user details. The async
    context manager automatically handles rollbacks on exceptions and safely closes
    the session upon completion.

    Args:
        clerk_user_id (str): The unique identifier provided by the Clerk authentication service.
        email (Optional[str], optional): The user's email address. Defaults to None.

    Raises:
        SQLAlchemyError: If a database constraint fails (e.g., duplicate clerk_user_id)
                         or the transaction cannot be committed.
    """

    logger = get_logger(_LOGGER)
    async_session = get_async_sessionmaker()

    try:
        async with async_session() as session:
            user = User(
                clerk_user_id=clerk_user_id,
                email=email
            )
            session.add(user)
            await session.commit()
            logger.info(f"Created user — clerk_user_id={clerk_user_id}")
    except SQLAlchemyError as e:
        logger.error(f"Failed to create user — clerk_user_id={clerk_user_id} error={e}")
        raise


async def get_user(
        clerk_user_id: str,
    ) -> Optional[User]:
    """
    Retrieves a user from the database by their Clerk user ID.

    Args:
        clerk_user_id (str): The unique identifier provided by the Clerk authentication service.

    Returns:
        Optional[User]: The corresponding User object if found, otherwise None.

    Raises:
        SQLAlchemyError: If the database query fails.
    """

    logger = get_logger(_LOGGER)
    async_session = get_async_sessionmaker()

    try:
        async with async_session() as session:
            result = await session.execute(
                        select(User).where(User.clerk_user_id == clerk_user_id)
                    )
            user = result.scalar_one_or_none()
            if user:
                logger.debug(f"Retrieved user — clerk_user_id={clerk_user_id}")
            else:
                logger.debug(f"User not found — clerk_user_id={clerk_user_id}")
            return user
    except SQLAlchemyError as e:
        logger.error(f"Failed to get user — clerk_user_id={clerk_user_id} error={e}")
        raise