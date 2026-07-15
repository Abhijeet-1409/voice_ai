from functools import cache

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker

from config.logger import get_logger
from config.settings import get_app_settings, AppBaseSettings


_LOGGER = "infra.postgres.database"


@cache
def get_async_engine() -> AsyncEngine:
    """
    Create and cache a singleton SQLAlchemy asynchronous engine.

    The engine configuration is retrieved from the application settings,
    including the database URL and query echo options.

    Returns:
        AsyncEngine: The configured asynchronous database engine instance.
    """

    settings: AppBaseSettings = get_app_settings()

    engine: AsyncEngine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DB_ECHO,
        pool_size=5,          # Number of permanent connections to keep
        max_overflow=10,      # Number of extra connections to allow during traffic spikes
        pool_pre_ping=True    # Highly recommended: checks if a connection is alive before using it
    )

    return engine


@cache
def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """
    Create and cache a singleton SQLAlchemy asynchronous session factory.

    Configures a sessionmaker bound to the cached async engine, ensuring that
    `expire_on_commit` is disabled to prevent accidental lazy-loading errors
    after a transaction commits.

    Returns:
        async_sessionmaker[AsyncSession]: A factory for generating new AsyncSession instances.
    """

    engine: AsyncEngine = get_async_engine()

    async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    return async_session


async def db_init():
    """
    Verify the database connection pool is operational on startup.

    Executes a simple 'SELECT 1' test query against the database engine.
    Logs a success message upon connection or raises a SQLAlchemyError
    if the connection fails.

    Raises:
        SQLAlchemyError: If the database is unreachable or connection fails.
    """

    engine: AsyncEngine = get_async_engine()
    logger = get_logger(_LOGGER)

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))  # Test the connection
        logger.info("Database connection established")
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def db_close():
    """
    Safely dispose of the database connection pool during application shutdown.

    This ensures all connections are gracefully closed and returned to the server,
    preventing connection leaks and noisy database error logs.
    """

    engine: AsyncEngine = get_async_engine()
    logger = get_logger(_LOGGER)

    try:
        await engine.dispose()
        logger.info("Database connection pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connection pool: {e}")