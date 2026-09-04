from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker

from pgvector.asyncpg import register_vector

from shared.logging_setup import get_logger
from shared.config import get_app_settings, AppBaseSettings


_LOGGER = "infra.postgres.database"


# Private singleton instance of the asynchronous database engine. 
# Lazy-loaded to ensure it is only created when first needed, preventing 
# connection pool corruption when processes fork (e.g., during worker pre-warming).
_engine: AsyncEngine | None = None

# Private singleton instance of the database session factory.
# Bound to the _engine above and used to generate new sessions for transactions.
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_async_engine() -> AsyncEngine:
    """
    Creates and configures a SQLAlchemy asynchronous engine.

    The engine configuration is retrieved from the application settings,
    configured with pgvector support for vector embeddings,
    including the database URL and query echo options.

    Returns:
        AsyncEngine: The configured asynchronous database engine instance.
    """
    global _engine
    settings: AppBaseSettings = get_app_settings()

    if _engine:
        return _engine

    engine: AsyncEngine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DB_ECHO,
        pool_size=5,          # Number of permanent connections to keep
        max_overflow=10,      # Number of extra connections to allow during traffic spikes
        pool_pre_ping=True    # Highly recommended: checks if a connection is alive before using it
    )

    _engine = engine

    # Listen for every new database connection created by the pool
    @event.listens_for(engine.sync_engine, "connect")
    def register_custom_types(dbapi_connection, connection_record):
        # Instruct the asyncpg driver to load the pgvector codec
        dbapi_connection.run_async(lambda conn: register_vector(conn))

    return engine


def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """
    Creates a SQLAlchemy asynchronous session factory.

    Configures a sessionmaker bound to the async engine, ensuring that
    `expire_on_commit` is disabled to prevent accidental lazy-loading errors
    after a transaction commits.

    Returns:
        async_sessionmaker[AsyncSession]: A factory for generating new AsyncSession instances.
    """
    global _sessionmaker

    engine: AsyncEngine = get_async_engine()

    if engine is None:
        raise RuntimeError("Initialize database engine.....")

    if _sessionmaker:
        return _sessionmaker
    
    sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    _sessionmaker = sessionmaker

    return sessionmaker


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
    global _engine, _sessionmaker

    # Check the private variable directly instead of calling get_async_engine()
    if _engine is None:
        raise RuntimeError("Initialize database first before closing...")

    logger = get_logger(_LOGGER)

    try:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
        logger.info("Database connection pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connection pool: {e}")