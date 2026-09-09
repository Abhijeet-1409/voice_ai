from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker

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


class DatabaseNotInitializedError(RuntimeError):
    """
    Exception raised for errors in the database initialization lifecycle.

    This is triggered when a database operation (like session creation or 
    pool disposal) is attempted before the asynchronous database engine 
    has been fully initialized.
    """
    def __init__(self, message="Database engine has not been initialized. Call get_async_engine() first."):
        self.message = message
        super().__init__(self.message)


def get_async_engine() -> AsyncEngine:
    """
    Retrieves or creates a lazy-loaded SQLAlchemy asynchronous engine.

    The engine configuration is retrieved from the application settings,
    configured with pgvector support for vector embeddings. Once created,
    the engine is cached in a module-level singleton variable.

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

    return engine


def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """
    Retrieves or creates a lazy-loaded SQLAlchemy asynchronous session factory.

    Configures a sessionmaker bound to the cached async engine, ensuring that
    `expire_on_commit` is disabled to prevent accidental lazy-loading errors
    after a transaction commits. Caches the factory at the module level.

    Returns:
        async_sessionmaker[AsyncSession]: A factory for generating new AsyncSession instances.
        
    Raises:
        DatabaseNotInitializedError: If the async engine fails to initialize or is None.
    """
    global _sessionmaker

    engine: AsyncEngine = get_async_engine()

    if engine is None:
        raise DatabaseNotInitializedError("Initialize database engine.....")

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
    Verifies the database connection pool is operational on startup.

    Executes a simple 'SELECT 1' test query against the database engine
    to ensure the lazy-loaded singleton is functioning correctly. Logs a 
    success message upon connection or raises an error if it fails.

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
    Safely disposes of the database connection pool during application shutdown.

    This ensures all connections are gracefully closed and returned to the server,
    preventing connection leaks. It also resets the module-level singletons 
    (`_engine` and `_sessionmaker`) to allow safe re-initialization if needed.

    Raises:
        DatabaseNotInitializedError: If called before the database engine has been initialized.
    """
    global _engine, _sessionmaker

    # Check the private variable directly instead of calling get_async_engine()
    if _engine is None:
        raise DatabaseNotInitializedError("Initialize database first before closing...")

    logger = get_logger(_LOGGER)

    try:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
        logger.info("Database connection pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connection pool: {e}")