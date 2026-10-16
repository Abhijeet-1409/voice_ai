"""Database connection management and lifecycle module.

Provides thread-safe and process-safe initialization, accessors, and teardown
handlers for both synchronous and asynchronous SQLAlchemy engines and session factories.
"""

from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker 
from sqlalchemy import text, create_engine, Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker

from shared.logging_setup import get_logger
from shared.config import get_app_settings, AppBaseSettings


_LOGGER: str = "infra.postgres.database"


# ------------------------------------------------------------------------------
# Module-level Global State (Lazy-loaded Singletons)
# ------------------------------------------------------------------------------

_async_engine: Optional[AsyncEngine] = None
"""Optional[AsyncEngine]: Cached global singleton instance for the asynchronous database engine.
Lazy-loaded to defer connection pool initialization until explicitly requested or initialized.
"""

_async_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
"""Optional[async_sessionmaker[AsyncSession]]: Cached global singleton factory for generating
AsyncSession instances bound to `_async_engine`.
"""

_sync_engine: Optional[Engine] = None
"""Optional[Engine]: Cached global singleton instance for the synchronous database engine.
Lazy-loaded to prevent pool duplication across process forks (e.g., Gunicorn workers).
"""

_sync_sessionmaker: Optional[sessionmaker] = None
"""Optional[sessionmaker]: Cached global singleton factory for generating synchronous
Session instances bound to `_sync_engine`.
"""


# ------------------------------------------------------------------------------
# Custom Lifecycle Exceptions
# ------------------------------------------------------------------------------

class DatabaseNotInitializedError(RuntimeError):
    """Exception raised when accessing or tearing down unitialized database components.

    Triggered when database operations (such as session creation, engine access,
    or connection pool disposal) are invoked before the underlying engine singleton
    has been initialized via `init_sync_engine()` or `init_async_engine()`.

    Args:
        message: Explanation of the lifecycle initialization failure.
            Defaults to requesting a call to `sync_db_init()` or `async_db_init()`.
    """

    def __init__(
        self,
        message: str = "Database engine has not been initialized. Call sync_db_init() or async_db_init() first."
    ) -> None:
        self.message = message
        super().__init__(self.message)


# ------------------------------------------------------------------------------
# Synchronous Engine & Session Lifecycle
# ------------------------------------------------------------------------------

def init_sync_engine() -> Engine:
    """Retrieves or initializes the lazy-loaded synchronous SQLAlchemy engine.

    Retrieves database settings from application configuration, configures the
    connection pool, and caches the resulting engine in the module-level singleton `_sync_engine`.

    Returns:
        Engine: The initialized synchronous SQLAlchemy engine instance.
    """
    global _sync_engine
    settings: AppBaseSettings = get_app_settings()

    if _sync_engine:
        return _sync_engine

    engine: Engine = create_engine(
        settings.DATABASE_URL.replace("+asyncpg", "+psycopg2"),  # Remove async driver for sync engine
        echo=settings.DB_ECHO,
        pool_size=5,          # Permanent connections kept open in pool
        max_overflow=10,      # Overflow connections permitted during traffic spikes
        pool_pre_ping=True    # Validates connection vitality prior to checkout
    )

    _sync_engine = engine

    return engine


def get_sync_engine() -> Engine:
    """Retrieves the cached synchronous SQLAlchemy engine instance.

    Returns:
        Engine: The cached synchronous database engine instance.

    Raises:
        DatabaseNotInitializedError: If the synchronous engine has not been initialized yet.
    """
    global _sync_engine

    if _sync_engine is None:
        raise DatabaseNotInitializedError("Initialize synchronous database first before accessing the engine.")

    return _sync_engine


def get_sync_sessionmaker() -> sessionmaker:
    """Retrieves or creates a cached synchronous SQLAlchemy session factory.

    Configures a `sessionmaker` bound to the cached synchronous engine with
    `expire_on_commit=False` to prevent detach/lazy-loading errors after commits.

    Returns:
        sessionmaker: The module-level session factory instance for creating synchronous sessions.

    Raises:
        DatabaseNotInitializedError: If the synchronous engine fails to initialize or is uninitialized.
    """
    global _sync_sessionmaker

    engine: Engine = get_sync_engine()

    if _sync_sessionmaker:
        return _sync_sessionmaker
    
    session_factory: sessionmaker = sessionmaker(
        bind=engine,
        expire_on_commit=False
    )

    _sync_sessionmaker = session_factory

    return session_factory


def sync_db_init() -> None:
    """Verifies that the synchronous database connection pool is operational on startup.

    Initializes the synchronous engine and performs a readiness ping (`SELECT 1`).
    Logs connection status and raises exceptions on connectivity failures.

    Raises:
        SQLAlchemyError: If the database server is unreachable or connection fails.
    """
    engine: Engine = init_sync_engine()
    logger = get_logger(_LOGGER)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # Test the connection
        logger.info("Synchronous database connection established")
    except SQLAlchemyError as e:
        logger.error(f"Synchronous database initialization failed: {e}")
        raise


def sync_db_close() -> None:
    """Safely disposes of the synchronous database connection pool during application shutdown.

    Closes all active pooled connections gracefully and resets the module-level singletons
    (`_sync_engine` and `_sync_sessionmaker`) to `None`.

    Raises:
        DatabaseNotInitializedError: If called before the synchronous database engine has been initialized.
    """
    global _sync_engine, _sync_sessionmaker

    # Check the private variable directly instead of calling get_sync_engine()
    if _sync_engine is None:
        raise DatabaseNotInitializedError("Initialize synchronous database first before closing...")

    logger = get_logger(_LOGGER)

    try:
        _sync_engine.dispose()
        logger.info("Synchronous database connection pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing synchronous database connection pool: {e}")
    finally:
        _sync_engine = None
        _sync_sessionmaker = None


# ------------------------------------------------------------------------------
# Asynchronous Engine & Session Lifecycle
# ------------------------------------------------------------------------------

def init_async_engine() -> AsyncEngine:
    """Retrieves or initializes the lazy-loaded asynchronous SQLAlchemy engine.

    Retrieves database settings from application configuration, configures the
    async connection pool, and caches the resulting engine in the module-level singleton `_async_engine`.

    Returns:
        AsyncEngine: The initialized asynchronous SQLAlchemy engine instance.
    """
    global _async_engine
    settings: AppBaseSettings = get_app_settings()

    if _async_engine:
        return _async_engine

    engine: AsyncEngine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DB_ECHO,
        pool_size=5,          # Permanent connections kept open in pool
        max_overflow=10,      # Overflow connections permitted during traffic spikes
        pool_pre_ping=True    # Validates connection vitality prior to checkout
    )

    _async_engine = engine

    return engine


def get_async_engine() -> AsyncEngine:
    """Retrieves the cached asynchronous SQLAlchemy engine instance.

    Returns:
        AsyncEngine: The cached asynchronous database engine instance.

    Raises:
        DatabaseNotInitializedError: If the asynchronous engine has not been initialized yet.
    """
    global _async_engine

    if _async_engine is None:
        raise DatabaseNotInitializedError("Initialize asynchronous database first before accessing the engine.")

    return _async_engine


def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Retrieves or creates a cached asynchronous SQLAlchemy session factory.

    Configures an `async_sessionmaker` bound to the cached asynchronous engine with
    `expire_on_commit=False` to prevent detach errors after transaction commits.

    Returns:
        async_sessionmaker[AsyncSession]: The module-level factory for creating AsyncSession instances.

    Raises:
        DatabaseNotInitializedError: If the async engine fails to initialize or is uninitialized.
    """
    global _async_sessionmaker

    engine: AsyncEngine = get_async_engine()

    if _async_sessionmaker:
        return _async_sessionmaker
    
    async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    _async_sessionmaker = async_session_factory

    return async_session_factory


async def async_db_init() -> None:
    """Verifies that the asynchronous database connection pool is operational on startup.

    Initializes the asynchronous engine and performs a readiness ping (`SELECT 1`).
    Logs connection status and raises exceptions on connectivity failures.

    Raises:
        SQLAlchemyError: If the database server is unreachable or connection fails.
    """
    engine: AsyncEngine = init_async_engine()
    logger = get_logger(_LOGGER)

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))  # Test the connection
        logger.info("Database connection established")
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def async_db_close() -> None:
    """Safely disposes of the asynchronous database connection pool during application shutdown.

    Closes all active pooled connections gracefully and resets the module-level singletons
    (`_async_engine` and `_async_sessionmaker`) to `None`.

    Raises:
        DatabaseNotInitializedError: If called before the asynchronous database engine has been initialized.
    """
    global _async_engine, _async_sessionmaker

    # Check the private variable directly instead of calling get_async_engine()
    if _async_engine is None:
        raise DatabaseNotInitializedError("Initialize database first before closing...")

    logger = get_logger(_LOGGER)

    try:
        await _async_engine.dispose()
        logger.info("Database connection pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connection pool: {e}")
    finally:
        _async_engine = None
        _async_sessionmaker = None