from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from config.logging import get_logger

logger = get_logger("infra.postgres.db")

engine = create_async_engine(
    settings.DATABASE_URL, 
    echo=settings.DB_ECHO
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def db_init():
    """
    Initialize the database connection and create tables if they don't exist.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")  # Test the connection
        logger.info("Database connection established")
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        raise