import redis as sync_redis
import redis.asyncio as async_redis
from redis.exceptions import RedisError

from shared.config import get_app_settings, AppBaseSettings
from shared.logging_setup import get_logger


_LOGGER = "infra.redis.client"


# Module-level singleton for the synchronous Redis client.
# Initialized lazily to avoid creating connections until the client is needed.
_sync_redis_client: sync_redis.Redis | None = None

# Module-level singleton for the asynchronous Redis client.
# Initialized lazily to avoid creating connections until the client is needed.
_async_redis_client: async_redis.Redis | None = None


class RedisNotInitializedError(RuntimeError):
    """Raised when a Redis client lifecycle operation requires prior initialization."""

    def __init__(
        self,
        message: str = "Redis client has not been initialized.",
    ) -> None:
        super().__init__(message)


def get_sync_redis_client() -> sync_redis.Redis:
    """Return the lazily initialized synchronous Redis client.

    The client uses application settings and a connection pool. Repeated calls
    return the same client instance.

    Returns:
        The configured synchronous Redis client.
    """
    global _sync_redis_client

    if _sync_redis_client is not None:
        return _sync_redis_client

    settings: AppBaseSettings = get_app_settings()

    # Creating the client configures its connection pool; it does not
    # necessarily establish a network connection immediately.
    client = sync_redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,       # Decode Redis responses as strings.
        health_check_interval=30,    # Check connection health after idle periods.
        socket_timeout=3.0,          # Limit individual socket operations.
        retry_on_timeout=True,       # Enable retries for timeout errors.
        max_connections=50,          # Maximum connections in the pool.
    )

    _sync_redis_client = client
    return client


def ping_redis_sync() -> bool:
    """Check whether the synchronous Redis server responds to PING.

    Returns:
        True if the ping succeeds; otherwise, False.
    """
    client = get_sync_redis_client()
    logger = get_logger(_LOGGER)

    try:
        client.ping()
        logger.debug("Successfully pinged synchronous Redis server.")
        return True
    except RedisError as err:
        logger.error(f"Failed to ping synchronous Redis server: {err}")
        return False


def close_redis_sync() -> None:
    """Close the synchronous Redis client during application shutdown.

    Resets the module-level singleton after closing so that it can be
    initialized again if required.

    Raises:
        RedisNotInitializedError: If the client has not been initialized.
        Exception: Re-raises an unexpected error encountered during cleanup.
    """
    global _sync_redis_client

    if _sync_redis_client is None:
        raise RedisNotInitializedError(
            "Synchronous Redis client must be initialized before closing."
        )

    logger = get_logger(_LOGGER)

    try:
        _sync_redis_client.close()
        _sync_redis_client = None
        logger.info("Synchronous Redis client closed successfully.")
    except Exception as err:
        logger.exception(f"Error closing synchronous Redis client: {err}")
        raise


def get_async_redis_client() -> async_redis.Redis:
    """Return the lazily initialized asynchronous Redis client.

    Use this client in async application code to avoid blocking the event loop
    while performing Redis operations. Repeated calls return the same client
    instance.

    Returns:
        The configured asynchronous Redis client.
    """
    global _async_redis_client

    if _async_redis_client is not None:
        return _async_redis_client

    settings: AppBaseSettings = get_app_settings()

    # The async client manages asynchronous connections through its own pool.
    # Actual network communication occurs when an operation is awaited.
    client = async_redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,       # Decode Redis responses as strings.
        health_check_interval=30,    # Check connection health after idle periods.
        socket_timeout=3.0,          # Limit individual socket operations.
        retry_on_timeout=True,       # Enable retries for timeout errors.
        max_connections=50,          # Maximum connections in the pool.
    )

    _async_redis_client = client
    return client


async def ping_redis_async() -> bool:
    """Check asynchronously whether the Redis server responds to PING.

    Returns:
        True if the ping succeeds; otherwise, False.
    """
    client = get_async_redis_client()
    logger = get_logger(_LOGGER)

    try:
        await client.ping()
        logger.debug("Successfully pinged asynchronous Redis server.")
        return True
    except RedisError as err:
        logger.error(f"Failed to ping asynchronous Redis server: {err}")
        return False


async def close_redis_async() -> None:
    """Close the asynchronous Redis client during application shutdown.

    Resets the module-level singleton after closing so that it can be
    initialized again if required.

    Raises:
        RedisNotInitializedError: If the client has not been initialized.
        Exception: Re-raises an unexpected error encountered during cleanup.
    """
    global _async_redis_client

    if _async_redis_client is None:
        raise RedisNotInitializedError(
            "Asynchronous Redis client must be initialized before closing."
        )

    logger = get_logger(_LOGGER)

    try:
        await _async_redis_client.aclose()
        _async_redis_client = None
        logger.info("Asynchronous Redis client closed successfully.")
    except Exception as err:
        logger.exception(f"Error closing asynchronous Redis client: {err}")
        raise