import redis.asyncio as redis
from redis.exceptions import RedisError

from shared.logging_setup import get_logger
from shared.config import get_app_settings, AppBaseSettings


_LOGGER = "infra.redis.client"


# Private singleton instance of the asynchronous Redis client.
# Lazy-loaded to ensure it is only created when first needed, preventing 
# socket corruption when processes fork (e.g., during worker pre-warming).
_redis_client: redis.Redis | None = None


class RedisNotInitializedError(RuntimeError):
    """
    Exception raised for errors in the Redis initialization lifecycle.

    This is triggered when a teardown operation is attempted before the 
    asynchronous Redis client has been initialized.
    """
    def __init__(self, message="Redis client has not been initialized. Call get_redis_client() first."):
        self.message = message
        super().__init__(self.message)


def get_redis_client() -> redis.Redis:
    """
    Retrieves or creates a lazy-loaded Redis asynchronous client.

    The client is configured from the application settings with production-ready
    defaults, including connection pooling, automatic string decoding, and
    network timeout handling. Once created, it is cached globally.

    Returns:
        redis.Redis: The configured asynchronous Redis client instance.
    """
    global _redis_client

    if _redis_client:
        return _redis_client

    settings: AppBaseSettings = get_app_settings()

    redis_client = redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,       # Return strings instead of bytes
        health_check_interval=30,    # Keep idle connections alive
        socket_timeout=3.0,          # Don't hang forever on network stalls
        retry_on_timeout=True,       # Recover from brief network blips
        max_connections=50           # Cap the connection pool
    )

    _redis_client = redis_client

    return redis_client


async def ping_redis() -> bool:
    """
    Verifies the Redis server is reachable and operational on startup.

    Executes a ping command against the lazy-loaded Redis client to ensure
    the connection pool is functioning correctly. Logs the success or failure
    of the connection attempt.

    Returns:
        bool: True if the Redis server responds successfully, False otherwise.
    """
    # Simply call the getter to ensure it's initialized before testing
    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    try:
        await redis_client.ping()
        logger.debug("Successfully pinged Redis server.")
        return True
    except RedisError as e:
        logger.error(f"Failed to ping Redis server: {e}")
        return False


async def close_redis():
    """
    Safely disposes of the Redis connection pool during application shutdown.

    This ensures all active connections are gracefully disconnected and returned
    to the OS, preventing socket leaks. It also resets the module-level 
    singleton (`_redis_client`) to allow safe re-initialization if needed.

    Raises:
        RedisNotInitializedError: If called before the Redis client has been initialized.
    """
    global _redis_client

    if _redis_client is None:
        raise RedisNotInitializedError("Redis client must be initialized before closing.")

    logger = get_logger(_LOGGER)

    try:
        # Note: Use .aclose() for redis-py version 5.0+.
        # For older versions, use: await redis_client.close()
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection pool closed successfully.")
    except Exception as e:
        logger.error(f"Error closing Redis connection pool: {e}")