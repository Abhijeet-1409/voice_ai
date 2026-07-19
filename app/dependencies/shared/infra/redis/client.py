from functools import cache

import redis.asyncio as redis
from redis.exceptions import RedisError

from config.logger import get_logger
from config.settings import get_app_settings, AppBaseSettings


_LOGGER_NAME = "infra.redis.client"


@cache
def get_redis_client() -> redis.Redis:
    """
    Create and cache a singleton Redis asynchronous client.

    The client is configured from the application settings with production-ready
    defaults, including connection pooling, automatic string decoding, and
    network timeout handling.

    Returns:
        redis.Redis: The configured asynchronous Redis client instance.
    """

    settings: AppBaseSettings = get_app_settings()

    redis_client = redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,       # Return strings instead of bytes
        health_check_interval=30,    # Keep idle connections alive
        socket_timeout=3.0,          # Don't hang forever on network stalls
        retry_on_timeout=True,       # Recover from brief network blips
        max_connections=50           # Cap the connection pool
    )

    return redis_client


async def ping_redis() -> bool:
    """
    Verify the Redis server is reachable and operational.

    This executes a ping command against the Redis server, typically used
    during application startup or health checks. Logs the success or failure
    of the connection attempt.

    Returns:
        bool: True if the Redis server responds successfully, False otherwise.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER_NAME)

    try:
        await redis_client.ping()
        logger.debug("Successfully pinged Redis server.")
        return True
    except RedisError as e:
        logger.error(f"Failed to ping Redis server: {e}")
        return False


async def close_redis():
    """
    Safely close the Redis connection pool during application shutdown.

    This ensures all active connections are gracefully disconnected and returned
    to the OS, preventing socket leaks, hanging connections, and server-side warnings.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER_NAME)

    try:
        # Note: Use .aclose() for redis-py version 5.0+.
        # For older versions, use: await redis_client.close()
        await redis_client.aclose()
        logger.info("Redis connection pool closed successfully.")
    except Exception as e:
        logger.error(f"Error closing Redis connection pool: {e}")