import redis.asyncio as redis
from redis.exceptions import RedisError

from config.settings import settings
from config.logging import get_logger

# Create a robust, production-ready Redis client
redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,       # Return strings instead of bytes
    health_check_interval=30,    # Keep idle connections alive
    socket_timeout=3.0,          # Don't hang forever on network stalls
    retry_on_timeout=True,       # Recover from brief network blips
    max_connections=50           # Cap the connection pool 
)

logger = get_logger("infra.redis.client")

async def ping_redis() -> bool:
    """
    Ping the Redis server to check if it's reachable.

    :return: True if the Redis server is reachable, False otherwise.
    """
    try:
        await redis_client.ping()
        logger.debug("Successfully pinged Redis server.")
        return True
    except RedisError as e:
        logger.error(f"Failed to ping Redis server: {e}")
        return False