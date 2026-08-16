from redis.exceptions import RedisError

from shared.logging_setup import get_logger
from shared.config import GEMINI_KEY_COOLDOWN_SECONDS
from shared.infra.redis.client import get_redis_client


_LOGGER = "infra.redis.api_key_state"


async def get_key_usage(key_id: str) -> int:
    """
    Retrieve the total usage count of a specific API key from Redis.

    Args:
        key_id (str): The unique identifier of the API key.

    Returns:
        int: The usage count. Returns 0 if the key is not found or if a Redis error occurs.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    try:
        usage = await redis_client.get(f"gemini_key_usage:{key_id}")
        return int(usage) if usage is not None else 0
    except RedisError as e:
        logger.error(f"Failed to get key usage — key_id={key_id} error={e}")
        return 0


async def increment_key_usage(key_id: str) -> None:
    """
    Increment the lifetime usage count of an API key in Redis.

    Note: This key has no TTL (Time To Live). Usage counters accumulate
    permanently for analytics and tracking purposes.

    Args:
        key_id (str): The unique identifier of the API key.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    try:
        await redis_client.incr(f"gemini_key_usage:{key_id}")
        logger.debug(f"Incremented usage — key_id={key_id}")
    except RedisError as e:
        logger.error(f"Failed to increment key usage — key_id={key_id} error={e}")


async def set_key_cooldown(key_id: str, cooldown_seconds: int | None = None) -> None:
    """
    Place an API key on a temporary cooldown in Redis.

    The cooldown key expires automatically after the specified number of seconds,
    meaning the API key will automatically become available again.

    Args:
        key_id (str): The unique identifier of the API key.
        cooldown_seconds (int | None, optional): Override the default cooldown duration.
            Defaults to GEMINI_KEY_COOLDOWN_SECONDS if not provided.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    ttl = cooldown_seconds if cooldown_seconds is not None else GEMINI_KEY_COOLDOWN_SECONDS

    try:
        await redis_client.setex(f"gemini_key_cooldown:{key_id}", ttl, "1")
        logger.warning(f"Key on cooldown — key_id={key_id} ttl={ttl}s")
    except RedisError as e:
        logger.error(f"Failed to set cooldown — key_id={key_id} error={e}")


async def is_key_on_cooldown(key_id: str) -> bool:
    """
    Check if an API key is currently under an active cooldown period.

    Args:
        key_id (str): The unique identifier of the API key.

    Returns:
        bool: True if the key is on cooldown. False if the key is available,
        or if the Redis server is unreachable (fail-open strategy).
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    try:
        cooldown_exists = await redis_client.exists(f"gemini_key_cooldown:{key_id}")
        return cooldown_exists > 0
    except RedisError as e:
        logger.error(f"Failed to check cooldown — key_id={key_id} error={e}")
        # Fail-open strategy: assume available if Redis is down so we don't block all traffic
        return False