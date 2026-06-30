from redis.exceptions import RedisError

from config.logging import get_logger
from config.constants import GEMINI_KEY_COOLDOWN_SECONDS

from infra.redis.client import redis_client


logger = get_logger("infra.redis.api_key_state")


async def get_key_usage(key_id: str) -> int:
    """
    Get the usage count of an API key from Redis.

    :param key_id: The ID of the API key.
    :return: The usage count, or 0 if not found or Redis fails.
    """
    try:
        usage = await redis_client.get(f"gemini_key_usage:{key_id}")
        return int(usage) if usage is not None else 0
    except RedisError as e:
        logger.error(f"Failed to get key usage — key_id={key_id} error={e}")
        return 0


async def increment_key_usage(key_id: str) -> None:
    """
    Increment the usage count of an API key in Redis.
    No TTL — usage counters accumulate permanently for analytics.

    :param key_id: The ID of the API key.
    """
    try:
        await redis_client.incr(f"gemini_key_usage:{key_id}")
        logger.debug(f"Incremented usage — key_id={key_id}")
    except RedisError as e:
        logger.error(f"Failed to increment key usage — key_id={key_id} error={e}")


async def set_key_cooldown(key_id: str, cooldown_seconds: int = None) -> None:
    """
    Put an API key on cooldown in Redis.
    Key expires automatically after cooldown_seconds.

    :param key_id: The ID of the API key.
    :param cooldown_seconds: Override default cooldown. Uses GEMINI_KEY_COOLDOWN_SECONDS if not provided.
    """
    ttl = cooldown_seconds if cooldown_seconds is not None else GEMINI_KEY_COOLDOWN_SECONDS
    try:
        await redis_client.setex(f"gemini_key_cooldown:{key_id}", ttl, "1")
        logger.warning(f"Key on cooldown — key_id={key_id} ttl={ttl}s")
    except RedisError as e:
        logger.error(f"Failed to set cooldown — key_id={key_id} error={e}")


async def is_key_on_cooldown(key_id: str) -> bool:
    """
    Check if an API key is currently on cooldown.

    :param key_id: The ID of the API key.
    :return: True if on cooldown, False if available or Redis fails.
    """
    try:
        cooldown = await redis_client.get(f"gemini_key_cooldown:{key_id}")
        return cooldown is not None
    except RedisError as e:
        logger.error(f"Failed to check cooldown — key_id={key_id} error={e}")
        return False  # assume available if Redis is down