from redis.exceptions import RedisError

from shared.infra.redis.client import get_redis_client
from shared.logging_setup import get_logger
from shared.config import WEB_TOKEN_RATE_LIMIT, WEB_TOKEN_RATE_WINDOW_SECONDS


_LOGGER = "infra.redis.web_token_limit"


async def check_token_rate_limit(
        identifier: str,
        limit: int = WEB_TOKEN_RATE_LIMIT,
        window_seconds: int = WEB_TOKEN_RATE_WINDOW_SECONDS
    ) -> bool:
    """
    Checks and increments a rate limit counter for LiveKit token issuance using Redis.

    This implements a simple fixed-window rate limiter. It increments a counter
    tied to the given identifier. If it is the first request in the window, it
    sets the key to expire after `window_seconds`.

    Note: If a Redis connection error occurs, this function "fails open" (returns True)
    so that users are not blocked from generating tokens during a cache outage.

    Args:
        identifier (str): A unique string identifying the requester (e.g., an IP address, user ID, or session ID).
        limit (int, optional): The maximum number of tokens allowed within the time window. Defaults to WEB_TOKEN_RATE_LIMIT.
        window_seconds (int, optional): The duration in seconds before the rate limit counter resets. Defaults to WEB_TOKEN_RATE_WINDOW_SECONDS.

    Returns:
        bool: True if the request is under the limit (or if Redis fails), False if the rate limit has been exceeded.
    """

    logger = get_logger(_LOGGER)
    redis_client = get_redis_client()

    key = f"token_rate_limit:{identifier}"

    try:
        await redis_client.incr(key)
        identifier_count = await redis_client.get(key)
        if int(identifier_count) == 1:
            await redis_client.expire(key, window_seconds)

        return int(identifier_count) <= limit
    except RedisError as e:
        logger.error(f"Rate limit check failed — identifier={identifier} error={e}")
        return True