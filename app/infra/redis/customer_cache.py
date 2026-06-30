import json

from redis.exceptions import RedisError

from config.settings import settings
from config.constants import CUSTOMER_CACHE_TTL
from config.logging import get_logger
from infra.redis.client import redis_client


logger = get_logger("infra.redis.customer_cache")


async def get_cached_customer(phone: str) -> dict | None:
    """
    Retrieve cached customer data from Redis.

    :param phone: The phone number of the customer.
    :return: A dictionary containing the cached customer data, or None if not found.
    """
    key = f"customer:{phone}"
    try:
        # Retrieve the cached customer data
        cached_data = await redis_client.get(key)
        if cached_data:
            # Decode and parse the cached data from JSON
            customer_data = json.loads(cached_data)
            logger.debug(f"Retrieved cached customer data for {phone}: {customer_data}")
            return customer_data
        else:
            logger.debug(f"No cached customer data found for {phone}")
            return None
    except RedisError as e:
        logger.error(f"Failed to retrieve cached customer data for {phone}: {e}")
        return None


async def cache_customer(phone: str, customer_data: dict) -> None:
    """
    Cache customer data in Redis.

    :param phone: The phone number of the customer.
    :param customer_data: A dictionary containing the customer data to cache.
    :return: None
    """
    key = f"customer:{phone}"
    try:
        # Serialize the customer data to JSON and store it in Redis with a TTL
        await redis_client.set(key, json.dumps(customer_data), ex=CUSTOMER_CACHE_TTL)
        logger.debug(f"Cached customer data for {phone}: {customer_data}")
    except RedisError as e:
        logger.error(f"Failed to cache customer data for {phone}: {e}")


async def delete_cached_customer(phone: str) -> None:
    """
    Delete cached customer data from Redis.

    :param phone: The phone number of the customer.
    :return: None
    """
    key = f"customer:{phone}"
    try:
        # Delete the cached customer data
        await redis_client.delete(key)
        logger.debug(f"Deleted cached customer data for {phone}")
    except RedisError as e:
        logger.error(f"Failed to delete cached customer data for {phone}: {e}")