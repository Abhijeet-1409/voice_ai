import json
from json import JSONDecodeError

from redis.exceptions import RedisError

from config.logger import get_logger
from config.constants import CUSTOMER_CACHE_TTL

from infra.redis.client import get_redis_client


_LOGGER_NAME = "infra.redis.customer_cache"


async def get_cached_customer(phone: str) -> dict | None:
    """
    Retrieve and deserialize cached customer data from Redis.

    Args:
        phone (str): The customer's phone number used as the cache key.

    Returns:
        dict | None: A dictionary containing the cached customer data,
        or None if the data is not found or an error occurs.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER_NAME)

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
    except JSONDecodeError as e:
        # Handle corrupted or invalid JSON in Redis
        logger.error(f"Invalid JSON data found in cache for {phone}: {e}")
        return None
    except RedisError as e:
        # Handle Redis connection/command failures
        logger.error(f"Failed to retrieve cached customer data for {phone}: {e}")
        return None


async def cache_customer(phone: str, customer_data: dict) -> None:
    """
    Serialize and cache customer data in Redis with a predefined TTL.

    Args:
        phone (str): The customer's phone number used as the cache key.
        customer_data (dict): The dictionary containing customer information to cache.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER_NAME)

    key = f"customer:{phone}"

    try:
        # Serialize the customer data to JSON and store it in Redis with a TTL
        await redis_client.set(key, json.dumps(customer_data), ex=CUSTOMER_CACHE_TTL)
        logger.debug(f"Cached customer data for {phone}: {customer_data}")
    except TypeError as e:
        # Handle cases where customer_data contains non-serializable objects (e.g., datetimes)
        logger.error(f"Failed to serialize customer data to JSON for {phone}: {e}")
    except RedisError as e:
        # Handle Redis connection/command failures
        logger.error(f"Failed to cache customer data for {phone}: {e}")


async def delete_cached_customer(phone: str) -> None:
    """
    Delete cached customer data from Redis.

    Args:
        phone (str): The customer's phone number used as the cache key.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER_NAME)

    key = f"customer:{phone}"

    try:
        # Delete the cached customer data
        await redis_client.delete(key)
        logger.debug(f"Deleted cached customer data for {phone}")
    except RedisError as e:
        logger.error(f"Failed to delete cached customer data for {phone}: {e}")