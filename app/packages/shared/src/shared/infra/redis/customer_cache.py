import json
from json import JSONDecodeError

from redis.exceptions import RedisError

from shared.logging_setup import get_logger
from shared.config import CUSTOMER_CACHE_TTL
from .client import get_redis_client


_LOGGER = "infra.redis.customer_cache"


async def get_cached_customer(identifier: str) -> dict | None:
    """
    Retrieves and deserializes cached customer data from Redis.

    Args:
        identifier (str): The unique identifier for the customer (e.g., phone number or Clerk ID).

    Returns:
        dict | None: A dictionary containing the cached customer data, or None if
                     not found, invalid, or a connection error occurs.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    key = f"customer:{identifier}"

    try:
        # Retrieve the cached customer data
        cached_data = await redis_client.get(key)
        if cached_data:
            # Decode and parse the cached data from JSON
            customer_data = json.loads(cached_data)
            logger.debug(f"Retrieved cached customer data for {identifier}: {customer_data}")
            return customer_data
        else:
            logger.debug(f"No cached customer data found for {identifier}")
            return None
    except JSONDecodeError as e:
        # Handle corrupted or invalid JSON in Redis
        logger.error(f"Invalid JSON data found in cache for {identifier}: {e}")
        return None
    except RedisError as e:
        # Handle Redis connection/command failures
        logger.error(f"Failed to retrieve cached customer data for {identifier}: {e}")
        return None


async def cache_customer(identifier: str, customer_data: dict) -> None:
    """
    Serializes and caches customer data in Redis with a predefined TTL.

    Args:
        identifier (str): The unique identifier for the customer (e.g., phone number or Clerk ID).
        customer_data (dict): The dictionary containing customer information to cache.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    key = f"customer:{identifier}"

    try:
        # Serialize the customer data to JSON and store it in Redis with a TTL
        await redis_client.set(key, json.dumps(customer_data), ex=CUSTOMER_CACHE_TTL)
        logger.debug(f"Cached customer data for {identifier}: {customer_data}")
    except TypeError as e:
        # Handle cases where customer_data contains non-serializable objects (e.g., datetimes)
        logger.error(f"Failed to serialize customer data to JSON for {identifier}: {e}")
    except RedisError as e:
        # Handle Redis connection/command failures
        logger.error(f"Failed to cache customer data for {identifier}: {e}")


async def delete_cached_customer(identifier: str) -> None:
    """
    Deletes cached customer data from Redis.

    Args:
        identifier (str): The unique identifier for the customer (e.g., phone number or Clerk ID).
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    key = f"customer:{identifier}"

    try:
        # Delete the cached customer data
        await redis_client.delete(key)
        logger.debug(f"Deleted cached customer data for {identifier}")
    except RedisError as e:
        logger.error(f"Failed to delete cached customer data for {identifier}: {e}")