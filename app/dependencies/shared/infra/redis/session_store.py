import json
from json import JSONDecodeError
from datetime import datetime, timezone

from redis.exceptions import RedisError

from shared.logging_setup import get_logger
from shared.config import CALL_SESSION_TTL
from shared.infra.redis.client import get_redis_client


_LOGGER = "infra.redis.session_store"


async def append_turn(stream_sid: str, speaker: str, text: str) -> None:
    """
    Append a conversational turn to the call session transcript in Redis.

    Args:
        stream_sid (str): The unique identifier for the call session.
        speaker (str): The speaker of the turn (e.g., "agent" or "customer").
        text (str): The text content of the turn.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    key = f"transcript:{stream_sid}"

    # Use timezone-aware UTC datetime instead of the deprecated utcnow()
    turn = {
        "speaker": speaker,
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    try:
        # Append the new turn to the existing transcript
        await redis_client.rpush(key, json.dumps(turn))
        # Set the TTL for the call session transcript
        await redis_client.expire(key, CALL_SESSION_TTL)
        logger.debug(f"Appended turn to session {stream_sid}: {turn}")
    except TypeError as e:
        logger.error(f"Failed to serialize turn for session {stream_sid}: {e}")
    except RedisError as e:
        logger.error(f"Failed to append turn to session {stream_sid}: {e}")


async def get_transcript(stream_sid: str) -> list:
    """
    Retrieve the full call session transcript from Redis.

    Args:
        stream_sid (str): The unique identifier for the call session.

    Returns:
        list: A list of dictionaries representing the turns in the transcript.
        Returns an empty list if not found or if an error occurs.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    key = f"transcript:{stream_sid}"

    try:
        # Retrieve all turns from the transcript
        turns = await redis_client.lrange(key, 0, -1)

        # Because decode_responses=True on the client, 'turn' is already a str.
        # No need to call .decode('utf-8')
        transcript = [json.loads(turn) for turn in turns]

        logger.debug(f"Retrieved transcript for session {stream_sid}: {transcript}")
        return transcript
    except JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in transcript for session {stream_sid}: {e}")
        return []
    except RedisError as e:
        logger.error(f"Failed to retrieve transcript for session {stream_sid}: {e}")
        return []


async def delete_transcript(stream_sid: str) -> None:
    """
    Delete the call session transcript from Redis.

    Args:
        stream_sid (str): The unique identifier for the call session.
    """

    redis_client = get_redis_client()
    logger = get_logger(_LOGGER)

    key = f"transcript:{stream_sid}"

    try:
        # Delete the transcript from Redis
        await redis_client.delete(key)
        logger.debug(f"Deleted transcript for session {stream_sid}")
    except RedisError as e:
        logger.error(f"Failed to delete transcript for session {stream_sid}: {e}")