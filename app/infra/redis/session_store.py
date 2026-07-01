import json
from datetime import datetime

from redis.exceptions import RedisError

from config.constants import CALL_SESSION_TTL
from config.logging import get_logger

from infra.redis.client import redis_client


logger = get_logger("infra.redis.session_store")


async def append_turn(stream_sid: str, speaker: str, text: str) -> None:
    """
    Append a turn to the call session transcript in Redis.

    :param stream_sid: The unique identifier for the call session.
    :param speaker: The speaker of the turn (e.g., "agent" or "customer").
    :param text: The text of the turn.
    :return: None
    """
    key = f"transcript:{stream_sid}"
    turn = {
        "speaker": speaker,
        "text": text,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        # Append the new turn to the existing transcript
        await redis_client.rpush(key, json.dumps(turn))
        # Set the TTL for the call session transcript
        await redis_client.expire(key, CALL_SESSION_TTL)
        logger.debug(f"Appended turn to session {stream_sid}: {turn}")
    except RedisError as e:
        logger.error(f"Failed to append turn to session {stream_sid}: {e}")


async def get_transcript(stream_sid: str) -> list:
    """
    Retrieve the call session transcript from Redis.

    :param stream_sid: The unique identifier for the call session.
    :return: A list of turns in the transcript.
    """
    key = f"transcript:{stream_sid}"
    try:
        # Retrieve all turns from the transcript
        turns = await redis_client.lrange(key, 0, -1)
        # Decode and parse each turn from JSON
        transcript = [json.loads(turn.decode('utf-8')) for turn in turns]
        logger.debug(f"Retrieved transcript for session {stream_sid}: {transcript}")
        return transcript
    except RedisError as e:
        logger.error(f"Failed to retrieve transcript for session {stream_sid}: {e}")
        return []


async def delete_transcript(stream_sid: str) -> None:
    """
    Delete the call session transcript from Redis.

    :param stream_sid: The unique identifier for the call session.
    :return: None
    """
    key = f"transcript:{stream_sid}"
    try:
        # Delete the transcript from Redis
        await redis_client.delete(key)
        logger.debug(f"Deleted transcript for session {stream_sid}")
    except RedisError as e:
        logger.error(f"Failed to delete transcript for session {stream_sid}: {e}")