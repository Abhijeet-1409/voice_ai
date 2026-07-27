from shared.config.logger import get_logger
from shared.infra.redis.api_key_state import is_key_on_cooldown, get_key_usage, increment_key_usage

from config.worker_settings import get_worker_settings


_LOGGER = "worker.agent.key_selector"
logger = get_logger(_LOGGER)


async def select_gemini_key() -> str:
    """
    Selects the least used, available Gemini API key from the worker settings.
    
    Checks Redis for cooldown statuses and current usage counts. Selects the 
    key with the lowest usage, increments its usage counter, and returns 
    the actual API key string.

    Returns:
        str: The selected Gemini API key value.
        
    Raises:
        AttributeError: If a constructed key is missing from WorkerSettings.
        RuntimeError: If all API keys are currently on cooldown.
    """
    settings = get_worker_settings()

    _key_indexs = ["1", "2", "3"]
    _key_prefix = "GEMINI_API_KEY_"

    try:
        constructed_keys = [f"{_key_prefix}{index}" for index in _key_indexs]
        
        key_value_map = {key: getattr(settings, key) for key in constructed_keys}    
        
        available_keys = [key for key in constructed_keys if not await is_key_on_cooldown(key)]
        
        if not available_keys:
            logger.error("All Gemini API keys are currently on cooldown!")
            raise RuntimeError("No available Gemini API keys.")

        key_usage_map = {key: await get_key_usage(key) for key in available_keys}
        
        key_id = min(key_usage_map, key=key_usage_map.get) 
        
        logger.debug(f"Selected '{key_id}' (Current usage: {key_usage_map[key_id]}).")
        
        await increment_key_usage(key_id)

        return key_value_map[key_id]
                
    except AttributeError as e:
        logger.error(f"Missing API key configuration in settings — {e}")
        raise