from shared.logging_setup import get_logger

from config import get_worker_settings


_LOGGER = "worker.agent.key_selector"
logger = get_logger(_LOGGER)

_KEY_PREFIX = "GEMINI_API_KEY_"


def select_gemini_key() -> str:
    """
    Returns the Gemini API key configured for manual use via
    GEMINI_ACTIVE_KEY_INDEX in settings.

    No rotation, no Redis, no usage/cooldown tracking — the active key
    is chosen manually by setting GEMINI_ACTIVE_KEY_INDEX in the
    environment (1, 2, or 3) and restarting the worker. Switch keys by
    changing that value if the active one gets rate-limited.

    Returns:
        str: The selected Gemini API key value.

    Raises:
        AttributeError: If GEMINI_ACTIVE_KEY_INDEX doesn't correspond
            to a configured GEMINI_API_KEY_N field in WorkerSettings.
    """
    settings = get_worker_settings()
    key_field = f"{_KEY_PREFIX}{settings.GEMINI_ACTIVE_KEY_INDEX}"

    try:
        key_value = getattr(settings, key_field)
    except AttributeError as e:
        logger.error(
            f"GEMINI_ACTIVE_KEY_INDEX={settings.GEMINI_ACTIVE_KEY_INDEX} does not "
            f"correspond to a configured key ({key_field} missing) — {e}"
        )
        raise

    logger.debug(f"Using '{key_field}' (manual selection).")
    return key_value