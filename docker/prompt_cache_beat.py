"""Celery Beat configuration for Prompt Cache Service recovery.

This adapter keeps the API and Celery application modules unchanged.
Celery Beat imports this module as its application entrypoint.
"""

from app.services.prompt_cache_service.config import get_prompt_cache_settings
from app.services.prompt_cache_service.workers.celery_app import celery_app

settings = get_prompt_cache_settings()

celery_app.conf.beat_schedule = {
    "prompt-cache-recovery": {
        "task": "prompt_cache.recover_jobs",
        "schedule": float(settings.RECOVERY_INTERVAL_SECONDS),
    },
}
