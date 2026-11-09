
"""Celery application configuration for Prompt Cache Service."""

from celery import Celery

from ..config import get_prompt_cache_settings


settings = get_prompt_cache_settings()

celery_app = Celery(
    "prompt_cache_service",
    broker=settings.REDIS_URL,
    include=[
        "app.services.prompt_cache_service.workers.tasks",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Time handling
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_ignore_result=True,
    worker_prefetch_multiplier=1,

    # Acknowledge tasks after successful delivery to the worker.
    # Job retries are managed by PostgreSQL and the transactional outbox,
    # not by Celery's automatic retry mechanism.
    task_acks_late=False,
    task_reject_on_worker_lost=False,

    # Broker connection
    broker_connection_retry_on_startup=True,

    # Default queue
    task_default_queue="prompt-cache",

    # Explicit task routing
    task_routes={
        "prompt_cache.create_cache": {
            "queue": "prompt-cache",
        },
        "prompt_cache.recover_jobs": {
            "queue": "prompt-cache",
        },
    },
)
