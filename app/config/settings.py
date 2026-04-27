from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────────
    gemini_api_key: str

    # ── Email ─────────────────────────────────────────────────────────────────
    gmail_address: str
    gmail_app_password: str
    notification_email: str

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str
    postgres_password: str

    # ── App ───────────────────────────────────────────────────────────────────
    app_port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()