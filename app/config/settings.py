from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────────
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

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

    # ── Cartesia ──────────────────────────────────────────────────────────────
    cartesia_api_key: str 
    cartesia_voice_id: str 

    # ── App ───────────────────────────────────────────────────────────────────
    app_port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()