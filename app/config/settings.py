from pydantic import BaseSettings

class Settings(BaseSettings):
     GEMINI_API_KEY_1 : str
     GEMINI_API_KEY_2 : str
     GEMINI_API_KEY_3 : str
     EXOTEL_ACCOUNT_SID : str
     EXOTEL_API_KEY : str
     EXOTEL_API_TOKEN : str
     EXOTEL_PHONE_NUMBER : str
     REDIS_URL : str
     DATABASE_URL : str
     PINECONE_API_KEY : str
     PINECONE_INDEX : str
     PINECONE_ENVIRONMENT : str
     CRM_TYPE : str
     CRM_API_KEY : str
     CRM_BASE_URL : str
     LOG_LEVEL : str = "INFO"

     class Config:
        env_file = ".env"

settings = Settings()