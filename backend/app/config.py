import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama LLM settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # Confidence thresholds for ML predictions
    SYMPTOM_CONFIDENCE_THRESHOLD: float = 0.55
    IMAGE_CONFIDENCE_THRESHOLD: float = 0.55

    # Number of alternative predictions to include
    MAX_ALTERNATIVES: int = 3

    # Gap between top-2 predictions that triggers uncertainty
    UNCERTAINTY_GAP_THRESHOLD: float = 0.10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
