"""
Configuration settings for the Food Recognition API
"""
import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Model configuration
    MODEL_PATH: str = "models/food_model.h5"
    IMAGE_SIZE: int = 224
    CONFIDENCE_THRESHOLD: float = 0.3
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "bmp"]
    
    # CORS settings - can be a comma-separated string or list
    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ]
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Split comma-separated string
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # Portion estimation
    PORTION_ESTIMATION_METHOD: str = "standard"  # standard, ml, or custom
    
    # Data paths
    NUTRITION_DB_PATH: str = "data/nutrition_db.json"
    CLASS_NAMES_PATH: str = "data/class_names.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
