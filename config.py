"""Configuration management for the Finance Agent."""

import os
from typing import Dict, Any

import dotenv
from loguru import logger

# Load environment variables from .env file
dotenv.load_dotenv()

class Config:
    """Configuration class for the Finance Agent application."""
    
    # API Keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    SARVAM_AI_API_KEY = os.getenv("SARVAM_AI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Database Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Scheduling
    TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")
    MARKET_BRIEF_HOUR = int(os.getenv("MARKET_BRIEF_HOUR", 8))
    MARKET_BRIEF_MINUTE = int(os.getenv("MARKET_BRIEF_MINUTE", 0))
    
    # FastAPI Settings
    FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))
    
    # Streamlit Settings
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
    
    # Model Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VOICE_MODEL = os.getenv("VOICE_MODEL", "meera")
    
    # External Service URLs
    FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v1")
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith("__") and not callable(value)}
    
    @classmethod
    def validate(cls) -> bool:
        """Validate the configuration.
        
        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        required_keys = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
        missing_keys = [key for key in required_keys if getattr(cls, key) is None]
        
        if missing_keys:
            logger.warning(f"Missing required configuration keys: {missing_keys}")
            return False
        
        return True

# Initialize logging
def setup_logging():
    """Set up logging with loguru."""
    logger.remove()  # Remove default handler
    logger.add(
        "logs/finance_agent_{time}.log",
        rotation="500 MB",
        level=Config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=Config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

# Initialize logging when the module is imported
setup_logging()
