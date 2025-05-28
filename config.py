"""Configuration management for the Finance Agent."""

import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    
    # MCP Configuration
    FIRECRAWL_MCP_SERVER = os.getenv("FIRECRAWL_MCP_SERVER", "puppeteer")
    FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL", "https://firecrawl-api.example.com")
    
    # LLM Provider
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
    
    # Database Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    
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
    
    # Required API Key Groups
    # At least one from each group must be present
    REQUIRED_API_GROUPS = {
        "vector_db": ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"],
        "voice": ["SARVAM_AI_API_KEY", "ELEVENLABS_API_KEY"],
        "llm": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOCAL_MODEL_PATH"],
        "scraping": ["FIRECRAWL_API_KEY"]
    }
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith("__") and not callable(value)}
    
    @classmethod
    def validate(cls) -> bool:
        """Validate the configuration and check for required API keys.
        
        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        # Check each API key group to ensure at least one key from each group is present
        missing_groups = []
        
        for group_name, keys in cls.REQUIRED_API_GROUPS.items():
            # Check if any key in this group is present
            if not any(getattr(cls, key) for key in keys):
                missing_groups.append(group_name)
                logger.warning(f"Missing API keys for {group_name} group. Need at least one of: {', '.join(keys)}")
        
        return len(missing_groups) == 0
    
    @classmethod
    def check_and_prompt_for_missing_keys(cls) -> None:
        """Check for missing API keys and prompt the user to enter them if needed."""
        for group_name, keys in cls.REQUIRED_API_GROUPS.items():
            # Check if any key in this group is present
            if not any(getattr(cls, key) for key in keys):
                print(f"\n{'='*80}")
                print(f"Missing required API keys for: {group_name.upper()}")
                print(f"You need at least one of the following: {', '.join(keys)}")
                print(f"{'='*80}\n")
                
                # Prompt for one of the keys in this group
                for key in keys:
                    print(f"Would you like to enter your {key}? (y/n)")
                    response = input("> ").strip().lower()
                    
                    if response in ['y', 'yes']:
                        print(f"Please enter your {key}:")
                        api_key = input("> ").strip()
                        
                        if api_key:
                            # Save the key to the environment and update the class attribute
                            os.environ[key] = api_key
                            setattr(cls, key, api_key)
                            
                            # Save to .env file
                            cls._save_key_to_env_file(key, api_key)
                            print(f"{key} saved successfully!")
                            break
                        else:
                            print("No key entered. Please try again.")
                    
                    # If we've gone through all options and user declined them all
                    if key == keys[-1] and response not in ['y', 'yes']:
                        print(f"Warning: No API key provided for {group_name}. Some functionality will be limited.")
    
    @classmethod
    def _save_key_to_env_file(cls, key: str, value: str) -> None:
        """Save an API key to the .env file.
        
        Args:
            key: The environment variable name
            value: The API key value
        """
        env_path = Path(".env")
        
        # Create .env file if it doesn't exist
        if not env_path.exists():
            with open(env_path, "w") as f:
                f.write(f"{key}={value}\n")
            return
        
        # Read existing .env file
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        # Check if key already exists in file
        key_exists = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                key_exists = True
                break
        
        # Add key if it doesn't exist
        if not key_exists:
            lines.append(f"{key}={value}\n")
        
        # Write updated content back to file
        with open(env_path, "w") as f:
            f.writelines(lines)

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
