"""
config.py
---------
Configuration management for the AI Document Processor
Handles API keys, model selection, and application settings
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Model Settings
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # Deduplication Settings
    FUZZY_MATCH_THRESHOLD = float(os.getenv("FUZZY_MATCH_THRESHOLD", "0.85"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    ENABLE_FUZZY_DEDUP = os.getenv("ENABLE_FUZZY_DEDUP", "true").lower() == "true"
    ENABLE_TRUNCATION_REPAIR = os.getenv("ENABLE_TRUNCATION_REPAIR", "true").lower() == "true"
    DEBUG_DEDUP = os.getenv("DEBUG_DEDUP", "false").lower() == "true"
    
    # File Paths
    TEMP_DIR = Path("temp")
    OUTPUT_DIR = Path("output")
    
    # Processing Settings
    PDF_X_TOLERANCE = 2  # Horizontal spacing tolerance for text extraction
    PDF_Y_TOLERANCE = 2  # Vertical spacing tolerance for text extraction
    
    # MongoDB Atlas Learning System
    MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
    MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "ai_document_processor")
    MONGODB_COLLECTION_PREFIX = os.getenv("MONGODB_COLLECTION_PREFIX", "prod_")
    
    # Learning System Settings
    LEARNING_ENABLED = os.getenv("LEARNING_ENABLED", "false").lower() == "true"
    LEARNING_BATCH_SIZE = int(os.getenv("LEARNING_BATCH_SIZE", "10"))
    LEARNING_ANALYTICS_ENABLED = os.getenv("LEARNING_ANALYTICS_ENABLED", "true").lower() == "true"
    
    # Automated Feedback Settings
    AUTOMATED_FEEDBACK_ENABLED = os.getenv("AUTOMATED_FEEDBACK_ENABLED", "true").lower() == "true"
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "No API key found! Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file"
            )
        
        # Create directories if they don't exist
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration as dictionary"""
        return {
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS
        }
    
    @classmethod
    def display_config(cls):
        """Display current configuration (for debugging)"""
        print("=" * 50)
        print("AI Document Processor - Configuration")
        print("=" * 50)
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Temperature: {cls.LLM_TEMPERATURE}")
        print(f"Max Tokens: {cls.LLM_MAX_TOKENS}")
        print(f"API Key Set: {'✅' if cls.OPENAI_API_KEY else '❌'}")
        print("=" * 50)


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️  Configuration Warning: {e}")
