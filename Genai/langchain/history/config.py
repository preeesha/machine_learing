"""
Configuration file for the History-Aware Chatbot
"""

import os
from typing import Optional

class ChatbotConfig:
    """Configuration class for chatbot settings"""
    
    # Google API Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "208"))
    
    # Memory Configuration
    MEMORY_WINDOW_SIZE: int = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
    
    # Vector Store Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "3"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.GOOGLE_API_KEY:
            print("❌ GOOGLE_API_KEY is required")
            return False
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 Chatbot Configuration:")
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  Memory Window: {cls.MEMORY_WINDOW_SIZE}")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Retriever K: {cls.RETRIEVER_K}")
        print(f"  API Key: {'✅ Set' if cls.GOOGLE_API_KEY else '❌ Not Set'}")
