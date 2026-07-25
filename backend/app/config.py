"""
Configuration settings loader.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Placeholder settings config class.
    """
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "127.0.0.1")

settings = Settings()
