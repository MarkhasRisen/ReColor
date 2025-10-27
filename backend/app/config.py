"""Application configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


# Load .env from the backend directory
# __file__ is backend/app/config.py
# parent is backend/app/
# parent.parent is backend/
backend_dir = Path(__file__).parent.parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)


def load_config() -> Dict[str, str]:
    """Load configuration values with sensible defaults."""
    return {
        "FIREBASE_CREDENTIAL_PATH": os.getenv("FIREBASE_CREDENTIAL_PATH", ""),
        "FIREBASE_PROJECT_ID": os.getenv("FIREBASE_PROJECT_ID", ""),
        "TFLITE_MODEL_DIR": os.getenv("TFLITE_MODEL_DIR", "models"),
        "PIPELINE_CACHE_TTL": os.getenv("PIPELINE_CACHE_TTL", "60"),
        "OFFLOAD_ENDPOINT": os.getenv("OFFLOAD_ENDPOINT", ""),
        "OFFLOAD_TIMEOUT": os.getenv("OFFLOAD_TIMEOUT", "5"),
    }
