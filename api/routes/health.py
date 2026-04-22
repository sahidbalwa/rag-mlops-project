"""
Health check route.
=== FILE: api/routes/health.py ===
"""

import time
from fastapi import APIRouter
from api.schemas.response import HealthResponse
from configs.loader import load_yaml_config

router = APIRouter()
START_TIME = time.time()

# Load config to get the version
try:
    config = load_yaml_config("config.yaml")
    APP_VERSION = config.get("pipeline", {}).get("version", "unknown")
except Exception:
    APP_VERSION = "unknown"

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns API status, uptime, and version.
    """
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="ok",
        uptime_seconds=uptime,
        version=APP_VERSION
    )
