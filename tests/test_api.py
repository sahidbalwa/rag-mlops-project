"""
Tests for API endpoints.
=== FILE: tests/test_api.py ===
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """
    Ensures the /health endpoint is available and returns the expected schema.
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "version" in data
