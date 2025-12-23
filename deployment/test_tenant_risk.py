"""
Test suite for Tenant Risk Scoring API
"""

import pytest
from fastapi.testclient import TestClient
from app import app
import logging

# Mute logging during tests to keep output clean
logging.getLogger("db_connection").setLevel(logging.WARNING)

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_tenant_risk_score_valid(client):
    """Test getting risk score for a valid tenant (ID=50 from seed data)."""
    # Tenant 50 is the first generated tenant
    tenant_id = 50
    response = client.post(f"/tenant-risk/{tenant_id}")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "tenant_id" in data
    assert data["tenant_id"] == tenant_id
    assert "trust_score" in data
    assert "risk_band" in data
    assert "risk_probability" in data
    assert "features" in data
    
    # Check values exist and are reasonable
    assert 0 <= data["trust_score"] <= 100
    assert data["risk_band"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert 0.0 <= data["risk_probability"] <= 1.0

    print(f"\nTenant {tenant_id} Score: {data['trust_score']} ({data['risk_band']})")
    print(f"Top Factors: {data['top_factors']}")

def test_tenant_risk_score_nonexistent(client):
    """Test getting risk score for a nonexistent tenant."""
    # ID 99999 shouldn't exist
    tenant_id = 99999
    response = client.post(f"/tenant-risk/{tenant_id}")
    
    # It might return a default score or 404 depending on implementation.
    # The current extract_tenant_features returns default 0 values if not found,
    # so it should likely return a score (probably 100/LOW due to no negative history).
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == tenant_id
    # Default behavior for empty history is Low Risk (High Trust)
    assert data["risk_band"] == "LOW"

def test_batch_tenant_risk(client):
    """Test batch risk scoring."""
    tenant_ids = [50, 51, 52]
    # TestClient handles list params correctly
    response = client.post("/tenant-risk/batch", params={"tenant_ids": tenant_ids})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["tenant_id"] == 50
