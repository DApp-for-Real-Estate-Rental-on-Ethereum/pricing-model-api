"""
Test suite for Tenant Risk Scoring API
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
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
    #@patch('deployment.routers.tenant_risk.execute_query_single')
def test_tenant_risk_score_valid(mock_db, client):
    """Test standard valid tenant risk request with mocked DB"""
    # Mock return values for the 4 DB calls: user, bookings, reclamations, transactions
    mock_db.side_effect = [
        # 1. User query
        {'rating': 4.8, 'score': 100, 'penalty_points': 0, 'is_suspended': False, 'is_complete': True, 'verification_expiration': '2025-01-01T00:00:00Z'},
        # 2. Bookings query
        {'total': 10, 'completed': 8, 'cancelled': 2, 'avg_price': 500.0, 'avg_stay_days': 5.0, 'recent_6m': 5},
        # 3. Reclamations query
        {'total': 0, 'low': 0, 'medium': 0, 'high': 0, 'critical': 0, 'open': 0, 'resolved_against': 0, 'total_penalty': 0, 'total_refund': 0.0},
        # 4. Transactions query
        {'total': 10, 'success': 10, 'failed': 0, 'avg_amount': 500.0}
    ]
    
    response = client.post("/tenant-risk/50")
    assert response.status_code == 200
    data = response.json()
    assert "trust_score" in data
    assert "risk_band" in data
    # High score expected due to good mocked data
    assert data['risk_band'] in ["LOW", "MEDIUM"]
    assert data['trust_score'] > 60
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

@patch('deployment.routers.tenant_risk.execute_query_single')
def test_tenant_risk_score_nonexistent(mock_db, client):
    """Test request for nonexistent tenant (DB returns None)"""
    # Mock DB returning empty/None for all queries
    mock_db.return_value = None
    
    response = client.post("/tenant-risk/99999")
    assert response.status_code == 200
    data = response.json()
    # Should fallback to default score (100) or heuristic
    assert data['trust_score'] >= 0  # Just ensure it returns a valid responsenot found,
    # so it should likely return a score (probably 100/LOW due to no negative history).
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == tenant_id
    # Default behavior for empty history is Low Risk (High Trust)

@patch('deployment.routers.tenant_risk.extract_tenant_features')
def test_batch_tenant_risk(mock_extract, client):
    """Test batch risk scoring"""
    # Mock the higher-level extract function to avoid mocking 4 DB calls per tenant
    # We return a dummy TenantRiskFeatures object
    from deployment.routers.tenant_risk import TenantRiskFeatures
    mock_extract.return_value = TenantRiskFeatures()

    tenant_ids = [50, 51, 52]
    # Pass ids as query parameters 'tenant_ids' multiple times
    query_string = "&".join([f"tenant_ids={tid}" for tid in tenant_ids])
    response = client.post(f"/tenant-risk/batch?{query_string}")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]['tenant_id'] in tenant_ids
    assert data[0]["tenant_id"] == 50
