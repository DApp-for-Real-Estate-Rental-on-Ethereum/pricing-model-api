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
    """Test standard valid tenant risk request with mocked DB"""
    tenant_id = 50
    
    # Mock return values for the 4 DB calls: user, bookings, reclamations, transactions
    mock_side_effect = [
        # 1. User query
        {'rating': 4.8, 'score': 100, 'penalty_points': 0, 'is_suspended': False, 'is_complete': True, 'verification_expiration': '2025-01-01T00:00:00Z'},
        # 2. Bookings query
        {'total': 10, 'completed': 8, 'cancelled': 2, 'avg_price': 500.0, 'avg_stay_days': 5.0, 'recent_6m': 5},
        # 3. Reclamations query
        {'total': 0, 'low': 0, 'medium': 0, 'high': 0, 'critical': 0, 'open': 0, 'resolved_against': 0, 'total_penalty': 0, 'total_refund': 0.0},
        # 4. Transactions query
        {'total': 10, 'success': 10, 'failed': 0, 'avg_amount': 500.0}
    ]
    
    with patch('deployment.routers.tenant_risk.execute_query_single') as mock_db, \
         patch('deployment.routers.tenant_risk.predict_risk_score') as mock_predict:
        
        mock_db.side_effect = mock_side_effect
        
        # Mock the prediction to return a known "good" result
        mock_predict.return_value = {
            'trust_score': 85,
            'risk_band': 'LOW',
            'risk_probability': 0.15,
            'top_factors': ['Good history']
        }
        
        response = client.post(f"/tenant-risk/{tenant_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "trust_score" in data
        assert "risk_band" in data
        # Now we can safely assert LOW because we forced it
        assert data['risk_band'] == "LOW"
        assert data['trust_score'] == 85
        assert data["tenant_id"] == tenant_id
        
        # Check structure
        assert "risk_probability" in data
        assert "features" in data
        assert "top_factors" in data

def test_tenant_risk_score_nonexistent(client):
    """Test request for nonexistent tenant (DB returns None)"""
    tenant_id = 99999
    
    with patch('deployment.routers.tenant_risk.execute_query_single') as mock_db, \
         patch('deployment.routers.tenant_risk.predict_risk_score') as mock_predict:
        # Mock DB returning empty/None for all queries
        mock_db.return_value = None
        
        # Mock the prediction to return a default "safe" result
        mock_predict.return_value = {
            'trust_score': 90,
            'risk_band': 'LOW',
            'risk_probability': 0.1,
            'top_factors': ['No negative history']
        }
        
        response = client.post(f"/tenant-risk/{tenant_id}")
        assert response.status_code == 200
        data = response.json()
        
        # Should match our mocked prediction
        assert data['trust_score'] == 90
        assert data["tenant_id"] == tenant_id
        assert data["risk_band"] == "LOW"

def test_batch_tenant_risk(client):
    """Test batch risk scoring"""
    tenant_ids = [50, 51, 52]
    
    with patch('deployment.routers.tenant_risk.extract_tenant_features') as mock_extract:
        # Mock the higher-level extract function to avoid mocking 4 DB calls per tenant
        # We return a dummy TenantRiskFeatures object
        from deployment.routers.tenant_risk import TenantRiskFeatures
        mock_extract.return_value = TenantRiskFeatures()
    
        # Pass ids as query parameters 'tenant_ids' multiple times
        query_string = "&".join([f"tenant_ids={tid}" for tid in tenant_ids])
        response = client.post(f"/tenant-risk/batch?{query_string}")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        # Response order is not guaranteed, but usually sequential. Check if ID exists in list.
        assert data[0]['tenant_id'] in tenant_ids
