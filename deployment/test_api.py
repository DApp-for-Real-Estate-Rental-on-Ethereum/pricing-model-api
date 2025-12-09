"""
Test suite for Morocco Airbnb Dynamic Pricing API
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="session")
def client():
    """Provide a TestClient that ensures lifespan events run."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Morocco Airbnb Dynamic Pricing API"
    assert data["version"] == "2.1.0"
    assert "endpoints" in data


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_model_info(client):
    """Test model information endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_metadata" in data
    assert "feature_columns" in data
    assert "performance" in data


def test_single_prediction_valid(client):
    """Test single prediction with valid input."""
    listing = {
        "stay_length_nights": 5,
        "discount_rate": 0.1,
        "bedroom_count": 2.0,
        "bed_count": 3.0,
        "rating_value": 4.8,
        "rating_count": 50,
        "image_count": 15,
        "badge_count": 1,
        "review_density": 0.8,
        "quality_proxy": 0.75,
        "city": "casablanca",
        "season_category": "summer"
    }
    
    response = client.post("/predict", json=listing)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_mad" in data
    assert "confidence_interval_lower" in data
    assert "confidence_interval_upper" in data
    assert data["predicted_price_mad"] > 0
    assert data["city"] == "casablanca"


def test_prediction_invalid_city(client):
    """Test prediction with invalid city."""
    listing = {
        "stay_length_nights": 5,
        "discount_rate": 0.1,
        "bedroom_count": 2.0,
        "bed_count": 3.0,
        "rating_value": 4.8,
        "rating_count": 50,
        "image_count": 15,
        "badge_count": 1,
        "review_density": 0.8,
        "quality_proxy": 0.75,
        "city": "paris",  # Invalid city
        "season_category": "summer"
    }
    
    response = client.post("/predict", json=listing)
    assert response.status_code == 422  # Validation error


def test_batch_prediction(client):
    """Test batch prediction endpoint."""
    listings = {
        "listings": [
            {
                "stay_length_nights": 5,
                "discount_rate": 0.1,
                "bedroom_count": 2.0,
                "bed_count": 3.0,
                "rating_value": 4.8,
                "rating_count": 50,
                "image_count": 15,
                "badge_count": 1,
                "review_density": 0.8,
                "quality_proxy": 0.75,
                "city": "casablanca",
                "season_category": "summer"
            },
            {
                "stay_length_nights": 3,
                "discount_rate": 0.0,
                "bedroom_count": 3.0,
                "bed_count": 4.0,
                "rating_value": 5.0,
                "rating_count": 100,
                "image_count": 20,
                "badge_count": 2,
                "review_density": 1.2,
                "quality_proxy": 0.90,
                "city": "marrakech",
                "season_category": "march"
            }
        ]
    }
    
    response = client.post("/batch-predict", json=listings)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert data["total_listings"] == 2
    assert "processing_time_ms" in data


def test_city_insights_casablanca(client):
    """Test city insights for Casablanca."""
    response = client.get("/city-insights/casablanca")
    assert response.status_code == 200
    data = response.json()
    assert data["market_type"] == "Business Hub"
    assert "recommendations" in data
    assert "key_drivers" in data
    assert len(data["recommendations"]) > 0


def test_city_insights_invalid(client):
    """Test city insights with invalid city."""
    response = client.get("/city-insights/invalid-city")
    assert response.status_code == 400


def test_prediction_confidence_intervals(client):
    """Test that confidence intervals are reasonable."""
    listing = {
        "stay_length_nights": 4,
        "discount_rate": 0.0,
        "bedroom_count": 1.0,
        "bed_count": 2.0,
        "rating_value": 4.5,
        "rating_count": 25,
        "image_count": 10,
        "badge_count": 0,
        "review_density": 0.5,
        "quality_proxy": 0.65,
        "city": "agadir",
        "season_category": "summer"
    }
    
    response = client.post("/predict", json=listing)
    assert response.status_code == 200
    data = response.json()
    
    # Check confidence intervals are reasonable
    predicted = data["predicted_price_mad"]
    ci_lower = data["confidence_interval_lower"]
    ci_upper = data["confidence_interval_upper"]
    
    assert ci_lower < predicted < ci_upper
    assert ci_lower >= 0  # Price can't be negative
    assert (ci_upper - ci_lower) < 1000  # Interval shouldn't be too wide (MAE ~85 for RandomForest)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
