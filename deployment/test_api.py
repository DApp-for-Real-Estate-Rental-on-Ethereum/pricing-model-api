"""
Test suite for Morocco Airbnb Dynamic Pricing API
"""

import pytest
from fastapi.testclient import TestClient
from app import app

# TestClient automatically triggers lifespan events (startup/shutdown)
# No manual model loading needed anymore
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Morocco Airbnb Dynamic Pricing API"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_model_info():
    """Test model information endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_metadata" in data
    assert "feature_columns" in data
    assert "performance" in data


def test_single_prediction_valid():
    """Test single prediction with valid input."""
    listing = {
        "city": "casablanca",
        "period": "summer",
        "geo_cluster": 1,
        "rating_value": 4.8,
        "rating_count": 50,
        "rating_density": 1.0,
        "has_rating": 1,
        "badge_superhost": False,
        "badge_guest_favorite": False,
        "badge_top_x": False,
        "any_badge": 0,
        "dist_to_center": 0.05,
        "struct_bedrooms": 2.0,
        "struct_bathrooms": 1.0,
        "struct_surface_m2": 80.0
    }
    
    response = client.post("/predict", json=listing)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_mad" in data
    assert "confidence_interval_lower" in data
    assert "confidence_interval_upper" in data
    assert data["predicted_price_mad"] > 0
    assert data["city"] == "casablanca"


def test_prediction_invalid_city():
    """Test prediction with invalid city."""
    listing = {
        "city": "paris",  # Invalid city
        "period": "summer",
        "geo_cluster": 1,
        "rating_value": 4.8,
        "rating_count": 50,
        "rating_density": 1.0,
        "has_rating": 1,
        "badge_superhost": False,
        "badge_guest_favorite": False,
        "badge_top_x": False,
        "any_badge": 0,
        "dist_to_center": 0.05,
        "struct_bedrooms": 2.0,
        "struct_bathrooms": 1.0,
        "struct_surface_m2": 80.0
    }
    
    response = client.post("/predict", json=listing)
    assert response.status_code == 422  # Validation error


def test_batch_prediction():
    """Test batch prediction endpoint."""
    listings = {
        "listings": [
            {
                "city": "casablanca",
                "period": "summer",
                "geo_cluster": 1,
                "rating_value": 4.8,
                "rating_count": 50,
                "rating_density": 1.0,
                "has_rating": 1,
                "badge_superhost": True,
                "badge_guest_favorite": False,
                "badge_top_x": False,
                "any_badge": 1,
                "dist_to_center": 0.05,
                "struct_bedrooms": 2.0,
                "struct_bathrooms": 1.0,
                "struct_surface_m2": 80.0
            },
            {
                "city": "marrakech",
                "period": "march",
                "geo_cluster": 0,
                "rating_value": 5.0,
                "rating_count": 100,
                "rating_density": 2.0,
                "has_rating": 1,
                "badge_superhost": True,
                "badge_guest_favorite": True,
                "badge_top_x": False,
                "any_badge": 1,
                "dist_to_center": 0.01,
                "struct_bedrooms": 3.0,
                "struct_bathrooms": 2.0,
                "struct_surface_m2": 120.0
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


def test_city_insights_casablanca():
    """Test city insights for Casablanca."""
    response = client.get("/city-insights/casablanca")
    assert response.status_code == 200
    data = response.json()
    assert data["market_type"] == "Business Hub"
    assert "recommendations" in data
    assert "key_drivers" in data
    assert len(data["recommendations"]) > 0


def test_city_insights_invalid():
    """Test city insights with invalid city."""
    response = client.get("/city-insights/invalid-city")
    assert response.status_code == 400


def test_prediction_confidence_intervals():
    """Test that confidence intervals are reasonable."""
    listing = {
        "city": "agadir",
        "period": "summer",
        "geo_cluster": 2,
        "rating_value": 4.5,
        "rating_count": 25,
        "rating_density": 0.8,
        "has_rating": 1,
        "badge_superhost": False,
        "badge_guest_favorite": False,
        "badge_top_x": False,
        "any_badge": 0,
        "dist_to_center": 0.1,
        "struct_bedrooms": 1.0,
        "struct_bathrooms": 1.0,
        "struct_surface_m2": 50.0
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
    assert (ci_upper - ci_lower) < 10000  # Interval shouldn't be too wide


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
