"""
Dynamic Pricing API for Airbnb Listings in Morocco
===================================================

FastAPI microservice for real-time nightly price predictions using RandomForest/XGBoost models.
Supports individual predictions and batch processing.

Author: AI-Powered Rental Platform
Version: 2.1.0
Model: Pricing Model (RandomForest/XGBoost) managed by ModelManager
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

from deployment.config import settings
from deployment.model_manager import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'stay_length_nights', 'discount_rate', 'bedroom_count', 'bed_count',
    'rating_value', 'rating_count', 'image_count', 'badge_count',
    'review_density', 'quality_proxy', 'city', 'season_category'
]

VALID_CITIES = ['casablanca', 'marrakech', 'agadir', 'rabat', 'fes', 'tangier']
VALID_SEASONS = ['march', 'april', 'summer', 'other']


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Pre-warm models on startup (optional, but good for health checks)
    try:
        model_manager.get_pricing_model()
        logger.info("✅ Pricing model pre-loaded")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-load pricing model: {e}")
        
    yield
    
    logger.info("🛑 Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered nightly price predictions for Airbnb listings across Moroccan cities",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS is handled at the API Gateway; avoid duplicate headers here

# Include routers for AI features
from deployment.routers import tenant_risk, recommendations, market_trends

app.include_router(tenant_risk.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(market_trends.router, prefix="/api")


class ListingFeatures(BaseModel):
    """Input features for a single listing prediction."""
    
    stay_length_nights: int = Field(..., ge=1, description="Length of stay in nights")
    discount_rate: float = Field(0.0, ge=0.0, le=1.0, description="Discount rate (0.0-1.0)")
    bedroom_count: float = Field(..., ge=0.0, description="Number of bedrooms")
    bed_count: float = Field(..., ge=0.0, description="Number of beds")
    rating_value: float = Field(0.0, ge=0.0, le=5.0, description="Average rating (0.0-5.0)")
    rating_count: int = Field(0, ge=0, description="Number of reviews")
    image_count: int = Field(0, ge=0, description="Number of property images")
    badge_count: int = Field(0, ge=0, description="Total number of badges")
    review_density: float = Field(0.0, ge=0.0, description="Review density metric")
    quality_proxy: float = Field(0.0, ge=0.0, description="Overall quality score")
    city: str = Field(..., description="City name (casablanca, marrakech, agadir, rabat, fes, tangier)")
    season_category: str = Field(..., description="Season (march, april, summer, other)")
    
    model_config = ConfigDict(protected_namespaces=())
    
    @field_validator('city')
    @classmethod
    def validate_city(cls, v):
        if v.lower() not in VALID_CITIES:
            raise ValueError(f"City must be one of: {', '.join(VALID_CITIES)}")
        return v.lower()
    
    @field_validator('season_category')
    @classmethod
    def validate_season(cls, v):
        if v.lower() not in VALID_SEASONS:
            raise ValueError(f"Season must be one of: {', '.join(VALID_SEASONS)}")
        return v.lower()


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    predicted_price_mad: float = Field(..., description="Predicted nightly price in MAD")
    predicted_price_usd: float = Field(..., description="Predicted nightly price in USD (approx)")
    confidence_interval_lower: float = Field(..., description="Lower bound (predicted - MAE)")
    confidence_interval_upper: float = Field(..., description="Upper bound (predicted + MAE)")
    city: str = Field(..., description="Input city")
    season: str = Field(..., description="Input season")
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    listings: List[ListingFeatures] = Field(..., max_length=100, description="List of listings (max 100)")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse]
    total_listings: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    models_status: Dict[str, str]
    timestamp: str


def prepare_features(listing: ListingFeatures) -> pd.DataFrame:
    """
    Convert ListingFeatures to DataFrame using ModelManager's known feature names.
    """
    model_feature_names = model_manager.get_feature_names("pricing")

    # Base features
    base_data = {
        'stay_length_nights': listing.stay_length_nights,
        'discount_rate': listing.discount_rate,
        'bedroom_count': listing.bedroom_count,
        'bed_count': listing.bed_count,
        'rating_value': listing.rating_value,
        'rating_count': listing.rating_count,
        'image_count': listing.image_count,
        'badge_count': listing.badge_count,
        'review_density': listing.review_density,
        'quality_proxy': listing.quality_proxy,
        'city': listing.city,
        'season_category': listing.season_category,
    }

    if not model_feature_names:
        # Fallback if feature names not found in model metadata
        df = pd.DataFrame([base_data])
        df['city'] = df['city'].astype('category')
        df['season_category'] = df['season_category'].astype('category')
        return df[FEATURE_COLS]

    # Align with model expectations
    row: Dict[str, Any] = {}
    city_norm = listing.city.lower()
    season_norm = listing.season_category.lower()

    for fname in model_feature_names:
        lname = fname.lower()

        if fname in base_data and isinstance(base_data[fname], (int, float)):
            row[fname] = float(base_data[fname])
            continue

        if lname.startswith("city_"):
            city_suffix = fname.split("_", 1)[1].lower() if "_" in fname else ""
            row[fname] = 1.0 if city_suffix == city_norm else 0.0
            continue

        if lname.startswith("season_") or lname.startswith("season_category_"):
            season_suffix = fname.split("_", 1)[-1].lower()
            row[fname] = 1.0 if season_suffix == season_norm else 0.0
            continue

        if lname == "beds_per_bedroom":
            try:
                bedrooms = float(base_data.get("bedroom_count", 1.0) or 1.0)
                beds = float(base_data.get("bed_count", 1.0) or 1.0)
                row[fname] = beds / max(bedrooms, 1.0)
            except Exception:
                row[fname] = 1.0
            continue

        row[fname] = 0.0

    df = pd.DataFrame([row])
    return df[model_feature_names]


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for monitoring."""
    status_map = model_manager.health_check()
    overall = "healthy" if status_map.get("pricing") == "loaded" else "degraded"
    
    return HealthResponse(
        status=overall,
        models_status=status_map,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/model-info", tags=["Monitoring"])
async def model_info():
    """Get detailed model information and metadata."""
    
    model = model_manager.get_pricing_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Pricing Model not available")
    
    metadata = model_manager.get_metadata("pricing")
    
    return {
        "model_metadata": metadata,
        "feature_columns": FEATURE_COLS,
        "valid_cities": VALID_CITIES,
        "model_type": type(model).__name__
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(listing: ListingFeatures):
    """Predict nightly price for a single listing."""
    
    model = model_manager.get_pricing_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Pricing model not yet loaded")
    
    try:
        X = prepare_features(listing)
        prediction = model.predict(X)[0]
        
        # Metadata access
        metadata = model_manager.get_metadata("pricing")
        mae = metadata.get('test_mae', 84.59)
        
        ci_lower = max(0, prediction - mae)
        ci_upper = prediction + mae
        prediction_usd = prediction * 0.10
        
        return PredictionResponse(
            predicted_price_mad=round(prediction, 2),
            predicted_price_usd=round(prediction_usd, 2),
            confidence_interval_lower=round(ci_lower, 2),
            confidence_interval_upper=round(ci_upper, 2),
            city=listing.city,
            season=listing.season_category,
            model_version=str(metadata.get('version', '2.0')),
            prediction_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """Predict prices for multiple listings in batch."""
    
    model = model_manager.get_pricing_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Pricing model not yet loaded")
    
    start_time = datetime.now(timezone.utc)
    predictions = []
    metadata = model_manager.get_metadata("pricing")
    mae = metadata.get('test_mae', 84.59)
    
    try:
        for listing in request.listings:
            X = prepare_features(listing)
            prediction = model.predict(X)[0]
            
            ci_lower = max(0, prediction - mae)
            ci_upper = prediction + mae
            prediction_usd = prediction * 0.10
            
            predictions.append(
                PredictionResponse(
                    predicted_price_mad=round(prediction, 2),
                    predicted_price_usd=round(prediction_usd, 2),
                    confidence_interval_lower=round(ci_lower, 2),
                    confidence_interval_upper=round(ci_upper, 2),
                    city=listing.city,
                    season=listing.season_category,
                    model_version=str(metadata.get('version', '2.0')),
                    prediction_timestamp=datetime.now(timezone.utc).isoformat()
                )
            )
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_listings=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/city-insights/{city}", tags=["Analytics"])
async def city_insights(city: str):
    """Get city-specific pricing insights."""
    city = city.lower()
    if city not in VALID_CITIES:
        raise HTTPException(status_code=400, detail=f"Invalid city. Must be one of: {', '.join(VALID_CITIES)}")
    
    # Static insights for now - could be dynamic later
    insights = {
        "casablanca": {
            "avg_price_mad": 49259,
            "market_type": "Business Hub",
            "key_drivers": [
                {"feature": "city_premium", "impact_mad": 4652, "rank": 1},
                {"feature": "dist_to_center", "impact_mad": 996, "rank": 2},
                {"feature": "rating_density", "impact_mad": 874, "rank": 3}
            ],
            "recommendations": [
                "Focus on building rating density",
                "Target 5.0 ratings for +8.2% price uplift",
                "Don't over-invest in prime downtown locations",
            ]
        },
        "marrakech": {
            "avg_price_mad": 46685,
            "market_type": "Tourist Destination",
            "key_drivers": [
                {"feature": "dist_to_center", "impact_mad": 1791, "rank": 1},
                {"feature": "struct_surface_m2", "impact_mad": 1091, "rank": 2}
            ],
            "recommendations": [
                "Location is KING: Prime spots add +4.9% premium",
                "Invest in property size and character",
                "Perfect 5.0 rating = +15.8% uplift"
            ]
        }
    }
    
    # Return default generic if city not detailed above, or exact match
    return insights.get(city, {
        "avg_price_mad": 40000,
        "market_type": "General",
        "key_drivers": [],
        "recommendations": ["Maintain high ratings", "Optimize listing photos"]
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )
