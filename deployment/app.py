"""
Dynamic Pricing API for Airbnb Listings in Morocco
===================================================

FastAPI microservice for real-time nightly price predictions using the trained
GradientBoosting model. Supports individual predictions and batch processing.

Author: AI-Powered Rental Platform
Version: 1.0.0
Model: pricing_gradient_boosting_v1.pkl
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Morocco Airbnb Dynamic Pricing API",
    description="AI-powered nightly price predictions for Airbnb listings across Moroccan cities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and metadata
MODEL = None
MODEL_METADATA = {}
FEATURE_COLS = [
    'city', 'period', 'geo_cluster',
    'rating_value', 'rating_count', 'rating_density', 'has_rating',
    'badge_superhost', 'badge_guest_favorite', 'badge_top_x', 'any_badge',
    'dist_to_center',
    'struct_bedrooms', 'struct_bathrooms', 'struct_surface_m2'
]

VALID_CITIES = ['casablanca', 'marrakech', 'agadir', 'rabat', 'fes', 'tangier']
VALID_PERIODS = ['march', 'april', 'summer']


# Pydantic models for request/response validation
class ListingFeatures(BaseModel):
    """Input features for a single listing prediction."""
    
    city: str = Field(..., description="City name (casablanca, marrakech, agadir, rabat, fes, tangier)")
    period: str = Field(..., description="Booking period (march, april, summer)")
    geo_cluster: int = Field(..., ge=-1, le=4, description="Geographic cluster ID (0-4, -1 for unknown)")
    rating_value: float = Field(..., ge=0.0, le=5.0, description="Average rating (0.0-5.0)")
    rating_count: int = Field(..., ge=0, description="Number of reviews")
    rating_density: float = Field(..., ge=0.0, description="Relative rating density vs city average")
    has_rating: int = Field(..., ge=0, le=1, description="Has any reviews (0 or 1)")
    badge_superhost: bool = Field(..., description="Has Superhost badge")
    badge_guest_favorite: bool = Field(..., description="Has Guest Favorite badge")
    badge_top_x: bool = Field(..., description="Has Top X% badge")
    any_badge: int = Field(..., ge=0, le=1, description="Has any badge (0 or 1)")
    dist_to_center: float = Field(..., ge=0.0, le=1.0, description="Distance to city center (normalized 0-1)")
    struct_bedrooms: Optional[float] = Field(2.0, ge=0.0, description="Number of bedrooms")
    struct_bathrooms: Optional[float] = Field(1.0, ge=0.0, description="Number of bathrooms")
    struct_surface_m2: Optional[float] = Field(80.0, ge=0.0, description="Surface area in m²")
    
    model_config = ConfigDict(protected_namespaces=())
    
    @field_validator('city')
    @classmethod
    def validate_city(cls, v):
        if v.lower() not in VALID_CITIES:
            raise ValueError(f"City must be one of: {', '.join(VALID_CITIES)}")
        return v.lower()
    
    @field_validator('period')
    @classmethod
    def validate_period(cls, v):
        if v.lower() not in VALID_PERIODS:
            raise ValueError(f"Period must be one of: {', '.join(VALID_PERIODS)}")
        return v.lower()


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    predicted_price_mad: float = Field(..., description="Predicted nightly price in MAD")
    predicted_price_usd: float = Field(..., description="Predicted nightly price in USD (approx)")
    confidence_interval_lower: float = Field(..., description="Lower bound (predicted - MAE)")
    confidence_interval_upper: float = Field(..., description="Upper bound (predicted + MAE)")
    city: str = Field(..., description="Input city")
    period: str = Field(..., description="Input period")
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
    model_loaded: bool
    model_version: str
    model_mae: float
    model_mape: float
    timestamp: str


# Startup event: Load model
@app.on_event("startup")
async def load_model():
    """Load the trained model and metadata on startup."""
    global MODEL, MODEL_METADATA
    
    try:
        # Try multiple possible paths for the model file
        # This handles both running from project root and from deployment directory
        possible_model_paths = [
            Path("models/pricing_gradient_boosting_v1.pkl"),  # From project root
            Path("../models/pricing_gradient_boosting_v1.pkl"),  # From deployment directory
            Path(__file__).parent.parent / "models" / "pricing_gradient_boosting_v1.pkl"  # Absolute from this file
        ]
        
        model_path = None
        for path in possible_model_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model file not found in any of: {possible_model_paths}")
        
        # Same logic for metrics path
        possible_metrics_paths = [
            Path("models/model_metrics_v1.csv"),
            Path("../models/model_metrics_v1.csv"),
            Path(__file__).parent.parent / "models" / "model_metrics_v1.csv"
        ]
        
        metrics_path = None
        for path in possible_metrics_paths:
            if path.exists():
                metrics_path = path
                break
        
        # Load model
        MODEL = joblib.load(model_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")
        
        # Load metadata
        if metrics_path is not None:
            metrics_df = pd.read_csv(metrics_path)
            MODEL_METADATA = metrics_df.iloc[0].to_dict()
            logger.info(f"✅ Model metadata loaded: MAE={MODEL_METADATA.get('mae_mad', 'N/A'):.2f} MAD")
        else:
            logger.warning("⚠️ Metrics file not found. Using default metadata.")
            MODEL_METADATA = {
                'model': 'GradientBoosting',
                'version': '1.0',
                'mae_mad': 2332.13,
                'mape_pct': 5.73
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


# Helper function: Prepare features for prediction
def prepare_features(listing: ListingFeatures) -> pd.DataFrame:
    """Convert ListingFeatures to DataFrame with proper dtypes."""
    
    data = {
        'city': listing.city,
        'period': listing.period,
        'geo_cluster': listing.geo_cluster,
        'rating_value': listing.rating_value,
        'rating_count': listing.rating_count,
        'rating_density': listing.rating_density,
        'has_rating': listing.has_rating,
        'badge_superhost': listing.badge_superhost,
        'badge_guest_favorite': listing.badge_guest_favorite,
        'badge_top_x': listing.badge_top_x,
        'any_badge': listing.any_badge,
        'dist_to_center': listing.dist_to_center,
        'struct_bedrooms': listing.struct_bedrooms,
        'struct_bathrooms': listing.struct_bathrooms,
        'struct_surface_m2': listing.struct_surface_m2
    }
    
    df = pd.DataFrame([data])
    
    # Set categorical dtypes
    df['city'] = df['city'].astype('category')
    df['period'] = df['period'].astype('category')
    
    return df


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Morocco Airbnb Dynamic Pricing API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_version=MODEL_METADATA.get('version', '1.0'),
        model_mae=MODEL_METADATA.get('mae_mad', 0.0),
        model_mape=MODEL_METADATA.get('mape_pct', 0.0),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/model-info", tags=["Monitoring"])
async def model_info():
    """Get detailed model information and metadata."""
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_metadata": MODEL_METADATA,
        "feature_columns": FEATURE_COLS,
        "valid_cities": VALID_CITIES,
        "valid_periods": VALID_PERIODS,
        "model_pipeline": {
            "preprocessing": "ColumnTransformer (StandardScaler + OneHotEncoder)",
            "algorithm": "GradientBoostingRegressor",
            "hyperparameters": {
                "learning_rate": 0.1,
                "max_depth": 5,
                "n_estimators": 200,
                "subsample": 0.8
            }
        },
        "performance": {
            "mae_mad": MODEL_METADATA.get('mae_mad', 'N/A'),
            "mape_pct": MODEL_METADATA.get('mape_pct', 'N/A'),
            "train_size": MODEL_METADATA.get('train_size', 'N/A'),
            "val_size": MODEL_METADATA.get('val_size', 'N/A')
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(listing: ListingFeatures):
    """
    Predict nightly price for a single listing.
    
    Returns predicted price in MAD with confidence intervals based on model MAE.
    """
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        X = prepare_features(listing)
        
        # Make prediction
        prediction = MODEL.predict(X)[0]
        
        # Calculate confidence interval (predicted ± MAE)
        mae = MODEL_METADATA.get('mae_mad', 2332.13)
        ci_lower = max(0, prediction - mae)
        ci_upper = prediction + mae
        
        # Convert to USD (approximate rate: 1 MAD ≈ 0.10 USD)
        prediction_usd = prediction * 0.10
        
        return PredictionResponse(
            predicted_price_mad=round(prediction, 2),
            predicted_price_usd=round(prediction_usd, 2),
            confidence_interval_lower=round(ci_lower, 2),
            confidence_interval_upper=round(ci_upper, 2),
            city=listing.city,
            period=listing.period,
            model_version=MODEL_METADATA.get('version', '1.0'),
            prediction_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict prices for multiple listings in batch.
    
    Maximum 100 listings per request for performance.
    """
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now(timezone.utc)
    predictions = []
    
    try:
        for listing in request.listings:
            # Prepare features
            X = prepare_features(listing)
            
            # Make prediction
            prediction = MODEL.predict(X)[0]
            
            # Calculate confidence interval
            mae = MODEL_METADATA.get('mae_mad', 2332.13)
            ci_lower = max(0, prediction - mae)
            ci_upper = prediction + mae
            
            # Convert to USD
            prediction_usd = prediction * 0.10
            
            predictions.append(
                PredictionResponse(
                    predicted_price_mad=round(prediction, 2),
                    predicted_price_usd=round(prediction_usd, 2),
                    confidence_interval_lower=round(ci_lower, 2),
                    confidence_interval_upper=round(ci_upper, 2),
                    city=listing.city,
                    period=listing.period,
                    model_version=MODEL_METADATA.get('version', '1.0'),
                    prediction_timestamp=datetime.now(timezone.utc).isoformat()
                )
            )
        
        # Calculate processing time
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
    """
    Get city-specific pricing insights and recommendations.
    
    Based on SHAP analysis and sensitivity testing.
    """
    
    city = city.lower()
    if city not in VALID_CITIES:
        raise HTTPException(status_code=400, detail=f"Invalid city. Must be one of: {', '.join(VALID_CITIES)}")
    
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
                "Focus on building rating density (get more reviews than competitors)",
                "Target 5.0 ratings for +8.2% price uplift (~4,000 MAD)",
                "Don't over-invest in prime downtown locations (paradoxical -7.2% impact)",
                "Superhost badge has minimal value (-0.8%)",
                "Modern mid-distance apartments outperform large central properties"
            ],
            "badge_superhost_impact_pct": -0.8,
            "perfect_rating_uplift_pct": 8.2,
            "prime_location_impact_pct": -7.2
        },
        "marrakech": {
            "avg_price_mad": 46685,
            "market_type": "Tourist Destination",
            "key_drivers": [
                {"feature": "dist_to_center", "impact_mad": 1791, "rank": 1},
                {"feature": "struct_surface_m2", "impact_mad": 1091, "rank": 2},
                {"feature": "rating_density", "impact_mad": 894, "rank": 3}
            ],
            "recommendations": [
                "Location is KING: Prime spots add +4.9% premium (~2,000 MAD)",
                "Invest in property size and character (1,091 MAD impact)",
                "Perfect 5.0 rating = +15.8% uplift (~6,350 MAD)",
                "Highlight authentic riad features and Medina proximity",
                "Size and character matter here unlike other cities"
            ],
            "badge_superhost_impact_pct": -0.9,
            "perfect_rating_uplift_pct": 15.8,
            "prime_location_impact_pct": 4.9
        },
        "agadir": {
            "avg_price_mad": 41694,
            "market_type": "Beach Resort",
            "key_drivers": [
                {"feature": "dist_to_center", "impact_mad": 1870, "rank": 1},
                {"feature": "rating_density", "impact_mad": 1229, "rank": 2},
                {"feature": "rating_value", "impact_mad": 877, "rank": 3}
            ],
            "recommendations": [
                "HIGHEST rating ROI: Perfect 5.0 = +19.0% uplift (~7,200 MAD)",
                "Build rating density aggressively (1,229 MAD impact)",
                "Superhost badge adds value here (+0.5% - only positive city)",
                "Emphasize beach proximity over city center",
                "Trust signals are critical for resort properties"
            ],
            "badge_superhost_impact_pct": 0.5,
            "perfect_rating_uplift_pct": 19.0,
            "prime_location_impact_pct": 2.0
        },
        "rabat": {
            "avg_price_mad": 40959,
            "market_type": "Capital/Administrative",
            "key_drivers": [
                {"feature": "dist_to_center", "impact_mad": 2279, "rank": 1},
                {"feature": "rating_density", "impact_mad": 1343, "rank": 2},
                {"feature": "rating_value", "impact_mad": 1166, "rank": 3}
            ],
            "recommendations": [
                "CRITICAL: Obtain Superhost badge (+5.5% = ~2,135 MAD - highest impact!)",
                "Perfect ratings yield +18.8% premium (~7,300 MAD)",
                "Professional travelers value credentials over location glamour",
                "Avoid chasing administrative center locations (-8.2% paradox)",
                "Emphasize reliability, consistency, business amenities"
            ],
            "badge_superhost_impact_pct": 5.5,
            "perfect_rating_uplift_pct": 18.8,
            "prime_location_impact_pct": -8.2
        },
        "fes": {
            "avg_price_mad": 37000,
            "market_type": "Cultural Heritage",
            "key_drivers": [
                {"feature": "city_fes", "impact_mad": 500, "rank": 1},
                {"feature": "rating_value", "impact_mad": 450, "rank": 2},
                {"feature": "dist_to_center", "impact_mad": 400, "rank": 3}
            ],
            "recommendations": [
                "Focus on cultural authenticity and heritage value",
                "Medina proximity important for tourists",
                "Maintain strong ratings for competitive positioning",
                "Smaller market - consistency matters"
            ],
            "badge_superhost_impact_pct": 0.0,
            "perfect_rating_uplift_pct": 12.0,
            "prime_location_impact_pct": 3.0
        },
        "tangier": {
            "avg_price_mad": 42000,
            "market_type": "Port City/Gateway",
            "key_drivers": [
                {"feature": "rating_value", "impact_mad": 600, "rank": 1},
                {"feature": "dist_to_center", "impact_mad": 550, "rank": 2},
                {"feature": "rating_density", "impact_mad": 500, "rank": 3}
            ],
            "recommendations": [
                "Balance tourist and business traveler needs",
                "Rating quality drives competitive advantage",
                "Port proximity and ferry access important",
                "Moderate sensitivity across features"
            ],
            "badge_superhost_impact_pct": 1.0,
            "perfect_rating_uplift_pct": 14.0,
            "prime_location_impact_pct": 2.5
        }
    }
    
    return insights.get(city, {"error": "City insights not available"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
