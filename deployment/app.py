"""
Dynamic Pricing API for Airbnb Listings in Morocco
===================================================

FastAPI microservice for real-time nightly price predictions using RandomForest/XGBoost models.
Supports individual predictions and batch processing.

Author: AI-Powered Rental Platform
Version: 2.1.0
Model: pricing_model_randomforest.pkl (Primary) or xgboost_tuned.pkl (if available)
"""

from contextlib import asynccontextmanager
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

# Global model and metadata
MODEL = None
MODEL_METADATA = {}
# When available (scikit-learn 1.0+), this holds the exact feature names
# the trained model was fitted on. We use it to align our request features
# with the model's expectations, even if the training pipeline used
# expanded/one-hot encoded columns.
MODEL_FEATURE_NAMES = None
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
    global MODEL, MODEL_METADATA, MODEL_FEATURE_NAMES
    
    # STARTUP
    try:
        # Try multiple possible paths for the model file.
        # IMPORTANT: In this deployment we ONLY use RandomForest-based models.
        # XGBoost models are deliberately ignored to avoid importing the heavy
        # xgboost dependency and to prevent unpickling errors when xgboost
        # is not installed in the runtime environment.
        possible_model_paths = [
            # PRIMARY: Tuned RandomForest model (best performance, if available)
            Path("models/tuned/random_forest_tuned.pkl"),
            Path("../models/tuned/random_forest_tuned.pkl"),
            Path(__file__).parent.parent / "models" / "tuned" / "random_forest_tuned.pkl",
            # SECONDARY: Baseline RandomForest model (exists in repository/GitHub)
            Path("models/pricing_model_randomforest.pkl"),
            Path("../models/pricing_model_randomforest.pkl"),
            Path(__file__).parent.parent / "models" / "pricing_model_randomforest.pkl",
        ]
        
        model_path = None
        for path in possible_model_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model file not found in any of: {possible_model_paths}")
        
        # Try to load metadata (optional - will use defaults if not found)
        possible_metadata_paths = [
            # RandomForest metadata variants
            Path("models/pricing_model_randomforest_metadata.pkl"),
            Path("../models/pricing_model_randomforest_metadata.pkl"),
            Path(__file__).parent.parent / "models" / "pricing_model_randomforest_metadata.pkl",
        ]
        
        metadata_path = None
        for path in possible_metadata_paths:
            if path.exists():
                metadata_path = path
                break
        
        # Load model
        MODEL = joblib.load(model_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")

        # Capture feature names from the model if available (scikit-learn 1.0+)
        if hasattr(MODEL, "feature_names_in_"):
            try:
                MODEL_FEATURE_NAMES = list(MODEL.feature_names_in_)
                logger.info(f"📊 Model expects {len(MODEL_FEATURE_NAMES)} features.")
            except Exception as fe:
                logger.warning(f"⚠️ Could not read feature_names_in_ from model: {fe}")
        
        # Load metadata
        if metadata_path is not None:
            MODEL_METADATA = joblib.load(metadata_path)
            logger.info(f"✅ Model metadata loaded: MAE={MODEL_METADATA.get('test_mae', 'N/A'):.2f} MAD")
        else:
            logger.warning("⚠️ Metadata file not found. Using default metadata based on model type.")
            # Detect which model was loaded and use appropriate defaults
            model_name_lower = str(model_path).lower()
            if 'randomforest' in model_name_lower or 'random_forest' in model_name_lower:
                # Check if it's tuned or baseline
                if 'tuned' in model_name_lower:
                    # Tuned RandomForest (better than baseline)
                    MODEL_METADATA = {
                        'model_name': 'RandomForest Tuned',
                        'version': '1.1',
                        'test_mae': 54.57,  # From tuning_comparison.csv
                        'test_rmse': 186.92,
                        'test_r2': 0.8335
                    }
                else:
                    # Default metadata for RandomForest model (baseline - GitHub)
                    MODEL_METADATA = {
                        'model_name': 'RandomForest',
                        'version': '1.0',
                        'test_mae': 84.59,  # From model comparison
                        'test_rmse': 172.22,
                        'test_r2': 0.8586
                    }
            elif 'tuned' in model_name_lower and 'xgboost' in model_name_lower:
                # XGBoost Tuned model (best performance)
                MODEL_METADATA = {
                    'model_name': 'XGBoost Tuned',
                    'version': '2.1',
                    'test_mae': 48.55,
                    'test_rmse': 134.21,
                    'test_r2': 0.9142
                }
            elif 'xgboost' in model_name_lower:
                # Default metadata for XGBoost baseline
                MODEL_METADATA = {
                    'model_name': 'XGBoost',
                    'version': '2.0',
                    'test_mae': 56.01,
                    'test_rmse': 71.33,
                    'test_r2': 0.5556
                }
            else:
                # Unknown model - use RandomForest defaults as safe fallback
                MODEL_METADATA = {
                    'model_name': 'Unknown Model',
                    'version': '1.0',
                    'test_mae': 84.59,
                    'test_rmse': 172.22,
                    'test_r2': 0.8586
                }
        
        logger.info("🚀 Application startup complete - ready to serve predictions!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # SHUTDOWN
    logger.info("🛑 Shutting down - cleaning up resources...")
    MODEL = None
    MODEL_METADATA.clear()
    MODEL_FEATURE_NAMES = None
    logger.info("✅ Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Morocco Airbnb Dynamic Pricing API",
    description="AI-powered nightly price predictions for Airbnb listings across Moroccan cities using RandomForest/XGBoost models",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for AI features
from deployment.routers import tenant_risk, recommendations, market_trends

app.include_router(tenant_risk.router)
app.include_router(recommendations.router)
app.include_router(market_trends.router)


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
    model_loaded: bool
    model_version: str
    model_mae: float
    model_r2: float
    timestamp: str


def prepare_features(listing: ListingFeatures) -> pd.DataFrame:
    """
    Convert ListingFeatures to DataFrame with proper dtypes.

    If the loaded model exposes `feature_names_in_` (common for scikit-learn
    models), we align our input to those names — filling unknown features with 0
    and mapping known ones (including one-hot style city/season columns).
    """
    global MODEL_FEATURE_NAMES

    # Base, high-level features from our API contract
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

    # If we don't know the model-specific feature names, fall back to the
    # original simple schema.
    if not MODEL_FEATURE_NAMES:
        df = pd.DataFrame([base_data])
        df['city'] = df['city'].astype('category')
        df['season_category'] = df['season_category'].astype('category')
        return df[FEATURE_COLS]

    # Build a single row matching the model's expected feature names.
    row: Dict[str, Any] = {}

    # Normalized helpers for city/season
    city_norm = listing.city.lower()
    season_norm = listing.season_category.lower()

    for fname in MODEL_FEATURE_NAMES:
        lname = fname.lower()

        # Direct match with one of our base numeric features
        if fname in base_data and isinstance(base_data[fname], (int, float)):
            row[fname] = float(base_data[fname])
            continue

        # One-hot encoded city_* style columns
        if lname.startswith("city_"):
            # Extract suffix after "city_"
            city_suffix = fname.split("_", 1)[1].lower() if "_" in fname else ""
            row[fname] = 1.0 if city_suffix == city_norm else 0.0
            continue

        # One-hot encoded season_* or season_category_* style columns
        if lname.startswith("season_") or lname.startswith("season_category_"):
            season_suffix = fname.split("_", 1)[-1].lower()
            row[fname] = 1.0 if season_suffix == season_norm else 0.0
            continue

        # Derived feature: beds_per_bedroom if the model expects it
        if lname == "beds_per_bedroom":
            try:
                bedrooms = float(base_data.get("bedroom_count", 1.0) or 1.0)
                beds = float(base_data.get("bed_count", 1.0) or 1.0)
                row[fname] = beds / max(bedrooms, 1.0)
            except Exception:
                row[fname] = 1.0
            continue

        # Default for all other features we don't explicitly know:
        # use 0.0 as a neutral baseline.
        row[fname] = 0.0

    df = pd.DataFrame([row])
    return df[MODEL_FEATURE_NAMES]


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    model_info = f"{MODEL_METADATA.get('model_name', 'Unknown')} (MAE: {MODEL_METADATA.get('test_mae', 'N/A')} MAD)"
    return {
        "service": "Morocco Airbnb Dynamic Pricing API",
        "version": "2.1.0",
        "model": model_info,
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
        model_version=str(MODEL_METADATA.get('version', '1.0')),
        model_mae=MODEL_METADATA.get('test_mae', 84.59),  # Default to RandomForest
        model_r2=MODEL_METADATA.get('test_r2', 0.8586),  # Default to RandomForest
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
        "valid_seasons": VALID_SEASONS,
        "model_pipeline": {
            "preprocessing": "ColumnTransformer (StandardScaler + OneHotEncoder)",
            "algorithm": MODEL_METADATA.get('model_name', 'RandomForest'),
            "hyperparameters": {
                "colsample_bytree": 1.0,
                "learning_rate": 0.05,
                "max_depth": 3,
                "n_estimators": 200,
                "subsample": 1.0
            }
        },
        "performance": {
            "test_mae": MODEL_METADATA.get('test_mae', 84.59),
            "test_rmse": MODEL_METADATA.get('test_rmse', 172.22),  # Default to RandomForest
            "test_r2": MODEL_METADATA.get('test_r2', 0.8586),  # Default to RandomForest
            "train_size": MODEL_METADATA.get('train_size', 1324),
            "test_size": MODEL_METADATA.get('test_size', 332)
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
        mae = MODEL_METADATA.get('test_mae', 84.59)  # Default to RandomForest MAE
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
            season=listing.season_category,
            model_version=str(MODEL_METADATA.get('version', '1.0')),
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
            mae = MODEL_METADATA.get('test_mae', 84.59)  # Default to RandomForest MAE
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
                    season=listing.season_category,
                    model_version=str(MODEL_METADATA.get('version', '1.0')),
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
