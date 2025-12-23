# Morocco Airbnb Dynamic Pricing API

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Python Version](https://img.shields.io/badge/python-3.12-blue)]()
[![Model Version](https://img.shields.io/badge/model-v2.0-orange)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue)]()

AI-powered dynamic pricing microservice for Airbnb listings across Moroccan cities. Predicts optimal nightly prices using a Random Forest model trained on 1,656+ real listings.

## Overview

This API delivers real-time price predictions for short-term rental properties in Morocco's major cities. Built with **FastAPI** and powered by a **Random Forest** machine learning model achieving **55.33 MAD mean absolute error** (~$5.50 USD).

### Key Features

- **Fast Predictions**: Single prediction <50ms, batch processing up to 100 listings
- **High Accuracy**: MAE of 55.33 MAD with R² of 0.5499
- **Multi-City Support**: Casablanca, Marrakech, Agadir, Rabat, Fes, Tangier
- **Seasonal Intelligence**: Dynamic pricing for different seasons
- **Production Ready**: Docker deployment, CI/CD pipeline, comprehensive tests
- **Model Transparency**: Feature importance analysis and confidence intervals

---

## Quick Start

### Using Docker (Recommended)

```bash
# Pull the latest image
docker pull medgm/morocco-pricing-api:latest

# Run the container
docker run -d \
  --name pricing-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  medgm/morocco-pricing-api:latest

# Test the API
curl http://localhost:8000/health
```

### Using Docker Compose

```bash
cd deployment
docker-compose up -d
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api.git
cd pricing-model-api

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r deployment/requirements.txt

# Run the API
cd deployment
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API docs at: **http://localhost:8000/docs**

---

## Model Performance

### Model v2.0 - XGBoost (Current)

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | 56.01 MAD | Mean Absolute Error (~$5.60 USD) |
| **RMSE** | 71.33 MAD | Root Mean Squared Error |
| **R²** | 0.5556 | Explains 55.56% of price variance |
| **Train Size** | 1,324 listings | 80% split |
| **Test Size** | 332 listings | 20% split |
| **Model Size** | 217 KB | Lightweight deployment |

### Model Comparison (Training Phase)

We evaluated 3 candidate models:

| Model | Test MAE | Test RMSE | Test R² | Rank |
|-------|----------|-----------|---------|------|
| **XGBoost** ⭐ | 56.01 | **71.33** | **0.5556** | 🥇 |
| RandomForest | **55.33** | 71.79 | 0.5499 | 🥈 |
| GradientBoosting | 56.32 | 72.03 | 0.5468 | 🥉 |

**Champion Model**: XGBoost selected for best R² (0.5556) and RMSE (71.33) - explains the most variance and handles outliers best. While RandomForest has slightly lower MAE (0.68 MAD difference = $0.07 USD), XGBoost wins on 2 out of 3 metrics.

---

## API Endpoints

### Health Check
```http
GET /health
```
Returns service health status and model metrics.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2.0",
  "model_mae": 55.33,
  "model_r2": 0.5499,
  "timestamp": "2025-11-21T19:45:00Z"
}
```

### Single Prediction
```http
POST /predict
```

**Request Body:**
```json
{
  "stay_length_nights": 5,
  "discount_rate": 0.0,
  "bedroom_count": 2,
  "bed_count": 3,
  "rating_value": 4.8,
  "rating_count": 50,
  "image_count": 15,
  "badge_count": 2,
  "review_density": 1.2,
  "quality_proxy": 0.85,
  "city": "casablanca",
  "season_category": "summer"
}
```

**Response:**
```json
{
  "predicted_price_mad": 425.50,
  "predicted_price_usd": 42.55,
  "confidence_interval_lower": 370.17,
  "confidence_interval_upper": 480.83,
  "city": "casablanca",
  "season": "summer",
  "model_version": "2.0",
  "prediction_timestamp": "2025-11-21T19:45:00Z"
}
```

### Batch Prediction
```http
POST /batch-predict
```

**Request Body:**
```json
{
  "listings": [
    {
      "stay_length_nights": 3,
      "discount_rate": 0.1,
      "bedroom_count": 1,
      "bed_count": 2,
      "rating_value": 4.5,
      "rating_count": 25,
      "image_count": 10,
      "badge_count": 1,
      "review_density": 0.8,
      "quality_proxy": 0.75,
      "city": "marrakech",
      "season_category": "april"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [...],
  "total_listings": 1,
  "processing_time_ms": 45.23
}
```

### Model Information
```http
GET /model-info
```

Returns detailed model metadata, feature schema, and performance metrics.

---

## Model Features

The model uses **12 features** to predict nightly prices:

### Numeric Features (10)

| Feature | Description | Example |
|---------|-------------|---------|
| `stay_length_nights` | Length of stay | 3 |
| `discount_rate` | Applied discount | 0.15 (15%) |
| `bedroom_count` | Number of bedrooms | 2.0 |
| `bed_count` | Number of beds | 3.0 |
| `rating_value` | Average rating | 4.8 |
| `rating_count` | Total reviews | 50 |
| `image_count` | Property photos | 15 |
| `badge_count` | Total badges | 2 |
| `review_density` | Review frequency metric | 1.2 |
| `quality_proxy` | Overall quality score | 0.85 |

### Categorical Features (2)

| Feature | Valid Values |
|---------|-------------|
| `city` | casablanca, marrakech, agadir, rabat, fes, tangier |
| `season_category` | march, april, summer, other |

---

## Feature Importance

Top price drivers (Random Forest feature importance):

1. **Season Category (other)** - 52.4%
2. **Rating Value** - 7.6%
3. **City (Casablanca)** - 7.3%
4. **Discount Rate** - 6.1%
5. **Bed Count** - 4.5%
6. **Review Density** - 3.8%
7. **Rating Count** - 3.7%
8. **Bedroom Count** - 2.5%
9. **City (Marrakech)** - 2.3%
10. **City (Fes)** - 2.0%

**Key Insight**: Seasonality dominates pricing (52%), followed by guest satisfaction metrics (rating_value, rating_count).

---

## City-Specific Insights

### Casablanca
- **Market Type**: Business Hub
- **Avg Price**: 440 MAD/night
- **Key Drivers**: Rating density, quality metrics
- **Recommendation**: Focus on building strong review profiles

### Marrakech
- **Market Type**: Tourist Destination  
- **Avg Price**: 465 MAD/night
- **Key Drivers**: Location, property character, ratings
- **Recommendation**: Invest in authentic features and size

### Agadir
- **Market Type**: Beach Resort
- **Avg Price**: 420 MAD/night
- **Key Drivers**: Ratings (highest ROI), trust signals
- **Recommendation**: Build rating density aggressively

### Rabat
- **Market Type**: Capital/Administrative
- **Avg Price**: 410 MAD/night
- **Key Drivers**: Professional credentials, consistency
- **Recommendation**: Emphasize reliability and business amenities

### Fes
- **Market Type**: Cultural Heritage
- **Avg Price**: 370 MAD/night
- **Key Drivers**: Cultural authenticity, Medina proximity
- **Recommendation**: Highlight heritage value

### Tangier
- **Market Type**: Port City/Gateway
- **Avg Price**: 420 MAD/night
- **Key Drivers**: Rating quality, port access
- **Recommendation**: Balance tourist and business needs

---

## Data Pipeline

### ETL Process
Located in `notebooks/airbnb_pricing_pipeline.ipynb`:

1. **Data Ingestion**: 19 JSON files from `data/raw/` (March, April, Summer)
2. **Data Cleaning**: 5,011 raw listings → 1,656 clean listings
3. **Feature Engineering**: 34 engineered features from nested JSON
4. **Price Normalization**: Extract per-night price from breakdown field
5. **EDA & Validation**: Statistical analysis and quality checks

### Model Training
Located in `notebooks/pricing_model_training.ipynb`:

1. **Feature Selection**: 12 predictive features
2. **Preprocessing**: StandardScaler + OneHotEncoder pipeline
3. **Model Training**: GridSearchCV with 5-fold cross-validation
4. **Model Comparison**: 3 algorithms evaluated
5. **Model Selection**: XGBoost chosen as champion (best R² and RMSE)
6. **Model Persistence**: Saved to `models/pricing_model_xgboost.pkl`

---

## Testing

### Run Tests Locally

```bash
cd deployment
pytest test_api.py -v
```

**Test Coverage:**
- ✅ Root endpoint
- ✅ Health check
- ✅ Model info
- ✅ Single prediction (valid inputs)
- ✅ Invalid city handling
- ✅ Batch predictions
- ✅ City insights
- ✅ Confidence intervals

**Result**: 9/9 tests passing ✅

---

## Deployment

### CI/CD Pipeline (Jenkins)

The project includes a complete Jenkins pipeline with the following stages:

1. **Checkout** - Clone repository
2. **Verify Project Layout** - Validate structure and model files
3. **Build & Test** - Run pytest in Docker container
4. **Code Quality Analysis** - SonarQube scanning
5. **Security Scan** - Safety + Bandit security checks
6. **Model Validation** - Verify model integrity
7. **Build Docker Image** - Create production image
8. **Test Docker Container** - Integration tests
9. **Push to Docker Hub** - Publish images
10. **Deploy to Local Registry** - Local deployment

### Environment Variables

```bash
# Optional: Override model path
MODEL_PATH=/app/models/pricing_model_xgboost.pkl

# Optional: Set log level
LOG_LEVEL=info
```

### Docker Image Tags

- `medgm/morocco-pricing-api:latest` - Latest stable version
- `medgm/morocco-pricing-api:<build-number>` - Specific build
- `medgm/morocco-pricing-api:<git-commit-hash>` - Git commit version

---

## Project Structure

```
pricing-model-api/
├── data/
│   ├── raw/                           # Raw JSON listings
│   ├── processed/                     # Processed datasets
│   └── used_or_will_be_used/
│       └── all_listings_clean.csv     # Clean dataset (1,656 listings)
├── deployment/
│   ├── Dockerfile                     # Production Docker image
│   ├── Jenkinsfile                    # CI/CD pipeline
│   ├── app.py                         # FastAPI application
│   ├── docker-compose.yml             # Docker Compose config
│   ├── requirements.txt               # Python dependencies
│   └── test_api.py                    # API tests
├── models/
│   ├── pricing_model_xgboost.pkl           # Trained model (217 KB)
│   └── pricing_model_xgboost_metadata.pkl  # Model metadata
├── notebooks/
│   ├── airbnb_pricing_pipeline.ipynb  # ETL pipeline
│   └── pricing_model_training.ipynb   # Model training
└── README.md                          # This file
```

---

## Technology Stack

### Backend
- **Framework**: FastAPI 0.115.6
- **Server**: Uvicorn (ASGI)
- **Validation**: Pydantic v2

### Machine Learning
- **Algorithm**: XGBoostRegressor
- **Library**: xgboost 3.1.2, scikit-learn 1.7.2
- **Model Size**: 217 KB
- **Preprocessing**: ColumnTransformer (StandardScaler + OneHotEncoder)

### Data Processing
- **pandas** 2.3.3 - Data manipulation
- **numpy** 2.3.5 - Numerical operations
- **joblib** 1.4.2 - Model serialization

### Development
- **Python**: 3.12.3
- **Testing**: pytest 9.0.1
- **Security**: Bandit, Safety
- **Code Quality**: SonarQube

### DevOps
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **CI/CD**: Jenkins
- **Registry**: Docker Hub, Local Registry

---

## Security

### Model Security
- Model files mounted as read-only in containers
- Non-root user execution in Docker
- Health checks for container monitoring

### API Security
- Input validation via Pydantic
- CORS middleware configured
- Rate limiting ready (configure as needed)

### Dependencies
- Regular security scans via Safety
- Bandit static analysis for Python code
- Automated dependency updates

---

## Performance Benchmarks

### API Response Times
- Single prediction: **~45ms**
- Batch (10 listings): **~120ms**
- Batch (100 listings): **~850ms**

### Resource Usage
- Memory: ~250 MB (container)
- CPU: <5% idle, ~20% under load
- Startup time: ~3 seconds

---

## Troubleshooting

### Model Not Loading

**Error**: `Model file not found`

**Solution**: 
```bash
# Ensure model files exist
ls -lh models/

# Should see:
# pricing_model_xgboost.pkl (217 KB)
# pricing_model_xgboost_metadata.pkl (409 bytes)
```

### Container Health Check Failing

**Error**: `Container unhealthy`

**Solution**:
```bash
# Check logs
docker logs morocco-pricing-api

# Test health endpoint manually
docker exec morocco-pricing-api curl http://localhost:8000/health
```

### Prediction Errors

**Error**: `422 Unprocessable Entity`

**Solution**: Verify all required features are provided with correct types and ranges.


## Roadmap

- [ ] Add more Moroccan cities (Essaouira, Chefchaouen, etc.)
- [ ] Implement dynamic pricing strategies (surge pricing, last-minute discounts)
- [ ] Add prediction explainability (SHAP values)
- [ ] Multi-language support for API docs
- [ ] GraphQL API endpoint
- [ ] Real-time model retraining pipeline
- [ ] A/B testing framework for model versions

---

**Built with ❤️ for the Moroccan short-term rental market**
