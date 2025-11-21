# Morocco Airbnb Dynamic Pricing API

AI-powered dynamic pricing model for Airbnb listings across Moroccan cities (Casablanca, Marrakech, Agadir, Rabat, Fes, Tangier).

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## 🎯 Features

- **5.73% MAPE Accuracy**: Production-grade GradientBoosting model
- **City-Specific Intelligence**: Tailored recommendations for 6 Moroccan markets
- **RESTful API**: FastAPI with automatic OpenAPI documentation
- **Batch Processing**: Handle up to 100 listings per request
- **SHAP Explainability**: Transparent feature attribution
- **Docker Ready**: One-command deployment
- **CI/CD Pipeline**: Jenkins automation with testing & security scans

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **MAE** | 2,332 MAD (~$230 USD) |
| **MAPE** | 5.73% |
| **Training Set** | 2,676 listings (March + April) |
| **Validation Set** | 1,287 listings (Summer) |
| **Cities Covered** | 6 (Casablanca, Marrakech, Agadir, Rabat, Fes, Tangier) |

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone git@github.com:DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api.git
cd pricing-model-api

# Start the API
cd deployment
docker-compose up -d

# Verify it's running
curl http://localhost:8000/health
```

### Manual Setup

```bash
# Install dependencies
pip install -r deployment/requirements.txt

# Run the API
cd deployment
uvicorn app:app --host 0.0.0.0 --port 8000

# Access the API
open http://localhost:8000/docs
```

## 📖 API Documentation

### Endpoints

- **`GET /`** - API information
- **`GET /health`** - Health check
- **`GET /model-info`** - Model metadata
- **`POST /predict`** - Single listing prediction
- **`POST /batch-predict`** - Batch predictions (up to 100)
- **`GET /city-insights/{city}`** - City-specific recommendations

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "casablanca",
    "period": "summer",
    "geo_cluster": 1,
    "rating_value": 4.8,
    "rating_count": 50,
    "rating_density": 1.0,
    "has_rating": 1,
    "badge_superhost": false,
    "badge_guest_favorite": false,
    "badge_top_x": false,
    "any_badge": 0,
    "dist_to_center": 0.05,
    "struct_bedrooms": 2.0,
    "struct_bathrooms": 1.0,
    "struct_surface_m2": 80.0
  }'
```

### Example Response

```json
{
  "predicted_price_mad": 50159.0,
  "predicted_price_usd": 5016.0,
  "confidence_interval_lower": 47827.0,
  "confidence_interval_upper": 52491.0,
  "city": "casablanca",
  "period": "summer",
  "model_version": "1.0"
}
```

## 🏗️ Project Structure

```
pricing-model-api/
├── deployment/
│   ├── app.py                  # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile             # Container definition
│   ├── docker-compose.yml     # Docker orchestration
│   ├── test_api.py            # Test suite
│   ├── Jenkinsfile            # CI/CD pipeline
│   └── deployment_report.md   # Technical documentation
│
├── models/
│   ├── pricing_gradient_boosting_v1.pkl  # Trained model
│   └── model_metrics_v1.csv              # Performance metrics
│
├── data/
│   └── used_or_will_be_used/
│       ├── all_listings_clean.csv   # Training data (3,963 listings)
│       └── houses_data_eng.csv      # Structural features (4,675 properties)
│
└── README.md
```

## 🧪 Testing

```bash
# Run unit tests
cd deployment
pytest test_api.py -v

# Run with coverage
pytest --cov=app --cov-report=html test_api.py

# Load testing
pip install locust
locust -f locustfile.py --host http://localhost:8000
```

## 🔄 CI/CD Pipeline

The Jenkins pipeline includes:

1. **Code Quality**: Linting (flake8, black)
2. **Security Scanning**: Safety, Bandit
3. **Model Validation**: Verify model can be loaded
4. **Unit Tests**: Pytest with coverage
5. **Docker Build**: Multi-stage build
6. **Container Testing**: Health checks & API tests
7. **Registry Push**: Docker Hub/private registry
8. **Deployment**: Staging → Production (with approval)
9. **Performance Testing**: Locust load tests

### Jenkins Setup

1. Create a new Pipeline job in Jenkins
2. Configure Git repository: `git@github.com:DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api.git`
3. Set Pipeline script path: `deployment/Jenkinsfile`
4. Add credentials:
   - `docker-hub-credentials`: Docker registry credentials
5. Configure webhooks for automatic builds

## 🌍 City-Specific Insights

### Casablanca (Business Hub)
- **Average Price**: 49,259 MAD
- **Key Drivers**: City premium (+4,652 MAD), rating density
- **Recommendation**: Focus on ratings over location

### Marrakech (Tourist Destination)
- **Average Price**: 46,685 MAD
- **Key Drivers**: Location (+1,791 MAD), property size
- **Recommendation**: Prioritize Medina proximity

### Agadir (Beach Resort)
- **Average Price**: 41,694 MAD
- **Key Drivers**: Rating quality (+19% uplift), trust signals
- **Recommendation**: Invest in reviews & Superhost

### Rabat (Capital/Admin)
- **Average Price**: 40,959 MAD
- **Key Drivers**: Superhost badge (+5.5%), credentials
- **Recommendation**: Professional amenities matter

## 📈 Performance Benchmarks

- **Response Time**: <100ms (p95)
- **Throughput**: 100+ requests/sec
- **Memory Usage**: ~500MB
- **Startup Time**: <10s

## 🔐 Security

- Non-root Docker user
- Input validation with Pydantic
- Dependency scanning with Safety
- Code security checks with Bandit
- CORS configuration
- Rate limiting (recommended for production)

## 🛠️ Development

```bash
# Install development dependencies
pip install -r deployment/requirements.txt
pip install pytest black flake8 locust

# Format code
black deployment/app.py

# Lint
flake8 deployment/app.py --max-line-length=120

# Type checking
mypy deployment/app.py
```

## 📝 Environment Variables

```bash
# Optional configuration
LOG_LEVEL=info                # Logging level
MODEL_PATH=/app/models/...    # Custom model path
MAX_BATCH_SIZE=100            # Max batch prediction size
```

## 🚢 Deployment Options

### Docker Compose (Single Server)
```bash
docker-compose up -d
```

### Kubernetes (Production)
```bash
kubectl apply -f k8s/deployment.yaml
```

### AWS Elastic Beanstalk
```bash
eb init
eb create pricing-api-prod
```

## 📊 Monitoring

- **Health Check**: `GET /health`
- **Metrics**: Prometheus-compatible (add `prometheus-fastapi-instrumentator`)
- **Logging**: Structured JSON logs
- **Tracing**: Add OpenTelemetry for distributed tracing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Training data: Airbnb Morocco (6 cities, 3,963 listings)
- Feature enrichment: Housing portal data (4,675 properties)
- ML framework: scikit-learn GradientBoosting
- Explainability: SHAP (SHapley Additive exPlanations)

## 📧 Contact

- **Project Repository**: [GitHub](https://github.com/DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api)
- **API Documentation**: http://localhost:8000/docs
- **Technical Report**: [deployment_report.md](deployment/deployment_report.md)

---

**Built with ❤️ for the Moroccan rental market**
