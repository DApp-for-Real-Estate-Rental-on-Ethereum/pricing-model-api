# AI Model Training Scripts

This directory contains training scripts for the three AI features:

1. **Tenant Risk Scoring** (`train_tenant_risk.py`)
2. **Property Clustering** (`train_property_clusters.py`)
3. **Market Trends** (no training needed - uses time-series analysis)

## Prerequisites

1. Install dependencies:
```bash
cd deployment
pip install -r requirements.txt
```

2. Ensure PostgreSQL database is running and accessible:
   - Database: `lotfi`
   - Host: `localhost:5432`
   - User: `postgres`
   - Password: `12345`

## Training Tenant Risk Model

Trains a RandomForestClassifier/GradientBoostingClassifier to predict tenant risk (0-100 trust score).

```bash
cd training
python train_tenant_risk.py
```

**Features extracted:**
- Booking behavior (completed, cancelled, avg value, stay length)
- Reclamations/complaints (counts by severity, penalty points)
- Transaction history (success/failure rates)
- User attributes (rating, score, suspension status)

**Output:**
- `deployment/models/tenant_risk_model.pkl` - Trained model
- `deployment/models/tenant_risk_scaler.pkl` - Feature scaler
- `deployment/models/tenant_risk_model_metadata.pkl` - Model metadata

## Training Property Clustering Model

Trains K-Means clustering to group properties into segments for recommendations.

```bash
cd training
python train_property_clusters.py
```

**Features extracted:**
- Price, capacity, bedrooms, bathrooms, beds
- City (one-hot encoded)
- Amenity count, image count
- Negotiation percentage

**Output:**
- `deployment/models/property_cluster_model.pkl` - Trained K-Means model
- `deployment/models/property_cluster_scaler.pkl` - Feature scaler
- `deployment/models/property_cluster_model_metadata.pkl` - Metadata

## Model Evaluation

Both scripts output:
- Cross-validation scores
- Test set metrics (accuracy, ROC-AUC, silhouette score)
- Classification/clustering reports
- Feature importance (for risk model)

## Hyperparameter Tuning

- **Tenant Risk**: Uses GridSearchCV with 5-fold CV
  - RandomForest: n_estimators, max_depth, min_samples_split, class_weight
  - GradientBoosting: n_estimators, max_depth, learning_rate, subsample

- **Property Clusters**: Uses silhouette score to find optimal k (2-10 clusters)

## Retraining

Models should be retrained periodically as new data accumulates:
- **Risk Model**: Monthly or when significant new bookings/reclamations occur
- **Clustering**: Quarterly or when property inventory changes significantly

## Notes

- If database is empty, scripts will generate sample data for demonstration
- Models use heuristic fallbacks if training data is insufficient
- All models are saved in `deployment/models/` directory

