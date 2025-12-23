"""
Training Script for Property Clustering Model
=============================================

Trains K-Means clustering model to group properties into segments.
Used for recommendation engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deployment"))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import logging
from db_connection import execute_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_property_data() -> pd.DataFrame:
    """Load property features from database."""
    logger.info("Loading property data from database...")
    
    query = """
        SELECT 
            p.id,
            p.daily_price,
            p.capacity,
            p.number_of_bedrooms as bedrooms,
            p.number_of_bathrooms as bathrooms,
            p.number_of_beds as beds,
            p.negotiation_percentage,
            COALESCE(p.city, a.city, 'casablanca') as city,
            COUNT(DISTINCT pi.id) as image_count,
            COUNT(DISTINCT pa.amenity_id) as amenity_count
        FROM properties p
        LEFT JOIN addresses a ON p.address_id = a.id
        LEFT JOIN property_images pi ON pi.propety_id = p.id
        LEFT JOIN properties_amenities pa ON pa.property_id = p.id
        WHERE p.status IN ('APPROVED', 'VISIBLE_ONLY_FOR_TENANTS')
        GROUP BY p.id, p.daily_price, p.capacity, p.number_of_bedrooms, 
                 p.number_of_bathrooms, p.number_of_beds, p.negotiation_percentage, 
                 p.city, a.city
    """
    
    results = execute_query(query)
    
    if not results:
        logger.warning("No properties found. Using sample data.")
        return create_sample_data()
    
    df = pd.DataFrame(results)
    logger.info(f"Loaded {len(df)} properties")
    return df


def create_sample_data() -> pd.DataFrame:
    """Create sample property data."""
    np.random.seed(42)
    n_samples = 150
    
    cities = ['casablanca', 'marrakech', 'agadir', 'rabat', 'fes', 'tangier']
    
    data = {
        'id': [f'prop-{i}' for i in range(n_samples)],
        'daily_price': np.random.normal(400, 150, n_samples).clip(100, 1000),
        'capacity': np.random.choice([2, 4, 6, 8], n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4], n_samples),
        'bathrooms': np.random.choice([1, 2, 3], n_samples),
        'beds': np.random.choice([1, 2, 3, 4], n_samples),
        'negotiation_percentage': np.random.choice([0, 5, 10, 15], n_samples),
        'city': np.random.choice(cities, n_samples),
        'image_count': np.random.poisson(5, n_samples),
        'amenity_count': np.random.poisson(8, n_samples),
    }
    
    return pd.DataFrame(data)


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature vectors for clustering.
    
    Returns:
        (feature_matrix, feature_names, property_ids)
    """
    # Normalize price (log scale)
    df['price_normalized'] = np.log1p(df['daily_price'])
    
    # Normalize numeric features
    numeric_cols = ['capacity', 'bedrooms', 'bathrooms', 'beds', 'image_count', 'amenity_count']
    for col in numeric_cols:
        if col in df.columns:
            max_val = df[col].max()
            if max_val > 0:
                df[f'{col}_normalized'] = df[col] / max_val
            else:
                df[f'{col}_normalized'] = 0.0
    
    # One-hot encode city
    cities = df['city'].str.lower().unique()
    city_cols = []
    for city in cities:
        col_name = f'city_{city}'
        df[col_name] = (df['city'].str.lower() == city).astype(int)
        city_cols.append(col_name)
    
    # Negotiation indicator
    df['is_negotiable'] = (df['negotiation_percentage'] > 0).astype(int)
    
    # Build feature matrix
    feature_cols = ['price_normalized'] + \
                   [f'{col}_normalized' for col in numeric_cols if f'{col}_normalized' in df.columns] + \
                   city_cols + \
                   ['is_negotiable']
    
    feature_matrix = df[feature_cols].values
    property_ids = df['id'].values
    
    return feature_matrix, feature_cols, property_ids


def find_optimal_clusters(X: np.ndarray, max_k: int = 10) -> int:
    """Find optimal number of clusters using silhouette score."""
    n_samples = len(X)
    
    if n_samples < 2:
        logger.warning(f"Only {n_samples} properties found. Cannot cluster. Using k=1.")
        return 1
    
    logger.info(f"Finding optimal number of clusters (n_samples={n_samples})...")
    
    # Limit max_k to n_samples
    max_k = min(max_k, n_samples - 1)
    if max_k < 2:
        logger.warning(f"Not enough samples for clustering. Using k=1.")
        return 1
    
    best_k = min(3, max_k)
    best_score = -1
    
    for k in range(2, max_k + 1):
        if k >= n_samples:
            break
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            
            logger.info(f"  k={k}: silhouette_score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            logger.warning(f"  k={k}: Failed - {e}")
            break
    
    logger.info(f"✅ Optimal k: {best_k} (silhouette_score: {best_score:.4f})")
    return best_k


def train_model():
    """Train K-Means clustering model."""
    logger.info("=" * 60)
    logger.info("TRAINING PROPERTY CLUSTERING MODEL")
    logger.info("=" * 60)
    
    # Load data
    df = load_property_data()
    
    # Prepare features
    X, feature_names, property_ids = prepare_features(df)
    
    if len(X) == 0:
        logger.error("No properties found after feature preparation. Cannot train model.")
        return None, None, None
    
    if len(X) < 2:
        logger.warning(f"Only {len(X)} property found. Need at least 2 for clustering. Using sample data.")
        df = create_sample_data()
        X, feature_names, property_ids = prepare_features(df)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_scaled, max_k=8)
    
    if optimal_k < 2:
        logger.warning("Not enough properties for clustering. Saving placeholder model.")
        # Create a dummy model that just assigns all to cluster 0
        from sklearn.base import BaseEstimator
        class DummyClusterer(BaseEstimator):
            def fit(self, X, y=None):
                return self
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
        
        kmeans = DummyClusterer()
        clusters = np.zeros(len(X_scaled), dtype=int)
        silhouette = 0.0
    else:
        # Train final model
        logger.info(f"\nTraining K-Means with k={optimal_k}...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(X_scaled)
    
    # Evaluate
    if optimal_k >= 2 and len(X_scaled) > 1:
        silhouette = silhouette_score(X_scaled, clusters)
        logger.info(f"Final silhouette score: {silhouette:.4f}")
    else:
        silhouette = 0.0
        logger.info("Skipping silhouette score (insufficient data)")
    
    # Analyze clusters
    if len(df) > 0:
        df['cluster'] = clusters
        logger.info("\nCluster analysis:")
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            logger.info(f"\nCluster {cluster_id} ({len(cluster_df)} properties):")
            if 'daily_price' in cluster_df.columns:
                logger.info(f"  Avg price: {cluster_df['daily_price'].mean():.0f} MAD")
            if 'capacity' in cluster_df.columns:
                logger.info(f"  Avg capacity: {cluster_df['capacity'].mean():.1f}")
            if 'bedrooms' in cluster_df.columns:
                logger.info(f"  Avg bedrooms: {cluster_df['bedrooms'].mean():.1f}")
            if 'city' in cluster_df.columns:
                logger.info(f"  Top cities: {cluster_df['city'].value_counts().head(3).to_dict()}")
    
    # Save model
    models_dir = Path(__file__).parent.parent / "deployment" / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "property_cluster_model.pkl"
    scaler_path = models_dir / "property_cluster_scaler.pkl"
    metadata_path = models_dir / "property_cluster_model_metadata.pkl"
    
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        'n_clusters': optimal_k,
        'version': '1.0',
        'feature_names': feature_names,
        'silhouette_score': float(silhouette),
        'n_properties': len(X_scaled),
        'cluster_distribution': pd.Series(clusters).value_counts().to_dict() if len(clusters) > 0 else {}
    }
    
    joblib.dump(metadata, metadata_path)
    
    logger.info(f"\n✅ Model saved to {model_path}")
    logger.info(f"✅ Scaler saved to {scaler_path}")
    logger.info(f"✅ Metadata saved to {metadata_path}")
    
    return kmeans, scaler, metadata


if __name__ == "__main__":
    train_model()

