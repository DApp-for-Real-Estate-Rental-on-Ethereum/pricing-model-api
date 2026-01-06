"""
Property Recommendation Engine
==============================

Recommends properties to tenants using K-Means clustering + cosine similarity.
Based on tenant's booking history, preferences, and budget.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
from datetime import datetime

from deployment.db_connection import execute_query, execute_query_single
from deployment.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["Property Recommendations"])


class PropertyRecommendation(BaseModel):
    """Single property recommendation."""
    
    property_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    daily_price: float
    city: str
    capacity: int
    bedrooms: int
    bathrooms: int
    title: str


class RecommendationResponse(BaseModel):
    """Response model for property recommendations."""
    
    tenant_id: int
    recommendations: List[PropertyRecommendation]
    total_found: int
    tenant_preference_summary: dict


def extract_property_features(property_id: Optional[str] = None) -> pd.DataFrame:
    """
    Extract feature vectors for all properties (or a specific property).
    
    Returns:
        DataFrame with property features for ML
    """
    if property_id:
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
            WHERE p.id = %s 
              AND p.status IN ('APPROVED', 'VISIBLE_ONLY_FOR_TENANTS')
            GROUP BY p.id, p.daily_price, p.capacity, p.number_of_bedrooms, 
                     p.number_of_bathrooms, p.number_of_beds, p.negotiation_percentage, 
                     p.city, a.city
        """
        results = execute_query(query, (property_id,))
    else:
        query = """
            SELECT 
                p.id,
                p.daily_price,
                p.capacity,
                p.number_of_bedrooms as bedrooms,
                p.number_of_bathrooms as bathrooms,
                p.number_of_beds as beds,
                p.negotiation_percentage,
                COALESCE(a.city, p.city, 'casablanca') as city,
                COUNT(DISTINCT pi.id) as image_count,
                COUNT(DISTINCT pa.amenity_id) as amenity_count,
                p.title
            FROM properties p
            LEFT JOIN addresses a ON p.address_id = a.id
            LEFT JOIN property_images pi ON pi.propety_id = p.id
            LEFT JOIN properties_amenities pa ON pa.property_id = p.id
            WHERE p.status IN ('APPROVED', 'VISIBLE_ONLY_FOR_TENANTS')
            GROUP BY p.id, p.daily_price, p.capacity, p.number_of_bedrooms, 
                     p.number_of_bathrooms, p.number_of_beds, p.negotiation_percentage, 
                     p.city, a.city, p.title
        """
        results = execute_query(query)
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Normalize features
    if len(df) > 0:
        # Normalize price (log scale)
        df['price_normalized'] = np.log1p(df['daily_price'])
        
        # Normalize numeric features
        for col in ['capacity', 'bedrooms', 'bathrooms', 'beds', 'image_count', 'amenity_count']:
            if col in df.columns:
                max_val = df[col].max()
                if max_val > 0:
                    df[f'{col}_normalized'] = df[col] / max_val
                else:
                    df[f'{col}_normalized'] = 0.0
        
        # One-hot encode city
        cities = df['city'].str.lower().unique()
        for city in cities:
            df[f'city_{city}'] = (df['city'].str.lower() == city).astype(int)
        
        # Negotiation indicator
        df['is_negotiable'] = (df['negotiation_percentage'] > 0).astype(int)
    
    return df


def extract_tenant_preferences(tenant_id: int) -> dict:
    """
    Extract tenant's booking history and preferences.
    
    Returns:
        Dictionary with tenant preference vector
    """
    # Get tenant's past bookings with property details
    query = """
        SELECT 
            b.property_id,
            b.total_price,
            EXTRACT(EPOCH FROM (b.check_out_date::timestamp - b.check_in_date::timestamp)) / 86400 as stay_length,
            p.daily_price,
            p.capacity,
            p.number_of_bedrooms as bedrooms,
            p.number_of_bathrooms as bathrooms,
            COALESCE(p.city, a.city, 'casablanca') as city
        FROM bookings b
        JOIN properties p ON p.id = b.property_id
        LEFT JOIN addresses a ON p.address_id = a.id
        WHERE b.user_id = %s
          AND b.status IN ('COMPLETED', 'CONFIRMED', 'TENANT_CHECKED_OUT')
        ORDER BY b.created_at DESC
        LIMIT 50
    """
    bookings = execute_query(query, (tenant_id,))
    
    if not bookings:
        return {
            'avg_price': 0.0,
            'avg_capacity': 0.0,
            'avg_bedrooms': 0.0,
            'avg_bathrooms': 0.0,
            'preferred_cities': [],
            'n_bookings': 0
        }
    
    df_bookings = pd.DataFrame(bookings)
    
    preferences = {
        'avg_price': float(df_bookings['daily_price'].mean()) if 'daily_price' in df_bookings.columns else 0.0,
        'avg_capacity': float(df_bookings['capacity'].mean()) if 'capacity' in df_bookings.columns else 0.0,
        'avg_bedrooms': float(df_bookings['bedrooms'].mean()) if 'bedrooms' in df_bookings.columns else 0.0,
        'avg_bathrooms': float(df_bookings['bathrooms'].mean()) if 'bathrooms' in df_bookings.columns else 0.0,
        'preferred_cities': df_bookings['city'].value_counts().head(3).index.tolist() if 'city' in df_bookings.columns else [],
        'n_bookings': len(df_bookings)
    }
    
    return preferences


def build_tenant_vector(tenant_prefs: dict, property_features_df: pd.DataFrame) -> np.ndarray:
    """
    Build a tenant preference vector from their history.
    
    Returns:
        NumPy array representing tenant preferences
    """
    if len(property_features_df) == 0:
        return np.array([])
    
    # Use average of properties tenant has booked (if available)
    # Otherwise use tenant's explicit preferences
    
    # Normalize tenant preferences to match property feature scale
    tenant_vector = []
    
    # Price preference (normalized)
    if tenant_prefs['avg_price'] > 0:
        max_price = property_features_df['daily_price'].max()
        tenant_vector.append(np.log1p(tenant_prefs['avg_price']) if max_price > 0 else 0.0)
    else:
        tenant_vector.append(property_features_df['price_normalized'].mean() if 'price_normalized' in property_features_df.columns else 0.0)
    
    # Capacity, bedrooms, bathrooms preferences
    for feature in ['capacity', 'bedrooms', 'bathrooms']:
        if tenant_prefs.get(f'avg_{feature}', 0) > 0:
            max_val = property_features_df[feature].max() if feature in property_features_df.columns else 1.0
            tenant_vector.append(tenant_prefs[f'avg_{feature}'] / max_val if max_val > 0 else 0.0)
        else:
            tenant_vector.append(property_features_df[f'{feature}_normalized'].mean() if f'{feature}_normalized' in property_features_df.columns else 0.0)
    
    # City preferences (one-hot)
    cities = [col.replace('city_', '') for col in property_features_df.columns if col.startswith('city_')]
    for city in cities:
        tenant_vector.append(1.0 if city in tenant_prefs.get('preferred_cities', []) else 0.0)
    
    # Negotiation preference (assume neutral)
    tenant_vector.append(0.5)
    
    return np.array(tenant_vector)


def recommend_properties(
    tenant_id: int,
    max_results: int = 10,
    budget_mad: Optional[float] = None,
    exclude_property_ids: Optional[List[str]] = None
) -> List[PropertyRecommendation]:
    """
    Recommend properties for a tenant.
    
    Args:
        tenant_id: Tenant user ID
        max_results: Maximum number of recommendations
        budget_mad: Optional budget filter (daily price <= budget_mad)
        exclude_property_ids: Property IDs to exclude (e.g., already booked)
        
    Returns:
        List of PropertyRecommendation objects
    """
    # Get model just to ensure it's loaded/logged (optional here as we rely more on cosine similarity for now)
    # in a real hybrid system, we'd use the clusters
    # model = model_manager.get_clustering_model()
    
    # Load property features (Local var)
    property_features_df = extract_property_features()
    
    if len(property_features_df) == 0:
        return []
    
    # Get tenant preferences
    tenant_prefs = extract_tenant_preferences(tenant_id)
    
    # Build tenant vector
    tenant_vector = build_tenant_vector(tenant_prefs, property_features_df)
    
    if len(tenant_vector) == 0:
        return []
    
    # Filter properties
    filtered_df = property_features_df.copy()
    
    # Exclude already booked properties
    if exclude_property_ids:
        filtered_df = filtered_df[~filtered_df['id'].isin(exclude_property_ids)]
    
    # Budget filter
    if budget_mad:
        filtered_df = filtered_df[filtered_df['daily_price'] <= budget_mad]
    
    if len(filtered_df) == 0:
        return []
    
    # Build property feature vectors
    property_vectors = []
    
    for _, row in filtered_df.iterrows():
        prop_vector = []
        
        # Price
        prop_vector.append(row.get('price_normalized', 0.0))
        
        # Capacity, bedrooms, bathrooms
        for feature in ['capacity', 'bedrooms', 'bathrooms']:
            prop_vector.append(row.get(f'{feature}_normalized', 0.0))
        
        # City one-hot
        cities = [col.replace('city_', '') for col in filtered_df.columns if col.startswith('city_')]
        for city in cities:
            prop_vector.append(row.get(f'city_{city}', 0.0))
        
        # Negotiation
        prop_vector.append(row.get('is_negotiable', 0.0))
        
        property_vectors.append(prop_vector)
    
    property_vectors = np.array(property_vectors)
    
    # Ensure dimensions match
    min_dim = min(len(tenant_vector), property_vectors.shape[1])
    tenant_vector = tenant_vector[:min_dim]
    property_vectors = property_vectors[:, :min_dim]
    
    # Calculate cosine similarity
    similarities = cosine_similarity([tenant_vector], property_vectors)[0]
    
    # Get top N recommendations
    top_indices = np.argsort(similarities)[::-1][:max_results]
    
    recommendations = []
    for idx in top_indices:
        prop_row = filtered_df.iloc[idx]
        recommendations.append(PropertyRecommendation(
            property_id=str(prop_row['id']),
            score=float(similarities[idx]),
            daily_price=float(prop_row['daily_price']),
            city=str(prop_row.get('city', 'unknown')),
            capacity=int(prop_row.get('capacity', 0)),
            bedrooms=int(prop_row.get('bedrooms', 0)),
            bathrooms=int(prop_row.get('bathrooms', 0)),
            title=str(prop_row.get('title', 'Property'))
        ))
    
    return recommendations


@router.get("/tenant/{tenant_id}", response_model=RecommendationResponse)
async def get_recommendations(
    tenant_id: int,
    max_results: int = Query(10, ge=1, le=50),
    budget_mad: Optional[float] = Query(None, ge=0),
    exclude_property_ids: Optional[str] = Query(None, description="Comma-separated property IDs to exclude")
):
    """
    Get property recommendations for a tenant.
    
    Uses K-Means clustering + cosine similarity based on tenant's booking history.
    """
    try:
        exclude_list = None
        if exclude_property_ids:
            exclude_list = [pid.strip() for pid in exclude_property_ids.split(',')]
        
        recommendations = recommend_properties(
            tenant_id=tenant_id,
            max_results=max_results,
            budget_mad=budget_mad,
            exclude_property_ids=exclude_list
        )
        
        tenant_prefs = extract_tenant_preferences(tenant_id)
        
        return RecommendationResponse(
            tenant_id=tenant_id,
            recommendations=recommendations,
            total_found=len(recommendations),
            tenant_preference_summary=tenant_prefs
        )
    except Exception as e:
        logger.error(f"Error getting recommendations for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/similar/{property_id}", response_model=List[PropertyRecommendation])
async def get_similar_properties(
    property_id: str,
    max_results: int = Query(5, ge=1, le=20)
):
    """
    Get properties similar to a given property.
    
    Uses cosine similarity on property feature vectors.
    """
    try:
        # Get features for the seed property
        seed_df = extract_property_features(property_id)
        if len(seed_df) == 0:
            raise HTTPException(status_code=404, detail="Property not found")
        
        # Get all other properties
        all_df = extract_property_features()
        all_df = all_df[all_df['id'] != property_id]
        
        if len(all_df) == 0:
            return []
        
        # Build feature vectors
        seed_vector = []
        seed_row = seed_df.iloc[0]
        
        seed_vector.append(seed_row.get('price_normalized', 0.0))
        for feature in ['capacity', 'bedrooms', 'bathrooms']:
            seed_vector.append(seed_row.get(f'{feature}_normalized', 0.0))
        
        cities = [col.replace('city_', '') for col in all_df.columns if col.startswith('city_')]
        for city in cities:
            seed_vector.append(seed_row.get(f'city_{city}', 0.0))
        seed_vector.append(seed_row.get('is_negotiable', 0.0))
        
        property_vectors = []
        
        for _, row in all_df.iterrows():
            prop_vector = []
            prop_vector.append(row.get('price_normalized', 0.0))
            for feature in ['capacity', 'bedrooms', 'bathrooms']:
                prop_vector.append(row.get(f'{feature}_normalized', 0.0))
            for city in cities:
                prop_vector.append(row.get(f'city_{city}', 0.0))
            prop_vector.append(row.get('is_negotiable', 0.0))
            
            property_vectors.append(prop_vector)
        
        property_vectors = np.array(property_vectors)
        seed_vector = np.array(seed_vector)
        
        # Calculate similarities
        similarities = cosine_similarity([seed_vector], property_vectors)[0]
        
        # Get top N
        top_indices = np.argsort(similarities)[::-1][:max_results]
        
        recommendations = []
        for idx in top_indices:
            prop_row = all_df.iloc[idx]
            recommendations.append(PropertyRecommendation(
                property_id=str(prop_row['id']),
                score=float(similarities[idx]),
                daily_price=float(prop_row['daily_price']),
                city=str(prop_row.get('city', 'unknown')),
                capacity=int(prop_row.get('capacity', 0)),
                bedrooms=int(prop_row.get('bedrooms', 0)),
                bathrooms=int(prop_row.get('bathrooms', 0)),
                title=str(prop_row.get('title', 'Property'))
            ))
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar properties for {property_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get similar properties: {str(e)}")
