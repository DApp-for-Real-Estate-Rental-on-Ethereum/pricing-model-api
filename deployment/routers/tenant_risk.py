"""
Tenant Risk Scoring Module
==========================

Provides 0-100 trust score for tenants using RandomForestClassifier.
Features extracted from bookings, reclamations, transactions, and user data.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

from deployment.db_connection import execute_query, execute_query_single

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenant-risk", tags=["Tenant Risk Scoring"])

# Global model
RISK_MODEL = None
RISK_MODEL_METADATA = {}


class TenantRiskFeatures(BaseModel):
    """Feature vector for tenant risk scoring."""
    
    # Booking behavior
    n_bookings_total: int = 0
    n_completed_bookings: int = 0
    n_cancelled_bookings: int = 0
    avg_booking_value: float = 0.0
    avg_stay_length_days: float = 0.0
    recent_bookings_last_6m: int = 0
    
    # Disputes/complaints
    n_reclamations_as_target: int = 0
    n_reclamations_low: int = 0
    n_reclamations_medium: int = 0
    n_reclamations_high: int = 0
    n_reclamations_critical: int = 0
    n_reclamations_open: int = 0
    n_reclamations_resolved_against_user: int = 0
    total_penalty_points: int = 0
    total_refund_amount: float = 0.0
    
    # Payments
    n_transactions_total: int = 0
    n_transactions_success: int = 0
    n_transactions_failed: int = 0
    failed_transaction_rate: float = 0.0
    avg_transaction_amount: float = 0.0
    
    # Account attributes
    user_rating: float = 0.0
    user_score: int = 100
    user_penalty_points: int = 0
    is_suspended: bool = False
    account_age_days: int = 0
    has_verified_profile: bool = False


class TenantRiskResponse(BaseModel):
    """Response model for tenant risk scoring."""
    
    tenant_id: int
    trust_score: int = Field(..., ge=0, le=100, description="Trust score from 0-100")
    risk_band: str = Field(..., description="Risk category: LOW, MEDIUM, HIGH, CRITICAL")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being high-risk")
    top_factors: List[str] = Field(..., description="Top contributing factors")
    features: TenantRiskFeatures


def extract_tenant_features(user_id: int) -> TenantRiskFeatures:
    """
    Extract all features for a tenant from the database.
    
    Args:
        user_id: User ID of the tenant
        
    Returns:
        TenantRiskFeatures object with all computed features
    """
    features = TenantRiskFeatures()
    
    # Get user account info
    user_query = """
        SELECT rating, score, penalty_points, is_suspended, verification_expiration,
               (SELECT is_complete FROM user_profile_status WHERE user_id = users.id::text) as is_complete
        FROM users
        WHERE id = %s
    """
    user_data = execute_query_single(user_query, (user_id,))
    
    if user_data:
        features.user_rating = float(user_data.get('rating') or 0.0)
        features.user_score = int(user_data.get('score') or 100)
        features.user_penalty_points = int(user_data.get('penalty_points') or 0)
        features.is_suspended = bool(user_data.get('is_suspended') or False)
        features.has_verified_profile = bool(user_data.get('is_complete') or False)
        
        # Account age (days since verification expiration or first booking)
        verif_exp = user_data.get('verification_expiration')
        if verif_exp:
            try:
                if isinstance(verif_exp, str):
                    verif_exp = datetime.fromisoformat(verif_exp.replace('Z', '+00:00'))
                account_age = (datetime.now() - verif_exp.replace(tzinfo=None)).days
                features.account_age_days = max(0, account_age)
            except:
                features.account_age_days = 0
    
    # Get booking statistics
    bookings_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status IN ('COMPLETED', 'TENANT_CHECKED_OUT', 'CONFIRMED') THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status LIKE '%%CANCELLED%%' OR status = 'CANCELLED_BY_TENANT' THEN 1 ELSE 0 END) as cancelled,
            AVG(total_price) as avg_price,
            AVG(EXTRACT(EPOCH FROM (check_out_date::timestamp - check_in_date::timestamp)) / 86400) as avg_stay_days,
            SUM(CASE WHEN created_at >= NOW() - INTERVAL '6 months' THEN 1 ELSE 0 END) as recent_6m
        FROM bookings
        WHERE user_id = %s
    """
    booking_stats = execute_query_single(bookings_query, (user_id,))
    
    if booking_stats:
        features.n_bookings_total = int(booking_stats.get('total') or 0)
        features.n_completed_bookings = int(booking_stats.get('completed') or 0)
        features.n_cancelled_bookings = int(booking_stats.get('cancelled') or 0)
        features.avg_booking_value = float(booking_stats.get('avg_price') or 0.0)
        features.avg_stay_length_days = float(booking_stats.get('avg_stay_days') or 0.0)
        features.recent_bookings_last_6m = int(booking_stats.get('recent_6m') or 0)
    
    # Get reclamation statistics (complaints AGAINST this user)
    reclamations_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN severity = 'LOW' THEN 1 ELSE 0 END) as low,
            SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END) as medium,
            SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) as critical,
            SUM(CASE WHEN status = 'OPEN' OR status = 'IN_REVIEW' THEN 1 ELSE 0 END) as open,
            SUM(CASE WHEN status = 'RESOLVED' AND resolution_notes LIKE '%%against%%' THEN 1 ELSE 0 END) as resolved_against,
            SUM(penalty_points) as total_penalty,
            SUM(refund_amount) as total_refund
        FROM reclamations
        WHERE target_user_id = %s
    """
    reclamation_stats = execute_query_single(reclamations_query, (user_id,))
    
    if reclamation_stats:
        features.n_reclamations_as_target = int(reclamation_stats.get('total') or 0)
        features.n_reclamations_low = int(reclamation_stats.get('low') or 0)
        features.n_reclamations_medium = int(reclamation_stats.get('medium') or 0)
        features.n_reclamations_high = int(reclamation_stats.get('high') or 0)
        features.n_reclamations_critical = int(reclamation_stats.get('critical') or 0)
        features.n_reclamations_open = int(reclamation_stats.get('open') or 0)
        features.n_reclamations_resolved_against_user = int(reclamation_stats.get('resolved_against') or 0)
        features.total_penalty_points = int(reclamation_stats.get('total_penalty') or 0)
        features.total_refund_amount = float(reclamation_stats.get('total_refund') or 0.0)
    
    # Get transaction statistics
    transactions_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success,
            SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
            AVG(amount) as avg_amount
        FROM transactions
        WHERE user_id = %s
    """
    transaction_stats = execute_query_single(transactions_query, (user_id,))
    
    if transaction_stats:
        total_tx = int(transaction_stats.get('total') or 0)
        features.n_transactions_total = total_tx
        features.n_transactions_success = int(transaction_stats.get('success') or 0)
        features.n_transactions_failed = int(transaction_stats.get('failed') or 0)
        features.failed_transaction_rate = (features.n_transactions_failed / total_tx) if total_tx > 0 else 0.0
        features.avg_transaction_amount = float(transaction_stats.get('avg_amount') or 0.0)
    
    return features


def features_to_dataframe(features: TenantRiskFeatures) -> pd.DataFrame:
    """Convert TenantRiskFeatures to DataFrame for model prediction."""
    return pd.DataFrame([{
        'n_bookings_total': features.n_bookings_total,
        'n_completed_bookings': features.n_completed_bookings,
        'n_cancelled_bookings': features.n_cancelled_bookings,
        'avg_booking_value': features.avg_booking_value,
        'avg_stay_length_days': features.avg_stay_length_days,
        'recent_bookings_last_6m': features.recent_bookings_last_6m,
        'n_reclamations_as_target': features.n_reclamations_as_target,
        'n_reclamations_low': features.n_reclamations_low,
        'n_reclamations_medium': features.n_reclamations_medium,
        'n_reclamations_high': features.n_reclamations_high,
        'n_reclamations_critical': features.n_reclamations_critical,
        'n_reclamations_open': features.n_reclamations_open,
        'n_reclamations_resolved_against_user': features.n_reclamations_resolved_against_user,
        'total_penalty_points': features.total_penalty_points,
        'total_refund_amount': features.total_refund_amount,
        'n_transactions_total': features.n_transactions_total,
        'n_transactions_success': features.n_transactions_success,
        'n_transactions_failed': features.n_transactions_failed,
        'failed_transaction_rate': features.failed_transaction_rate,
        'avg_transaction_amount': features.avg_transaction_amount,
        'user_rating': features.user_rating,
        'user_score': features.user_score,
        'user_penalty_points': features.user_penalty_points,
        'is_suspended': int(features.is_suspended),
        'account_age_days': features.account_age_days,
        'has_verified_profile': int(features.has_verified_profile),
    }])


def predict_risk_score(features: TenantRiskFeatures) -> dict:
    """
    Predict tenant risk score using the trained model.
    
    Returns:
        Dictionary with trust_score, risk_band, risk_probability, top_factors
    """
    global RISK_MODEL
    
    if RISK_MODEL is None:
        # Fallback to heuristic scoring if model not loaded
        return heuristic_risk_score(features)
    
    try:
        df = features_to_dataframe(features)
        
        # Predict probability of being high-risk
        if hasattr(RISK_MODEL, 'predict_proba'):
            proba = RISK_MODEL.predict_proba(df)[0]
            # Assuming binary classification: [prob_low_risk, prob_high_risk]
            risk_probability = proba[1] if len(proba) > 1 else proba[0]
        else:
            # If model only has predict, use a simple heuristic
            risk_probability = 0.5
        
        # Convert to trust score (0-100)
        trust_score = round((1 - risk_probability) * 100)
        
        # Determine risk band
        if trust_score >= 80:
            risk_band = "LOW"
        elif trust_score >= 60:
            risk_band = "MEDIUM"
        elif trust_score >= 40:
            risk_band = "HIGH"
        else:
            risk_band = "CRITICAL"
        
        # Extract top contributing factors
        top_factors = extract_top_factors(features)
        
        return {
            'trust_score': trust_score,
            'risk_band': risk_band,
            'risk_probability': float(risk_probability),
            'top_factors': top_factors
        }
    except Exception as e:
        logger.error(f"Error predicting risk score: {e}")
        return heuristic_risk_score(features)


def heuristic_risk_score(features: TenantRiskFeatures) -> dict:
    """
    Fallback heuristic risk scoring when model is not available.
    """
    score = 100
    
    # Deduct points for negative factors
    if features.is_suspended:
        score -= 50
    if features.n_reclamations_critical > 0:
        score -= 30 * features.n_reclamations_critical
    if features.n_reclamations_high > 0:
        score -= 20 * features.n_reclamations_high
    if features.n_reclamations_medium > 0:
        score -= 10 * features.n_reclamations_medium
    if features.failed_transaction_rate > 0.3:
        score -= 25
    if features.n_cancelled_bookings > features.n_completed_bookings:
        score -= 15
    
    # Add points for positive factors
    if features.n_completed_bookings >= 5:
        score += 10
    if features.user_rating >= 4.5:
        score += 10
    if features.has_verified_profile:
        score += 5
    
    score = max(0, min(100, score))
    
    if score >= 80:
        risk_band = "LOW"
    elif score >= 60:
        risk_band = "MEDIUM"
    elif score >= 40:
        risk_band = "HIGH"
    else:
        risk_band = "CRITICAL"
    
    top_factors = extract_top_factors(features)
    
    return {
        'trust_score': score,
        'risk_band': risk_band,
        'risk_probability': (100 - score) / 100.0,
        'top_factors': top_factors
    }


def extract_top_factors(features: TenantRiskFeatures) -> List[str]:
    """Extract top contributing factors for risk assessment."""
    factors = []
    
    if features.n_completed_bookings > 0:
        factors.append(f"{features.n_completed_bookings} completed bookings")
    if features.n_reclamations_as_target == 0:
        factors.append("No disputes")
    if features.n_transactions_failed == 0 and features.n_transactions_total > 0:
        factors.append("No failed payments")
    if features.n_reclamations_critical > 0:
        factors.append(f"{features.n_reclamations_critical} critical complaints")
    if features.n_reclamations_high > 0:
        factors.append(f"{features.n_reclamations_high} high-severity complaints")
    if features.is_suspended:
        factors.append("Account suspended")
    if features.failed_transaction_rate > 0.3:
        factors.append("High payment failure rate")
    
    return factors[:5] if factors else ["Insufficient data"]


@router.post("/batch", response_model=List[TenantRiskResponse])
def get_batch_tenant_risk_scores(tenant_ids: List[int] = Query(..., max_length=100)):
    """
    Get risk scores for multiple tenants in batch (max 100).
    """
    results = []
    for tenant_id in tenant_ids:
        try:
            features = extract_tenant_features(tenant_id)
            prediction = predict_risk_score(features)
            results.append(TenantRiskResponse(
                tenant_id=tenant_id,
                trust_score=prediction['trust_score'],
                risk_band=prediction['risk_band'],
                risk_probability=prediction['risk_probability'],
                top_factors=prediction['top_factors'],
                features=features
            ))
        except Exception as e:
            logger.error(f"Error processing tenant {tenant_id}: {e}")
            continue
    
    return results


@router.post("/{tenant_id}", response_model=TenantRiskResponse)
def get_tenant_risk_score(tenant_id: int):
    """
    Get tenant risk score (0-100 trust score) for a given tenant.
    
    Features are automatically extracted from database.
    """
    try:
        # Extract features from database
        features = extract_tenant_features(tenant_id)
        
        # Predict risk score
        prediction = predict_risk_score(features)
        
        return TenantRiskResponse(
            tenant_id=tenant_id,
            trust_score=prediction['trust_score'],
            risk_band=prediction['risk_band'],
            risk_probability=prediction['risk_probability'],
            top_factors=prediction['top_factors'],
            features=features
        )
    except Exception as e:
        logger.error(f"Error getting tenant risk score for {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compute risk score: {str(e)}")


def load_risk_model():
    """Load the trained risk model from disk."""
    global RISK_MODEL, RISK_MODEL_METADATA
    
    # Path to consolidated production models
    base_path = Path(__file__).parent.parent.parent / "models" / "production"
    model_path = base_path / "tenant_risk_model.pkl"
    
    if model_path.exists():
        try:
            RISK_MODEL = joblib.load(model_path)
            logger.info(f"✅ Tenant risk model loaded from {model_path}")
            
            # Try to load metadata
            metadata_path = base_path / "tenant_risk_model_metadata.pkl"
            if metadata_path.exists():
                RISK_MODEL_METADATA = joblib.load(metadata_path)
            else:
                RISK_MODEL_METADATA = {'version': '1.0', 'model_type': 'RandomForestClassifier'}
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load risk model from {model_path}: {e}")
            return False
    else:
        logger.warning(f"⚠️ Tenant risk model not found at {model_path}. Using heuristic scoring.")
        return False


# Initialize model on module load
load_risk_model()

