"""
Tenant Risk Scoring Module
==========================

Provides 0-100 trust score for tenants using RandomForestClassifier.
Features extracted from bookings, reclamations, transactions, and user data.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# from deployment.db_connection import execute_query, execute_query_single
from deployment.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenant-risk", tags=["Tenant Risk Scoring"])


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


from deployment.external_services import external_services

async def extract_tenant_features(user_id: int) -> TenantRiskFeatures:
    """
    Extract all features for a tenant via microservice APIs.
    
    Args:
        user_id: User ID of the tenant
        
    Returns:
        TenantRiskFeatures object with all computed features
    """
    features = TenantRiskFeatures()
    
    # 1. Fetch User Stats (Parallel execution would be better but keeping it simple for now)
    user_data = await external_services.get_user_stats(user_id)
    features.user_rating = float(user_data.get('rating') or 0.0)
    features.user_score = int(user_data.get('score') or 100)
    # features.user_penalty_points = int(user_data.get('penalty_points') or 0) # Not currently in UserStatsDTO
    features.has_verified_profile = bool(user_data.get('verified') or False)
    
    # Account age logic (approximation if not provided by service)
    # If createdAt is provided by service in future, use it. For now default 0.
    features.account_age_days = 0 

    # 2. Fetch Booking Stats
    booking_stats = await external_services.get_booking_stats(user_id)
    features.n_bookings_total = int(booking_stats.get('total') or 0)
    features.n_completed_bookings = int(booking_stats.get('completed') or 0)
    features.n_cancelled_bookings = int(booking_stats.get('cancelled') or 0)
    features.avg_booking_value = float(booking_stats.get('avgPrice') or 0.0)
    features.avg_stay_length_days = float(booking_stats.get('avgStayDays') or 0.0)
    features.recent_bookings_last_6m = int(booking_stats.get('recentLast6Months') or 0)

    # 3. Fetch Reclamation Stats
    reclamation_stats = await external_services.get_reclamation_stats(user_id)
    # Note: ReclamationStatsDTO needs strict mapping
    features.n_reclamations_as_target = int(reclamation_stats.get('totalReceived') or 0)
    features.n_reclamations_low = int(reclamation_stats.get('receivedLowSeverity') or 0)
    features.n_reclamations_medium = int(reclamation_stats.get('receivedMediumSeverity') or 0)
    features.n_reclamations_high = int(reclamation_stats.get('receivedHighSeverity') or 0)
    features.n_reclamations_critical = int(reclamation_stats.get('receivedCriticalSeverity') or 0)
    features.n_reclamations_open = int(reclamation_stats.get('receivedOpen') or 0)
    features.n_reclamations_resolved_against_user = int(reclamation_stats.get('receivedResolvedAgainst') or 0)
    features.total_penalty_points = int(reclamation_stats.get('totalPenaltyPoints') or 0)
    features.total_refund_amount = float(reclamation_stats.get('totalRefundAmount') or 0.0)

    # 4. Fetch Payment Stats
    payment_stats = await external_services.get_payment_stats(user_id)
    features.n_transactions_total = int(payment_stats.get('totalTransactions') or 0)
    features.n_transactions_success = int(payment_stats.get('successfulTransactions') or 0)
    features.n_transactions_failed = int(payment_stats.get('failedTransactions') or 0)
    features.avg_transaction_amount = float(payment_stats.get('avgTransactionAmount') or 0.0)
    
    if features.n_transactions_total > 0:
        features.failed_transaction_rate = features.n_transactions_failed / features.n_transactions_total
    
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


def predict_risk_score(features: TenantRiskFeatures) -> dict:
    """
    Predict tenant risk score using the trained model.
    
    Returns:
        Dictionary with trust_score, risk_band, risk_probability, top_factors
    """
    risk_model = model_manager.get_risk_model()
    
    if risk_model is None:
        # Fallback to heuristic scoring if model not loaded
        return heuristic_risk_score(features)
    
    try:
        df = features_to_dataframe(features)
        
        # Predict probability of being high-risk
        if hasattr(risk_model, 'predict_proba'):
            proba = risk_model.predict_proba(df)[0]
            # Assuming binary classification: [prob_low_risk, prob_high_risk]
            risk_probability = proba[1] if len(proba) > 1 else proba[0]
        else:
            # If model only has predict, use a simple heuristic
            risk_probability = 0.5
        
        # Convert to trust score (0-100)
        model_trust_score = round((1 - risk_probability) * 100)
        
        # Get Heuristic Score for calibration
        heuristic = heuristic_risk_score(features)
        heuristic_score = heuristic['trust_score']
        
        # Weighted Blend (40% Model, 60% Heuristic) to stabilize
        trust_score = int((model_trust_score * 0.4) + (heuristic_score * 0.6))
        
        # Recalculate Risk Probability
        risk_probability = 1.0 - (trust_score / 100.0)
        
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
            'top_factors': top_factors,
            'calibration': 'blended'
        }
    except Exception as e:
        logger.error(f"Error predicting risk score: {e}")
        return heuristic_risk_score(features)


@router.post("/batch", response_model=List[TenantRiskResponse])
async def get_batch_tenant_risk_scores(tenant_ids: List[int] = Query(..., max_length=100)):
    """
    Get risk scores for multiple tenants in batch (max 100).
    """
    results = []
    for tenant_id in tenant_ids:
        try:
            features = await extract_tenant_features(tenant_id)
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
async def get_tenant_risk_score(tenant_id: int):
    """
    Get tenant risk score (0-100 trust score) for a given tenant.
    
    Features are automatically extracted from microservices.
    """
    try:
        # Extract features from microservices
        features = await extract_tenant_features(tenant_id)
        
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
