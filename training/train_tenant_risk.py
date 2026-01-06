"""
Training Script for Tenant Risk Scoring Model
==============================================

Trains a RandomForestClassifier to predict tenant risk (0-100 trust score).
Uses GridSearchCV for hyperparameter tuning and evaluates with multiple metrics.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deployment"))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import logging
from db_connection import execute_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """
    Load training data from database.
    
    Extracts features for all users with role TENANT.
    """
    logger.info("Loading training data from database...")
    
    query = """
        SELECT 
            u.id as user_id,
            -- Booking features
            (SELECT COUNT(*) FROM bookings WHERE user_id = u.id) as n_bookings_total,
            (SELECT COUNT(*) FROM bookings WHERE user_id = u.id 
             AND status IN ('COMPLETED', 'TENANT_CHECKED_OUT', 'CONFIRMED')) as n_completed_bookings,
            (SELECT COUNT(*) FROM bookings WHERE user_id = u.id 
             AND (status LIKE '%CANCELLED%' OR status = 'CANCELLED_BY_TENANT')) as n_cancelled_bookings,
            (SELECT AVG(total_price) FROM bookings WHERE user_id = u.id) as avg_booking_value,
            (SELECT AVG(EXTRACT(EPOCH FROM (check_out_date::timestamp - check_in_date::timestamp)) / 86400) 
             FROM bookings WHERE user_id = u.id AND check_out_date IS NOT NULL AND check_in_date IS NOT NULL) as avg_stay_length_days,
            (SELECT COUNT(*) FROM bookings WHERE user_id = u.id 
             AND created_at >= NOW() - INTERVAL '6 months') as recent_bookings_last_6m,
            
            -- Reclamation features
            (SELECT COUNT(*) FROM reclamations WHERE target_user_id = u.id) as n_reclamations_as_target,
            (SELECT COUNT(*) FROM reclamations WHERE target_user_id = u.id AND severity = 'LOW') as n_reclamations_low,
            (SELECT COUNT(*) FROM reclamations WHERE target_user_id = u.id AND severity = 'MEDIUM') as n_reclamations_medium,
            (SELECT COUNT(*) FROM reclamations WHERE target_user_id = u.id AND severity = 'HIGH') as n_reclamations_high,
            (SELECT COUNT(*) FROM reclamations WHERE target_user_id = u.id AND severity = 'CRITICAL') as n_reclamations_critical,
            (SELECT COUNT(*) FROM reclamations WHERE target_user_id = u.id 
             AND (status = 'OPEN' OR status = 'IN_REVIEW')) as n_reclamations_open,
            (SELECT COUNT(*) FROM reclamations WHERE target_user_id = u.id 
             AND status = 'RESOLVED' AND resolution_notes LIKE '%against%') as n_reclamations_resolved_against_user,
            (SELECT SUM(penalty_points) FROM reclamations WHERE target_user_id = u.id) as total_penalty_points,
            (SELECT SUM(refund_amount) FROM reclamations WHERE target_user_id = u.id) as total_refund_amount,
            
            -- Transaction features
            (SELECT COUNT(*) FROM transactions WHERE user_id = u.id) as n_transactions_total,
            (SELECT COUNT(*) FROM transactions WHERE user_id = u.id AND status = 'SUCCESS') as n_transactions_success,
            (SELECT COUNT(*) FROM transactions WHERE user_id = u.id AND status = 'FAILED') as n_transactions_failed,
            (SELECT AVG(amount) FROM transactions WHERE user_id = u.id) as avg_transaction_amount,
            
            -- User attributes
            u.rating as user_rating,
            u.score as user_score,
            u.penalty_points as user_penalty_points,
            u.is_suspended as is_suspended,
            EXTRACT(EPOCH FROM (NOW() - COALESCE(u.verification_expiration::timestamp, NOW()))) / 86400 as account_age_days,
            (SELECT is_complete FROM user_profile_status WHERE user_id = u.id::text) as has_verified_profile
        FROM users u
        JOIN users_roles ur ON ur.user_id = u.id
        WHERE ur.role = 'TENANT'
    """
    
    results = execute_query(query)
    
    if not results:
        logger.warning("No data found. Using sample data for demonstration.")
        return create_sample_data()
    
    df = pd.DataFrame(results)
    
    # Calculate derived features
    df['failed_transaction_rate'] = df.apply(
        lambda row: (row['n_transactions_failed'] / row['n_transactions_total']) 
        if row['n_transactions_total'] > 0 else 0.0, axis=1
    )
    
    # Fill NaN values (suppress FutureWarning)
    with pd.option_context('future.no_silent_downcasting', True):
        df = df.fillna(0)
    
    logger.info(f"Loaded {len(df)} tenant records")
    return df


def create_sample_data() -> pd.DataFrame:
    """Create sample data for demonstration if database is empty."""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'user_id': range(1, n_samples + 1),
        'n_bookings_total': np.random.poisson(5, n_samples),
        'n_completed_bookings': np.random.poisson(4, n_samples),
        'n_cancelled_bookings': np.random.poisson(1, n_samples),
        'avg_booking_value': np.random.normal(500, 200, n_samples),
        'avg_stay_length_days': np.random.normal(7, 3, n_samples),
        'recent_bookings_last_6m': np.random.poisson(2, n_samples),
        'n_reclamations_as_target': np.random.poisson(0.5, n_samples),
        'n_reclamations_low': np.random.poisson(0.3, n_samples),
        'n_reclamations_medium': np.random.poisson(0.2, n_samples),
        'n_reclamations_high': np.random.poisson(0.1, n_samples),
        'n_reclamations_critical': np.random.poisson(0.05, n_samples),
        'n_reclamations_open': np.random.poisson(0.2, n_samples),
        'n_reclamations_resolved_against_user': np.random.poisson(0.1, n_samples),
        'total_penalty_points': np.random.poisson(5, n_samples),
        'total_refund_amount': np.random.normal(100, 50, n_samples),
        'n_transactions_total': np.random.poisson(5, n_samples),
        'n_transactions_success': np.random.poisson(4, n_samples),
        'n_transactions_failed': np.random.poisson(1, n_samples),
        'failed_transaction_rate': np.random.beta(2, 8, n_samples),
        'avg_transaction_amount': np.random.normal(500, 200, n_samples),
        'user_rating': np.random.normal(4.2, 0.8, n_samples).clip(0, 5),
        'user_score': np.random.normal(85, 15, n_samples).clip(0, 100),
        'user_penalty_points': np.random.poisson(2, n_samples),
        'is_suspended': np.random.binomial(1, 0.05, n_samples),
        'account_age_days': np.random.normal(180, 90, n_samples),
        'has_verified_profile': np.random.binomial(1, 0.7, n_samples),
    }
    
    return pd.DataFrame(data)


def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create binary labels for training (high-risk vs low-risk).
    
    Heuristic: Mark as high-risk if:
    - is_suspended = True, OR
    - n_reclamations_critical >= 2, OR
    - n_reclamations_high >= 3, OR
    - failed_transaction_rate > 0.3, OR
    - user_penalty_points > 20
    """
    labels = (
        (df['is_suspended'] == 1) |
        (df['n_reclamations_critical'] >= 2) |
        (df['n_reclamations_high'] >= 3) |
        (df['failed_transaction_rate'] > 0.3) |
        (df['user_penalty_points'] > 20)
    ).astype(int)
    
    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
    return labels


def train_model():
    """Train the tenant risk model with hyperparameter tuning."""
    logger.info("=" * 60)
    logger.info("TRAINING TENANT RISK SCORING MODEL")
    logger.info("=" * 60)
    
    # Load data
    df = load_training_data()
    
    # Feature columns (exclude user_id and labels)
    feature_cols = [col for col in df.columns if col != 'user_id']
    
    X = df[feature_cols].values
    y = create_labels(df)
    
    # Handle class imbalance
    if y.sum() < len(y) * 0.1:
        logger.warning("Very imbalanced dataset. Consider using class_weight='balanced'")
    
    # Check if we have enough samples for cross-validation
    min_samples_for_cv = 10  # Need at least 10 samples for 5-fold CV
    
    if len(X) < min_samples_for_cv:
        logger.warning(f"Only {len(X)} samples found. Need at least {min_samples_for_cv} for proper training.")
        logger.warning("Using sample data for demonstration...")
        df_sample = create_sample_data()
        X_sample = df_sample[[col for col in df_sample.columns if col != 'user_id']].values
        y_sample = create_labels(df_sample)
        X = np.vstack([X, X_sample]) if len(X) > 0 else X_sample
        y = np.hstack([y, y_sample]) if len(y) > 0 else y_sample
        logger.info(f"Using {len(X)} total samples (real + synthetic)")
    
    # Split data
    if len(X) >= 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    else:
        # If too few samples, use all for training
        logger.warning("Too few samples for train/test split. Using all data for training.")
        X_train, X_test = X, X
        y_train, y_test = y, y
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Determine CV folds based on sample size
    n_splits = min(5, max(2, len(X_train) // 2))
    if n_splits < 2:
        n_splits = 2
        logger.warning(f"Using {n_splits}-fold CV (limited by sample size)")
    
    # Hyperparameter grid for RandomForest
    param_grid_rf = {
        'n_estimators': [100, 200] if len(X_train) >= 10 else [100],
        'max_depth': [10, 20, None] if len(X_train) >= 10 else [10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    
    # Train RandomForest with GridSearch
    logger.info(f"Training RandomForestClassifier with GridSearchCV ({n_splits}-fold CV)...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search_rf = GridSearchCV(
        rf, param_grid_rf, cv=n_splits, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search_rf.fit(X_train_scaled, y_train)
    
    logger.info(f"Best RandomForest params: {grid_search_rf.best_params_}")
    logger.info(f"Best RandomForest CV score: {grid_search_rf.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred_rf = grid_search_rf.predict(X_test_scaled)
    y_proba_rf = grid_search_rf.predict_proba(X_test_scaled)[:, 1]
    
    logger.info("\nRandomForest Test Results:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    if len(np.unique(y_test)) > 1:
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
    else:
        logger.warning("Cannot calculate ROC-AUC (only one class in test set)")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred_rf))
    
    # Try GradientBoosting as alternative
    logger.info(f"\nTraining GradientBoostingClassifier ({n_splits}-fold CV)...")
    param_grid_gb = {
        'n_estimators': [100, 200] if len(X_train) >= 10 else [100],
        'max_depth': [3, 5, 7] if len(X_train) >= 10 else [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search_gb = GridSearchCV(
        gb, param_grid_gb, cv=n_splits, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search_gb.fit(X_train_scaled, y_train)
    
    logger.info(f"Best GradientBoosting params: {grid_search_gb.best_params_}")
    logger.info(f"Best GradientBoosting CV score: {grid_search_gb.best_score_:.4f}")
    
    y_pred_gb = grid_search_gb.predict(X_test_scaled)
    y_proba_gb = grid_search_gb.predict_proba(X_test_scaled)[:, 1]
    
    logger.info("\nGradientBoosting Test Results:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
    if len(np.unique(y_test)) > 1:
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba_gb):.4f}")
    else:
        logger.warning("Cannot calculate ROC-AUC (only one class in test set)")
    
    # Choose best model
    if len(np.unique(y_test)) > 1:
        rf_score = roc_auc_score(y_test, y_proba_rf)
        gb_score = roc_auc_score(y_test, y_proba_gb)
    else:
        # If only one class, use accuracy or CV score
        rf_score = grid_search_rf.best_score_
        gb_score = grid_search_gb.best_score_
    
    if rf_score >= gb_score:
        best_model = grid_search_rf.best_estimator_
        model_name = "RandomForest"
        best_score = rf_score
    else:
        best_model = grid_search_gb.best_estimator_
        model_name = "GradientBoosting"
        best_score = gb_score
    
    logger.info(f"\n✅ Selected {model_name} model (ROC-AUC: {best_score:.4f})")
    
    # Save model and metadata
    models_dir = Path(__file__).parent.parent / "deployment" / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "tenant_risk_model.pkl"
    scaler_path = models_dir / "tenant_risk_scaler.pkl"
    metadata_path = models_dir / "tenant_risk_model_metadata.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    test_roc_auc = best_score if len(np.unique(y_test)) > 1 else None
    test_accuracy = float(accuracy_score(y_test, best_model.predict(X_test_scaled)))
    
    metadata = {
        'model_type': model_name,
        'version': '1.0',
        'feature_columns': feature_cols,
        'test_roc_auc': float(test_roc_auc) if test_roc_auc is not None else None,
        'test_accuracy': test_accuracy,
        'cv_score': float(grid_search_rf.best_score_ if model_name == "RandomForest" else grid_search_gb.best_score_),
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'class_distribution': pd.Series(y).value_counts().to_dict(),
        'note': 'Trained with limited data - consider retraining when more samples available'
    }
    
    joblib.dump(metadata, metadata_path)
    
    logger.info(f"\n✅ Model saved to {model_path}")
    logger.info(f"✅ Scaler saved to {scaler_path}")
    logger.info(f"✅ Metadata saved to {metadata_path}")
    
    return best_model, scaler, metadata


if __name__ == "__main__":
    train_model()

