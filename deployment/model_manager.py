
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from threading import Lock
from .config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton manager for loading, caching, and serving ML models.
    Thread-safe lazy loading prevents cold start latency impact until first use,
    or can be pre-warmed on startup.
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.feature_names: Dict[str, list] = {}
        self._initialized = True
        logger.info(" ModelManager initialized")

    def _load_model(self, model_key: str, filename: str, metadata_filename: Optional[str] = None) -> Any:
        """Internal helper to load a specific model safely."""
        try:
            path = settings.MODELS_DIR / filename
            if not path.exists():
                logger.warning(f"⚠️ Model file not found: {path}")
                return None
                
            logger.info(f"Loading model '{model_key}' from {path}...")
            model = joblib.load(path)
            self.models[model_key] = model
            
            # Extract feature names if available (scikit-learn 1.0+)
            if hasattr(model, "feature_names_in_"):
                self.feature_names[model_key] = list(model.feature_names_in_)
            
            # Load metadata if exists
            if metadata_filename:
                meta_path = settings.MODELS_DIR / metadata_filename
                if meta_path.exists():
                    self.metadata[model_key] = joblib.load(meta_path)
                else:
                    self.metadata[model_key] = {"version": "unknown", "source": "local"}
            
            logger.info(f" Model '{model_key}' loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f" Failed to load model '{model_key}': {e}")
            return None

    def get_pricing_model(self):
        """Get or load the Pricing/Regression model."""
        if "pricing" not in self.models:
            with self._lock:  # Double-checked locking for thread safety
                if "pricing" not in self.models:
                    self._load_model("pricing", settings.MODEL_PRICING_FILENAME)
        return self.models.get("pricing")

    def get_clustering_model(self):
        """Get or load the Property Clustering model."""
        if "clustering" not in self.models:
            with self._lock:
                if "clustering" not in self.models:
                    self._load_model(
                        "clustering", 
                        settings.MODEL_CLUSTERING_FILENAME, 
                        "property_cluster_model_metadata.pkl"
                    )
        return self.models.get("clustering")

    def get_risk_model(self):
        """Get or load the Tenant Risk Classification model."""
        if "risk" not in self.models:
            with self._lock:
                if "risk" not in self.models:
                    self._load_model(
                        "risk", 
                        settings.MODEL_RISK_FILENAME, 
                        "tenant_risk_model_metadata.pkl"
                    )
        return self.models.get("risk")

    def get_metadata(self, model_key: str) -> Dict:
        return self.metadata.get(model_key, {})

    def get_feature_names(self, model_key: str) -> Optional[list]:
        return self.feature_names.get(model_key)

    def health_check(self) -> Dict[str, str]:
        """Return status of all managed models."""
        return {
            "pricing": "loaded" if "pricing" in self.models else "not_loaded",
            "clustering": "loaded" if "clustering" in self.models else "not_loaded",
            "risk": "loaded" if "risk" in self.models else "not_loaded",
        }

# Global instance
model_manager = ModelManager()
