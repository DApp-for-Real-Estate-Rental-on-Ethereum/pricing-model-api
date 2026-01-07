import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration via Environment Variables.
    Strictly typed and validated.
    """
    # Service Info
    APP_NAME: str = "Morocco Airbnb Dynamic Pricing API"
    APP_VERSION: str = "2.1.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = BASE_DIR / "models" / "production"
    
    # Model Filenames (can be overridden via ENV)
    MODEL_PRICING_FILENAME: str = "random_forest_tuned.pkl.gz"
    MODEL_CLUSTERING_FILENAME: str = "property_cluster_model.pkl"
    MODEL_RISK_FILENAME: str = "tenant_risk_model.pkl"
    
    # Database (Fallback to defaults if not provided)
    DB_HOST: str = "postgres-service"
    DB_PORT: str = "5432"
    DB_NAME: str = "derentdb"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "12345"
    
    # Microservices URLs (defaults for K8s)
    USER_SERVICE_URL: str = "http://user-service:8082"
    BOOKING_SERVICE_URL: str = "http://booking-service:8083"
    RECLAMATION_SERVICE_URL: str = "http://reclamation-service:8091"
    PAYMENT_SERVICE_URL: str = "http://payment-service:8085"
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
