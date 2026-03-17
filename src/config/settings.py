import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # MinIO/S3 settings
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "minio_admin")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_password")
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Model constants
    ALIAS_PRODUCTION: str = os.getenv("ALIAS_PRODUCTION", "production")
    MODEL_NAME_GLOBAL: str = os.getenv("MODEL_NAME_GLOBAL", "PricingModelGlobal")
    MODEL_NAME_CLIENT_FEATURES: str = os.getenv("MODEL_NAME_CLIENT_FEATURES", "ClientFeatures")

    # Tags for additional information
    TAG_FORMULA: str = "linear_formula"
    TAG_FEATURE_IMPORTANCE: str = "feature_importance"
    TAG_SHAP_SUCCESS: str = "shap_success"

    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Set environment variables for MinIO/S3
os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MLFLOW_S3_ENDPOINT_URL
os.environ['AWS_DEFAULT_REGION'] = settings.AWS_DEFAULT_REGION