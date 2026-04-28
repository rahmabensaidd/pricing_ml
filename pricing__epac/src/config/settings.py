import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _discover_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "pricing__epac").exists():
            return parent
    return current.parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_ROOT: Path = Field(default_factory=_discover_project_root)

    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI", "MLFLOW_URI"),
    )
    MLFLOW_S3_ENDPOINT_URL: str = Field(
        default="http://localhost:9000",
        validation_alias=AliasChoices("MLFLOW_S3_ENDPOINT_URL", "MINIO_ENDPOINT_URL"),
    )
    AWS_ACCESS_KEY_ID: str = Field(
        default="minio_admin",
        validation_alias=AliasChoices(
            "AWS_ACCESS_KEY_ID",
            "MLFLOW_S3_ACCESS_KEY",
            "MINIO_ROOT_USER",
        ),
    )
    AWS_SECRET_ACCESS_KEY: str = Field(
        default="minio_password",
        validation_alias=AliasChoices(
            "AWS_SECRET_ACCESS_KEY",
            "MLFLOW_S3_SECRET_KEY",
            "MINIO_ROOT_PASSWORD",
        ),
    )
    AWS_DEFAULT_REGION: str = "us-east-1"
    MLFLOW_BUCKET_NAME: str = "mlflow-artifacts"

    MLFLOW_POSTGRES_USER: str = "mlflow_user"
    MLFLOW_POSTGRES_PASSWORD: str = "mlflow_password"
    MLFLOW_POSTGRES_DB: str = "mlflow_db"
    MLFLOW_POSTGRES_HOST: str = "localhost"
    MLFLOW_POSTGRES_PORT: int = 5433

    ALIAS_PRODUCTION: str = "production"
    MODEL_NAME_GLOBAL: str = "PricingModelGlobal"
    MODEL_NAME_CLIENT_FEATURES: str = "ClientFeatures"

    TAG_FORMULA: str = "linear_formula"
    TAG_FEATURE_IMPORTANCE: str = "feature_importance"
    TAG_SHAP_SUCCESS: str = "shap_success"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @property
    def PACKAGE_ROOT(self) -> Path:
        return self.PROJECT_ROOT / "pricing__epac"

    @property
    def DATA_ROOT(self) -> Path:
        return self.PACKAGE_ROOT / "data"

    @property
    def ARTIFACTS_ROOT(self) -> Path:
        return self.PACKAGE_ROOT / "artifacts"

    @property
    def MODELS_ARTIFACT_ROOT(self) -> Path:
        return self.ARTIFACTS_ROOT / "models"

    @property
    def DVC_TRACKING_ROOT(self) -> Path:
        return self.ARTIFACTS_ROOT / "dvc_tracking"

    @property
    def RUNTIME_ROOT(self) -> Path:
        return self.PACKAGE_ROOT / "runtime"

    @property
    def RUNTIME_LOGS_ROOT(self) -> Path:
        return self.RUNTIME_ROOT / "logs"

    @property
    def WATCHER_RUNTIME_ROOT(self) -> Path:
        return self.RUNTIME_ROOT / "watcher"

    @property
    def PIPELINE_RESULTS_ROOT(self) -> Path:
        return self.RUNTIME_ROOT / "pipeline_results"

    @property
    def WATCHER_TRACKING_FILE(self) -> Path:
        return self.WATCHER_RUNTIME_ROOT / "dumps_tracking.json"

    @property
    def SQL_WATCH_DIR(self) -> Path:
        return self.DATA_ROOT / "raw" / "dumps" / "sql"

    @property
    def MLFLOW_BACKEND_STORE_URI(self) -> str:
        return (
            "postgresql://"
            f"{self.MLFLOW_POSTGRES_USER}:{self.MLFLOW_POSTGRES_PASSWORD}"
            f"@{self.MLFLOW_POSTGRES_HOST}:{self.MLFLOW_POSTGRES_PORT}/{self.MLFLOW_POSTGRES_DB}"
        )


settings = Settings()

os.environ["PROJECT_ROOT"] = str(settings.PROJECT_ROOT)
os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION
