#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data loading, validation, configuration, and persistence helpers."""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from dotenv import load_dotenv

from pricing__epac.src.config.settings import settings
from pricing__epac.src.shared.logging import configure_logging


PROJECT_ROOT = settings.PROJECT_ROOT
MODELS_DIR = settings.MODELS_ARTIFACT_ROOT
DATA_DIR = settings.DATA_ROOT / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file() -> bool:
    """Load .env file from project root."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        return True
    return False


load_env_file()


@dataclass
class TrainingConfig:
    """Configuration for the global training pipeline."""

    target_column: str = "unit_price"
    test_size: float = float(os.getenv("TRAIN_TEST_SPLIT", 0.25))
    random_state: int = int(os.getenv("RANDOM_STATE", 42))
    cv_folds: int = int(os.getenv("CV_FOLDS", 5))
    min_target_value: float = 0.5

    numeric_columns: List[str] = field(default_factory=lambda: [
        "quantity", "production_page", "height", "thickness", "width",
        "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
        "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill",
    ])

    categorical_columns: List[str] = field(default_factory=lambda: [
        "text_paper_type", "text_color", "cover_finish_type", "cover_color",
        "cover_size", "cover_paper_type", "head_and_tail", "priority_level",
        "binding_type", "siren",
    ])

    boolean_columns: List[str] = field(default_factory=lambda: [
        "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
        "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill",
    ])

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow_s3_endpoint: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", settings.MLFLOW_S3_ENDPOINT_URL)
    mlflow_s3_access_key: str = os.getenv("MLFLOW_S3_ACCESS_KEY", settings.AWS_ACCESS_KEY_ID)
    mlflow_s3_secret_key: str = os.getenv("MLFLOW_S3_SECRET_KEY", settings.AWS_SECRET_ACCESS_KEY)
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", settings.AWS_DEFAULT_REGION)
    max_features_for_importance: int = 50


def setup_logging() -> logging.Logger:
    warnings.filterwarnings("ignore")
    return configure_logging(level=logging.INFO, logger_name=__name__)


logger = setup_logging()


def load_and_validate_data(file_path: Path, config: TrainingConfig) -> pd.DataFrame:
    """Load training data and validate required schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {file_path}")

    df = pd.read_excel(file_path)
    required = set(config.numeric_columns + config.categorical_columns + [config.target_column])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input dataset is empty")

    return df


def clean_data(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Apply deterministic cleaning rules used by the legacy training pipeline."""
    cleaned = df.copy()

    cleaned = cleaned[cleaned[config.target_column] > config.min_target_value].copy()
    if cleaned.empty:
        raise ValueError("No rows left after target filtering")

    for col in config.numeric_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in config.boolean_columns:
        cleaned[col] = cleaned[col].astype(int)

    for col in config.categorical_columns:
        cleaned[col] = cleaned[col].astype(str).fillna("missing")
        cleaned[col] = cleaned[col].replace({"nan": "missing", "None": "missing"})

    cleaned[config.target_column] = pd.to_numeric(cleaned[config.target_column], errors="coerce")
    cleaned = cleaned.dropna(subset=[config.target_column])
    if cleaned.empty:
        raise ValueError("No rows left after cleaning")
    return cleaned


def save_training_outputs(
    best_pipeline,
    results: List[Dict],
    feature_importance: Dict[str, float],
    best_model_name: str,
    best_rmse: float,
    feature_cols: List[str],
    X,
    config: TrainingConfig,
    timestamp: str,
) -> Path:
    """Persist local artifacts and metadata for the best run."""
    save_dir = MODELS_DIR / f"best_{best_model_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, save_dir / "pipeline_complete.joblib")
    pd.DataFrame(results).to_csv(save_dir / "comparison_results.csv", index=False)

    if feature_importance:
        with open(save_dir / "feature_importance.json", "w", encoding="utf-8") as handle:
            json.dump(feature_importance, handle, indent=2)

    metadata = {
        "best_model": best_model_name,
        "best_rmse": best_rmse,
        "timestamp": timestamp,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "num_cols": config.numeric_columns,
        "cat_cols": config.categorical_columns,
        "has_feature_importance": bool(feature_importance),
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return save_dir
