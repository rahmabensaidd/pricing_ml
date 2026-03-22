#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EPAC PRICING MODEL TRAINING PIPELINE
================================================================================

A comprehensive, production-ready model training pipeline that trains and compares
multiple regression models for pricing prediction.

This pipeline handles:
- Data loading and validation
- Feature engineering and preprocessing
- Training and comparing 15+ regression models
- MLflow integration for experiment tracking
- Feature importance extraction
- Model versioning and registration

================================================================================
PROCESSING FLOWCHART
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 1: LOAD AND VALIDATE DATA                                        │
    │ - Load cleaned Excel file                                             │
    │ - Validate required columns exist                                     │
    │ - Check data quality (missing values, outliers)                       │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 2: DATA CLEANING                                                 │
    │ - Remove target outliers (unit_price > 0.5)                          │
    │ - Handle missing values (numeric: median, categorical: 'missing')    │
    │ - Convert boolean columns to int (0/1)                               │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 3: FEATURE PREPROCESSING                                         │
    │ - Numeric features: StandardScaler                                   │
    │ - Categorical features: OneHotEncoder (min_frequency=10)             │
    │ - Log transform target (unit_price → log(unit_price + 1))            │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 4: TRAIN/TEST SPLIT (Stratified by target quantiles)            │
    │ - 75% training, 25% test                                             │
    │ - Stratified sampling to maintain distribution                       │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 5: TRAIN ALL MODELS (15+ algorithms)                            │
    │ ┌────────────────────────────────────────────────────────────────┐   │
    │ │ Tree-based: RandomForest, XGBoost, LightGBM, GradientBoosting, │   │
    │ │             ExtraTrees, CatBoost                               │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ Ensemble: AdaBoost, Bagging                                    │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ Linear/SVM: SVR (RBF, Poly)                                    │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ Neural: MLPRegressor                                           │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ Others: GaussianProcess, KernelRidge, DecisionTree             │   │
    │ └────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 6: CROSS-VALIDATION & EVALUATION                                 │
    │ - 5-fold cross-validation for stability                              │
    │ - Metrics: RMSE, R², MAE, MAPE                                       │
    │ - Compare all models, select best                                    │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 7: FEATURE IMPORTANCE EXTRACTION                                 │
    │ - Extract importance from best model                                 │
    │ - Support tree-based, linear, CatBoost                               │
    │ - Save top 50 features                                               │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 8: MLFLOW REGISTRATION                                           │
    │ - Log parameters, metrics, models                                    │
    │ - Create custom signature                                            │
    │ - Register model with versioning                                     │
    │ - Add tags and aliases                                               │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 9: SAVE & REPORT                                                 │
    │ - Save best model locally                                            │
    │ - Save comparison results                                            │
    │ - Generate prediction examples                                       │
    │ - Export metrics and metadata                                        │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
ENVIRONMENT VARIABLES (loaded from .env)
================================================================================

Required:
    MLFLOW_S3_ACCESS_KEY    - MinIO access key (default: minio_admin)
    MLFLOW_S3_SECRET_KEY    - MinIO secret key (default: minio_password)

Optional:
    MLFLOW_TRACKING_URI     - MLflow server URI (default: http://localhost:5000)
    MLFLOW_S3_ENDPOINT_URL  - S3 endpoint URL (default: http://localhost:9000)
    AWS_DEFAULT_REGION      - AWS region (default: us-east-1)
    TRAIN_TEST_SPLIT        - Test size ratio (default: 0.25)
    RANDOM_STATE            - Random seed (default: 42)
    CV_FOLDS                - Number of CV folds (default: 5)

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import os
import warnings
import random
import logging
import joblib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv


# ========== LOAD ENVIRONMENT VARIABLES ==========
def load_env_file():
    """Load .env file from project root"""
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    env_file = project_root / '.env'

    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ .env file loaded from: {env_file}")
        return True
    else:
        print(f"⚠️ .env file not found at: {env_file}")
        return False


load_env_file()


# ========== END ENV LOADING ==========

# ========== CONFIGURATION ==========
@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Data settings
    target_column: str = "unit_price"
    test_size: float = float(os.getenv('TRAIN_TEST_SPLIT', 0.25))
    random_state: int = int(os.getenv('RANDOM_STATE', 42))
    cv_folds: int = int(os.getenv('CV_FOLDS', 5))
    min_target_value: float = 0.5

    # Feature columns
    numeric_columns: List[str] = field(default_factory=lambda: [
        "quantity", "production_page", "height", "thickness", "width",
        "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
        "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill"
    ])

    categorical_columns: List[str] = field(default_factory=lambda: [
        "text_paper_type", "text_color", "cover_finish_type", "cover_color",
        "cover_size", "cover_paper_type", "head_and_tail", "priority_level",
        "binding_type", "siren"
    ])

    # Boolean columns (treated as numeric)
    boolean_columns: List[str] = field(default_factory=lambda: [
        "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
        "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill"
    ])

    # MLflow settings
    mlflow_tracking_uri: str = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow_s3_endpoint: str = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
    mlflow_s3_access_key: str = os.getenv('MLFLOW_S3_ACCESS_KEY', 'minio_admin')
    mlflow_s3_secret_key: str = os.getenv('MLFLOW_S3_SECRET_KEY', 'minio_password')
    aws_region: str = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

    # Model settings
    max_features_for_importance: int = 50
    catboost_available: bool = False


# ========== LOGGING SETUP ==========
def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')
    return logging.getLogger(__name__)


logger = setup_logging()

# ========== PATH CONFIGURATION ==========
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ========== MLFLOW CONFIGURATION ==========
os.environ['URLLIB3_USE_PYOPENSSL'] = '0'
warnings.filterwarnings('ignore', module='urllib3.contrib.pyopenssl')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('MLFLOW_S3_ACCESS_KEY', 'minio_admin')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('MLFLOW_S3_SECRET_KEY', 'minio_password')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# ========== MODEL IMPORTS ==========
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
)

# Tree-based models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor

# SVM and Neural Networks
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge

# Boosting libraries
import xgboost as xgb
import lightgbm as lgb

# CatBoost (optional)
try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
    logger.info("✅ CatBoost installed and available")
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("⚠️ CatBoost not installed. Install with: pip install catboost")

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


class MLflowRunManager:
    """Context manager for MLflow runs"""

    def __init__(self, run_name: str, config: TrainingConfig):
        self.run_name = run_name
        self.config = config
        self.run = None

    def __enter__(self):
        # Terminate any active run
        try:
            mlflow.end_run()
        except:
            pass

        # Start new run
        self.run = mlflow.start_run(run_name=self.run_name)
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()


def safe_mlflow_start_run(run_name: str):
    """Start MLflow run safely"""
    mlflow.end_run()
    return mlflow.start_run(run_name=run_name)


def create_custom_signature(
        input_df: pd.DataFrame,
        model,
        config: TrainingConfig
) -> ModelSignature:
    """
    Create custom signature with correct data types.

    Args:
        input_df: Sample input data
        model: Trained model
        config: Training configuration

    Returns:
        Model signature for MLflow
    """
    schema_list = []

    # Numeric columns (double)
    for col in ["height", "thickness", "width"]:
        if col in input_df.columns:
            schema_list.append(ColSpec("double", col, required=True))

    # Integer columns (long)
    for col in ["quantity", "production_page"]:
        if col in input_df.columns:
            schema_list.append(ColSpec("long", col, required=True))

    # Boolean columns (long)
    for col in config.boolean_columns:
        if col in input_df.columns:
            schema_list.append(ColSpec("long", col, required=True))

    # Categorical columns (string)
    for col in config.categorical_columns:
        if col in input_df.columns:
            schema_list.append(ColSpec("string", col, required=True))

    # Create input schema
    input_schema = Schema(schema_list)

    # Output signature
    sample_output = model.predict(input_df.head(5))
    output_schema = infer_signature(sample_output).inputs

    return ModelSignature(inputs=input_schema, outputs=output_schema)


def extract_feature_importance(
        pipeline: Pipeline,
        feature_cols: List[str],
        config: TrainingConfig
) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Extract feature importance from the best model.

    Args:
        pipeline: Trained pipeline
        feature_cols: Original feature names
        config: Training configuration

    Returns:
        Tuple of (feature_importance_dict, json_string)
    """
    feature_importance = {}

    try:
        model_step = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']

        # Get feature names after transformation
        try:
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = feature_cols
        except:
            feature_names = [f"feature_{i}" for i in range(1000)]

        # Extract importance based on model type
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            for i, imp in enumerate(importances):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                feature_importance[name] = float(imp)
            logger.info("Feature importance extracted (tree-based model)")

        elif hasattr(model_step, 'coef_'):
            coefs = model_step.coef_.flatten() if len(model_step.coef_.shape) > 1 else model_step.coef_
            for i, coef in enumerate(np.abs(coefs)):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                feature_importance[name] = float(coef)
            logger.info("Coefficients extracted (linear model)")

        elif hasattr(model_step, 'get_feature_importance'):
            try:
                importances = model_step.get_feature_importance()
                for i, imp in enumerate(importances):
                    name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    feature_importance[name] = float(imp)
                logger.info("Feature importance extracted (CatBoost)")
            except:
                logger.warning("Unable to extract CatBoost importances")
                return {}, None

        else:
            logger.warning("Model has no feature_importances_ or coef_")
            return {}, None

        # Sort and keep top features
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:config.max_features_for_importance])

        # Log top features
        logger.info(f"Top 5 features:")
        for i, (feat, imp) in enumerate(list(sorted_importance.items())[:5]):
            logger.info(f"  {i + 1}. {feat}: {imp:.4f}")

        feature_importance_json = json.dumps(sorted_importance)
        return sorted_importance, feature_importance_json

    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
        return {}, None


def load_and_validate_data(
        file_path: Path,
        config: TrainingConfig
) -> pd.DataFrame:
    """
    Load and validate input data.

    Args:
        file_path: Path to input file
        config: Training configuration

    Returns:
        Validated DataFrame
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:
        raise ValueError(f"File too large: {file_size_mb:.2f} MB")

    logger.info(f"Loading data: {file_path}")
    df = pd.read_excel(file_path, engine="openpyxl")
    logger.info(f"Initial shape: {df.shape}")

    # Check required columns
    all_required = config.numeric_columns + config.categorical_columns + [config.target_column]
    missing_cols = [col for col in all_required if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")

    # Filter to existing columns
    config.numeric_columns = [col for col in config.numeric_columns if col in df.columns]
    config.categorical_columns = [col for col in config.categorical_columns if col in df.columns]

    logger.info(f"Using {len(config.numeric_columns)} numeric columns")
    logger.info(f"Using {len(config.categorical_columns)} categorical columns")

    return df


def clean_data(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """
    Clean the data.

    Args:
        df: Input DataFrame
        config: Training configuration

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    initial_shape = df.shape

    # Remove target outliers
    df = df[df[config.target_column] > config.min_target_value].copy()
    df = df[np.isfinite(df[config.target_column])].copy()

    # Remove rows with too many missing values
    df = df.dropna(thresh=len(df.columns) - 5)

    # Fill missing values
    for col in config.numeric_columns:
        if col in df.columns:
            if col in config.boolean_columns:
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(df[col].median()).astype(float)

    for col in config.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)

    logger.info(f"Cleaned shape: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)")
    return df


def create_preprocessor(config: TrainingConfig) -> ColumnTransformer:
    """
    Create preprocessing pipeline.

    Args:
        config: Training configuration

    Returns:
        ColumnTransformer for preprocessing
    """
    transformers = []

    if config.numeric_columns:
        transformers.append(("num", StandardScaler(), config.numeric_columns))

    if config.categorical_columns:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
            config.categorical_columns
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


def get_model_configs(config: TrainingConfig) -> List[Dict[str, Any]]:
    """
    Get model configurations.

    Args:
        config: Training configuration

    Returns:
        List of model configurations
    """
    models = [
        {
            "name": "RandomForest",
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 200, "max_depth": 15, "min_samples_split": 10,
                "min_samples_leaf": 5, "random_state": config.random_state, "n_jobs": -1
            }
        },
        {
            "name": "XGBoost",
            "model": xgb.XGBRegressor,
            "params": {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": config.random_state,
                "n_jobs": -1, "verbosity": 0
            }
        },
        {
            "name": "LightGBM",
            "model": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": config.random_state,
                "n_jobs": -1, "verbose": -1
            }
        },
        {
            "name": "GradientBoosting",
            "model": GradientBoostingRegressor,
            "params": {
                "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8, "random_state": config.random_state
            }
        },
        {
            "name": "ExtraTrees",
            "model": ExtraTreesRegressor,
            "params": {
                "n_estimators": 200, "max_depth": 15, "min_samples_split": 10,
                "min_samples_leaf": 5, "random_state": config.random_state, "n_jobs": -1
            }
        },
        {
            "name": "AdaBoost",
            "model": AdaBoostRegressor,
            "params": {
                "n_estimators": 100, "learning_rate": 0.1, "loss": "linear",
                "random_state": config.random_state
            }
        },
        {
            "name": "Bagging",
            "model": BaggingRegressor,
            "params": {
                "n_estimators": 50, "max_samples": 0.8, "max_features": 0.8,
                "random_state": config.random_state, "n_jobs": -1
            }
        },
        {
            "name": "DecisionTree",
            "model": DecisionTreeRegressor,
            "params": {
                "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 5,
                "random_state": config.random_state
            }
        },
        {
            "name": "SVR_RBF",
            "model": SVR,
            "params": {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"}
        },
        {
            "name": "SVR_Poly",
            "model": SVR,
            "params": {"kernel": "poly", "degree": 3, "C": 10.0, "epsilon": 0.1, "gamma": "scale"}
        },
        {
            "name": "NeuralNetwork",
            "model": MLPRegressor,
            "params": {
                "hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam",
                "alpha": 0.001, "batch_size": "auto", "learning_rate": "adaptive",
                "max_iter": 500, "random_state": config.random_state
            }
        },
        {
            "name": "GaussianProcess",
            "model": GaussianProcessRegressor,
            "params": {
                "alpha": 1e-6, "normalize_y": True, "n_restarts_optimizer": 5,
                "random_state": config.random_state
            }
        },
        {
            "name": "KernelRidge",
            "model": KernelRidge,
            "params": {"alpha": 1.0, "kernel": "rbf", "gamma": 0.1}
        }
    ]

    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        models.append({
            "name": "CatBoost",
            "model": CatBoostRegressor,
            "params": {
                "iterations": 300, "depth": 6, "learning_rate": 0.05,
                "loss_function": "RMSE", "eval_metric": "RMSE", "random_seed": config.random_state,
                "verbose": False, "allow_writing_files": False, "task_type": "CPU", "thread_count": 4
            }
        })

    return models


def log_to_mlflow(
        run,
        best_pipeline: Pipeline,
        best_model_name: str,
        feature_importance_json: Optional[str],
        feature_importance: Dict[str, float],
        results_sorted: List[Dict],
        best_rmse: float,
        X_train: pd.DataFrame,
        config: TrainingConfig,
        timestamp: str,
        feature_cols: List[str],
        log_params: bool = True
):
    """
    Log training results to MLflow.

    Args:
        run: MLflow run object
        best_pipeline: Best trained pipeline
        best_model_name: Name of best model
        feature_importance_json: JSON string of feature importance
        feature_importance: Feature importance dictionary
        results_sorted: Sorted results list
        best_rmse: Best RMSE value
        X_train: Training data
        config: Training configuration
        timestamp: Training timestamp
        feature_cols: Feature column names
        log_params: Whether to log parameters
    """
    # Tags
    mlflow.set_tag("model_type", "global")
    mlflow.set_tag("best_model", best_model_name)
    mlflow.set_tag("training_date", timestamp)
    if feature_importance_json:
        mlflow.set_tag("has_feature_importance", "true")

    # Parameters
    if log_params:
        mlflow.log_params({
            "features_count": len(feature_cols),
            "num_features": len(config.numeric_columns),
            "cat_features": len(config.categorical_columns),
            "target": config.target_column,
            "test_size": config.test_size,
            "random_state": config.random_state,
            "cv_folds": config.cv_folds
        })

    # Metrics
    mlflow.log_metrics({
        "rmse": best_rmse,
        "r2": results_sorted[0]['r2'],
        "mae": results_sorted[0]['mae'],
        "mape": results_sorted[0]['mape'] if not np.isnan(results_sorted[0]['mape']) else 0
    })

    # Log metrics for all models
    for res in results_sorted:
        model_name = res['model_name'].replace(' ', '_').replace('-', '_')
        mlflow.log_metrics({
            f"{model_name}_rmse": res['rmse'],
            f"{model_name}_r2": res['r2'],
            f"{model_name}_mae": res['mae'],
            f"{model_name}_mape": res['mape'] if not np.isnan(res['mape']) else 0
        })

    # Custom signature
    sample_input = X_train.head(5)
    signature = create_custom_signature(sample_input, best_pipeline, config)
    logger.info(f"Custom signature created with {len(signature.inputs.inputs)} inputs")

    # Model metadata
    model_metadata = {
        "best_model": best_model_name,
        "timestamp": timestamp,
        "num_cols": config.numeric_columns,
        "cat_cols": config.categorical_columns
    }
    if feature_importance:
        model_metadata["feature_importance"] = feature_importance

    # Log model
    model_info = mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="model",
        signature=signature,
        registered_model_name="PricingModelGlobal",
        metadata=model_metadata
    )

    version = getattr(model_info, 'registered_model_version', None)
    logger.info(f"Model registered: PricingModelGlobal v{version}")

    # Add tags to version
    if version:
        try:
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag("PricingModelGlobal", version, "model_type", "global")
            client.set_model_version_tag("PricingModelGlobal", version, "best_model", best_model_name)
            client.set_model_version_tag("PricingModelGlobal", version, "rmse", str(best_rmse))
            client.set_model_version_tag("PricingModelGlobal", version, "r2", str(results_sorted[0]['r2']))

            if feature_importance_json:
                client.set_model_version_tag("PricingModelGlobal", version, "feature_importance",
                                             feature_importance_json)

            # Assign production alias
            try:
                client.set_registered_model_alias("PricingModelGlobal", "production", str(version))
                logger.info(f"Alias 'production' assigned to version {version}")
            except Exception as e:
                logger.warning(f"Unable to assign alias: {e}")

        except Exception as e:
            logger.warning(f"Unable to add tags: {e}")


def train_and_compare(
        file_path: Optional[Path] = None,
        register_to_mlflow: bool = True,
        mlflow_run: Optional[Any] = None
) -> Tuple[str, List[Dict], Pipeline, pd.DataFrame, pd.Series, Pipeline, Dict[str, float], Optional[str]]:
    """
    Main training pipeline.

    Args:
        file_path: Path to input file
        register_to_mlflow: Whether to register to MLflow
        mlflow_run: Existing MLflow run (for nested runs)

    Returns:
        Tuple of (best_model_name, results, best_pipeline, X_test, y_test,
                  best_pipeline, feature_importance, feature_importance_json)
    """
    config = TrainingConfig()

    # Setup MLflow
    if register_to_mlflow and mlflow_run is None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment("Pricing_Global_Model")

    # Load data
    if file_path is None:
        file_path = DATA_DIR / "pricing_fully_cleaned.xlsx"
    else:
        file_path = Path(file_path)

    df = load_and_validate_data(file_path, config)

    # Clean data
    df = clean_data(df, config)

    # Prepare features and target
    y_original = df[config.target_column].copy()
    y_log = np.log1p(y_original)

    feature_cols = config.numeric_columns + config.categorical_columns
    X = df[feature_cols].copy()

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Target: min={y_original.min():.2f}, max={y_original.max():.2f}, mean={y_original.mean():.2f}")

    # Create preprocessor
    preprocessor = create_preprocessor(config)

    # Split data (with stratification by target quantiles)
    try:
        # Stratified split based on target quantiles
        y_binned = pd.qcut(y_log, q=10, duplicates='drop')
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=config.test_size, random_state=config.random_state,
            stratify=y_binned
        )
        logger.info("Stratified train/test split applied")
    except:
        # Fallback to simple split
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=config.test_size, random_state=config.random_state
        )
        logger.info("Simple train/test split applied")

    y_train_original = np.expm1(y_train_log)
    y_test_original = np.expm1(y_test_log)

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Train models
    model_configs = get_model_configs(config)
    best_rmse = float("inf")
    best_pipeline = None
    best_model_name = None
    results = []

    for model_config in model_configs:
        name = model_config["name"]
        ModelClass = model_config["model"]
        params = model_config["params"]

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training → {name}")
        logger.info(f"{'=' * 50}")

        try:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", ModelClass(**params))
            ])

            # Cross-validation
            logger.info("  Running cross-validation...")
            cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train_log, cv=cv,
                scoring='neg_root_mean_squared_error', n_jobs=-1
            )
            cv_rmse = -cv_scores.mean()
            logger.info(f"  CV RMSE (log): {cv_rmse:.4f} (+/- {cv_scores.std():.4f})")

            # Train final model
            pipeline.fit(X_train, y_train_log)

            # Predict
            y_pred_log = pipeline.predict(X_test)
            y_pred_original = np.expm1(y_pred_log)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            r2 = r2_score(y_test_original, y_pred_original)
            mae = mean_absolute_error(y_test_original, y_pred_original)

            mask = y_test_original > 0
            mape = mean_absolute_percentage_error(
                y_test_original[mask], y_pred_original[mask]
            ) * 100 if mask.sum() > 0 else np.nan

            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  RMSE: {rmse:.4f} €")
            logger.info(f"  MAE: {mae:.4f} €")
            logger.info(f"  MAPE: {mape:.2f} %")

            results.append({
                "model_name": name, "r2": r2, "rmse": rmse, "mae": mae,
                "mape": mape, "cv_rmse": cv_rmse, "cv_rmse_std": cv_scores.std()
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_pipeline = pipeline
                best_model_name = name
                logger.info(f"  🏆 NEW BEST MODEL!")

        except Exception as e:
            logger.error(f"Error with {name}: {e}")
            continue

    if best_pipeline is None:
        raise ValueError("No model could be trained successfully!")

    # Results ranking
    results_sorted = sorted(results, key=lambda x: x['rmse'])
    logger.info(f"\n{'=' * 70}")
    logger.info(f"🏆 BEST MODEL: {best_model_name}")
    logger.info(f"   Final RMSE = {best_rmse:.4f} €")
    logger.info(f"{'=' * 70}")

    logger.info("\n📋 RANKING:")
    for i, res in enumerate(results_sorted, 1):
        logger.info(f"{i}. {res['model_name']}: RMSE={res['rmse']:.4f} €, R²={res['r2']:.4f}")

    # Feature importance
    feature_importance, feature_importance_json = extract_feature_importance(
        best_pipeline, feature_cols, config
    )

    # Local save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = MODELS_DIR / f"best_{best_model_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, save_dir / "pipeline_complete.joblib")
    pd.DataFrame(results).to_csv(save_dir / "comparison_results.csv", index=False)

    if feature_importance:
        with open(save_dir / "feature_importance.json", "w") as f:
            json.dump(feature_importance, f, indent=2)

    metadata = {
        "best_model": best_model_name, "best_rmse": best_rmse, "timestamp": timestamp,
        "n_features": len(feature_cols), "n_samples": len(X),
        "num_cols": config.numeric_columns, "cat_cols": config.categorical_columns,
        "has_feature_importance": bool(feature_importance)
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved locally: {save_dir}")

    # MLflow registration
    if register_to_mlflow:
        logger.info("\n📦 MLflow Registration")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mlflow_run_name = f"global-{timestamp}"

        if mlflow_run is not None:
            log_to_mlflow(
                mlflow_run, best_pipeline, best_model_name, feature_importance_json,
                feature_importance, results_sorted, best_rmse, X_train, config,
                timestamp, feature_cols, log_params=False
            )
        else:
            with safe_mlflow_start_run(run_name=mlflow_run_name) as run:
                log_to_mlflow(
                    run, best_pipeline, best_model_name, feature_importance_json,
                    feature_importance, results_sorted, best_rmse, X_train, config,
                    timestamp, feature_cols, log_params=True
                )

    return (best_model_name, results, best_pipeline, X_test, y_test_original,
            best_pipeline, feature_importance, feature_importance_json)


def generate_prediction_examples(
        best_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test_original: pd.Series,
        n_examples: int = 10
) -> pd.DataFrame:
    """
    Generate random prediction examples.

    Args:
        best_pipeline: Trained pipeline
        X_test: Test features
        y_test_original: True target values
        n_examples: Number of examples to generate

    Returns:
        DataFrame with prediction examples
    """
    random.seed(42)
    n_samples = len(X_test)
    random_indices = random.sample(range(n_samples), min(n_examples, n_samples))

    examples = []
    for idx in random_indices:
        X_sample = X_test.iloc[[idx]]
        y_true = y_test_original.iloc[idx]

        y_pred_log = best_pipeline.predict(X_sample)
        y_pred = np.expm1(y_pred_log)[0]

        error = y_true - y_pred
        error_percent = (error / y_true) * 100 if y_true > 0 else 0

        examples.append({
            "index": idx, "actual_value": y_true, "predicted_value": y_pred,
            "abs_error": abs(error), "pct_error": abs(error_percent)
        })

    return pd.DataFrame(examples)


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🚀 STARTING MODEL TRAINING")
    logger.info("=" * 70)

    try:
        # Run training
        best_name, all_results, best_pipeline, X_test, y_test_original, model, feature_importance, feature_importance_json = train_and_compare(
            register_to_mlflow=False
        )

        logger.info("\n✨ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"🏁 Best model: {best_name}")
        if feature_importance:
            logger.info(f"📊 Feature importance extracted: {len(feature_importance)} features")

        # Generate prediction examples
        examples_df = generate_prediction_examples(best_pipeline, X_test, y_test_original)

        logger.info("\n" + "=" * 70)
        logger.info("📊 PREDICTION EXAMPLES (10 RANDOM SAMPLES)")
        logger.info("=" * 70)
        logger.info("\n" + examples_df.to_string(index=False, float_format="%.2f"))

        # Statistics
        logger.info("\n📈 STATISTICS:")
        logger.info(f"   Average absolute error: {examples_df['abs_error'].mean():.2f} €")
        logger.info(f"   Average relative error: {examples_df['pct_error'].mean():.2f} %")
        logger.info(f"   Min error: {examples_df['abs_error'].min():.2f} €")
        logger.info(f"   Max error: {examples_df['abs_error'].max():.2f} €")

        # Save examples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        examples_path = MODELS_DIR / f"prediction_examples_{timestamp}.csv"
        examples_df.to_csv(examples_path, index=False)
        logger.info(f"\n💾 Examples saved to: {examples_path}")

        # Top predictions
        logger.info("\n🏆 TOP 5 BEST PREDICTIONS:")
        top5 = examples_df.nsmallest(5, 'abs_error')
        logger.info(top5[['actual_value', 'predicted_value', 'abs_error', 'pct_error']].to_string(index=False,
                                                                                                  float_format="%.2f"))

        logger.info("\n⚠️ TOP 5 WORST PREDICTIONS:")
        bottom5 = examples_df.nlargest(5, 'abs_error')
        logger.info(bottom5[['actual_value', 'predicted_value', 'abs_error', 'pct_error']].to_string(index=False,
                                                                                                     float_format="%.2f"))

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)