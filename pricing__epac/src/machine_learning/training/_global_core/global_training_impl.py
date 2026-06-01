#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global training orchestrator.

This module intentionally stays short and delegates responsibilities to:
- data_io.py
- feature_prep.py
- model_registry.py
- evaluation.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from pricing__epac.src.machine_learning.training._global_core.data_io import (
    DATA_DIR,
    MODELS_DIR,
    TrainingConfig,
    clean_data,
    load_and_validate_data,
    logger,
    save_training_outputs,
)
from pricing__epac.src.machine_learning.training._global_core.evaluation import (
    extract_feature_importance,
    generate_prediction_examples,
    log_to_mlflow,
    safe_mlflow_start_run,
)
from pricing__epac.src.machine_learning.training._global_core.feature_prep import create_preprocessor
from pricing__epac.src.machine_learning.training._global_core.model_registry import get_model_configs


def train_and_compare(
    file_path: Optional[Path] = None,
    register_to_mlflow: bool = True,
    mlflow_run: Optional[Any] = None,
) -> Tuple[str, List[Dict], Pipeline, pd.DataFrame, pd.Series, Pipeline, Dict[str, float], Optional[str]]:
    """Train all candidate models, pick best, persist artifacts, optionally log to MLflow."""
    config = TrainingConfig()

    if register_to_mlflow and mlflow_run is None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment("Pricing_Global_Model")

    source_file = Path(file_path) if file_path is not None else DATA_DIR / "pricing_fully_cleaned.xlsx"
    df = clean_data(load_and_validate_data(source_file, config), config)

    y_original = df[config.target_column].copy()
    y_log = np.log1p(y_original)
    feature_cols = config.numeric_columns + config.categorical_columns
    X = df[feature_cols].copy()

    preprocessor = create_preprocessor(config)

    try:
        y_binned = pd.qcut(y_log, q=10, duplicates="drop")
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X,
            y_log,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y_binned,
        )
        logger.info("Stratified train/test split applied")
    except Exception:
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X,
            y_log,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        logger.info("Simple train/test split applied")

    y_test_original = np.expm1(y_test_log)

    best_rmse = float("inf")
    best_pipeline = None
    best_model_name = None
    results = []

    for model_config in get_model_configs(config):
        name = model_config["name"]
        model_class = model_config["model"]
        params = model_config["params"]

        try:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_class(**params)),
            ])

            cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train_log,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse = -cv_scores.mean()

            pipeline.fit(X_train, y_train_log)
            y_pred_original = np.expm1(pipeline.predict(X_test))

            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            r2 = r2_score(y_test_original, y_pred_original)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            mask = y_test_original > 0
            mape = (
                mean_absolute_percentage_error(y_test_original[mask], y_pred_original[mask]) * 100
                if mask.sum() > 0
                else np.nan
            )

            results.append({
                "model_name": name,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "cv_rmse": cv_rmse,
                "cv_rmse_std": cv_scores.std(),
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_pipeline = pipeline
                best_model_name = name

        except Exception as exc:
            logger.error("Error with %s: %s", name, exc)
            continue

    if best_pipeline is None or best_model_name is None:
        raise ValueError("No model could be trained successfully")

    results_sorted = sorted(results, key=lambda x: x["rmse"])
    feature_importance, feature_importance_json = extract_feature_importance(best_pipeline, feature_cols, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = save_training_outputs(
        best_pipeline=best_pipeline,
        results=results,
        feature_importance=feature_importance,
        best_model_name=best_model_name,
        best_rmse=best_rmse,
        feature_cols=feature_cols,
        X=X,
        config=config,
        timestamp=timestamp,
    )
    logger.info("Model saved locally: %s", save_dir)

    if register_to_mlflow:
        run_name = f"global-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if mlflow_run is not None:
            log_to_mlflow(
                logger,
                mlflow_run,
                best_pipeline,
                best_model_name,
                feature_importance_json,
                feature_importance,
                results_sorted,
                best_rmse,
                X_train,
                config,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
                feature_cols,
                log_params=False,
            )
        else:
            with safe_mlflow_start_run(run_name=run_name) as run:
                log_to_mlflow(
                    logger,
                    run,
                    best_pipeline,
                    best_model_name,
                    feature_importance_json,
                    feature_importance,
                    results_sorted,
                    best_rmse,
                    X_train,
                    config,
                    datetime.now().strftime("%Y%m%d_%H%M%S"),
                    feature_cols,
                    log_params=True,
                )

    return (
        best_model_name,
        results,
        best_pipeline,
        X_test,
        y_test_original,
        best_pipeline,
        feature_importance,
        feature_importance_json,
    )


__all__ = [
    "TrainingConfig",
    "train_and_compare",
    "generate_prediction_examples",
    "MODELS_DIR",
    "DATA_DIR",
]


if __name__ == "__main__":
    logger.info("Starting model training")
    best_name, _, best_pipeline, X_test, y_test_original, *_ = train_and_compare(register_to_mlflow=False)
    examples_df = generate_prediction_examples(best_pipeline, X_test, y_test_original)
    logger.info("Training done. Best model: %s", best_name)
    logger.info("Sample predictions:\n%s", examples_df.to_string(index=False, float_format="%.2f"))
    examples_path = MODELS_DIR / f"prediction_examples_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    examples_df.to_csv(examples_path, index=False)
    logger.info("Prediction examples saved to: %s", examples_path)

