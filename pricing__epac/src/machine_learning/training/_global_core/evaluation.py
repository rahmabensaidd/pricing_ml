"""Evaluation, MLflow logging, and prediction examples."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, Schema


class MLflowRunManager:
    @staticmethod
    def start_run(run_name: str):
        active = mlflow.active_run()
        if active is not None:
            return mlflow.start_run(run_name=run_name, nested=True)
        return mlflow.start_run(run_name=run_name)


@contextmanager
def safe_mlflow_start_run(run_name: str):
    run = MLflowRunManager.start_run(run_name)
    try:
        yield run
    finally:
        mlflow.end_run()


def create_custom_signature(sample_input, best_pipeline, config) -> ModelSignature:
    output = best_pipeline.predict(sample_input.head(1))
    inputs = []
    for col in config.numeric_columns + config.boolean_columns:
        inputs.append(ColSpec(DataType.double, col, required=False))
    for col in config.categorical_columns:
        inputs.append(ColSpec(DataType.string, col, required=False))
    outputs = [ColSpec(DataType.double, "predicted_price", required=True)]
    return ModelSignature(inputs=Schema(inputs), outputs=Schema(outputs))


def extract_feature_importance(best_pipeline, feature_cols: List[str], config) -> Tuple[Dict[str, float], Optional[str]]:
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = feature_cols

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        return {}, None

    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[: config.max_features_for_importance]
    fi = {name: float(score) for name, score in ranked}
    return fi, pd.Series(fi).to_json()


def log_to_mlflow(
    logger,
    run,
    best_pipeline,
    best_model_name: str,
    feature_importance_json: Optional[str],
    feature_importance: Dict[str, float],
    results_sorted: List[Dict],
    best_rmse: float,
    X_train: pd.DataFrame,
    config,
    timestamp: str,
    feature_cols: List[str],
    log_params: bool = True,
):
    mlflow.set_tag("model_type", "global")
    mlflow.set_tag("best_model", best_model_name)
    mlflow.set_tag("training_date", timestamp)
    if feature_importance_json:
        mlflow.set_tag("has_feature_importance", "true")

    if log_params:
        mlflow.log_params({
            "features_count": len(feature_cols),
            "num_features": len(config.numeric_columns),
            "cat_features": len(config.categorical_columns),
            "target": config.target_column,
            "test_size": config.test_size,
            "random_state": config.random_state,
            "cv_folds": config.cv_folds,
        })

    mlflow.log_metrics({
        "rmse": best_rmse,
        "r2": results_sorted[0]["r2"],
        "mae": results_sorted[0]["mae"],
        "mape": results_sorted[0]["mape"] if not np.isnan(results_sorted[0]["mape"]) else 0,
    })

    for res in results_sorted:
        model_name = res["model_name"].replace(" ", "_").replace("-", "_")
        mlflow.log_metrics({
            f"{model_name}_rmse": res["rmse"],
            f"{model_name}_r2": res["r2"],
            f"{model_name}_mae": res["mae"],
            f"{model_name}_mape": res["mape"] if not np.isnan(res["mape"]) else 0,
        })

    signature = create_custom_signature(X_train.head(5), best_pipeline, config)
    model_metadata = {
        "best_model": best_model_name,
        "timestamp": timestamp,
        "num_cols": config.numeric_columns,
        "cat_cols": config.categorical_columns,
    }
    if feature_importance:
        model_metadata["feature_importance"] = feature_importance

    model_info = mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        name="model",
        signature=signature,
        registered_model_name="PricingModelGlobal",
        metadata=model_metadata,
    )
    version = getattr(model_info, "registered_model_version", None)
    logger.info("Model registered: PricingModelGlobal v%s", version)

    if version:
        try:
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag("PricingModelGlobal", version, "model_type", "global")
            client.set_model_version_tag("PricingModelGlobal", version, "best_model", best_model_name)
            client.set_model_version_tag("PricingModelGlobal", version, "rmse", str(best_rmse))
            client.set_model_version_tag("PricingModelGlobal", version, "r2", str(results_sorted[0]["r2"]))
            if feature_importance_json:
                client.set_model_version_tag("PricingModelGlobal", version, "feature_importance", feature_importance_json)
            client.set_registered_model_alias("PricingModelGlobal", "production", str(version))
        except Exception as exc:
            logger.warning("Model version tagging/aliasing failed: %s", exc)


def generate_prediction_examples(best_pipeline, X_test: pd.DataFrame, y_test_original: pd.Series, n_examples: int = 10) -> pd.DataFrame:
    random.seed(42)
    random_indices = random.sample(range(len(X_test)), min(n_examples, len(X_test)))
    examples = []
    for idx in random_indices:
        X_sample = X_test.iloc[[idx]]
        y_true = y_test_original.iloc[idx]
        y_pred = np.expm1(best_pipeline.predict(X_sample))[0]
        error = y_true - y_pred
        examples.append({
            "index": idx,
            "actual_value": y_true,
            "predicted_value": y_pred,
            "abs_error": abs(error),
            "pct_error": abs((error / y_true) * 100) if y_true > 0 else 0,
        })
    return pd.DataFrame(examples)
