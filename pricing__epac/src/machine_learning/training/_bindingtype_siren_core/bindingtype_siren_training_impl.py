# ============================================================
# EPAC – TRAINING BY COUPLE (BINDING_TYPE + SIREN)
#     - Version with ultra-strong regularization
#     - LOG TRANSFORMATION of prices
#     - OUTLIER CLIPPING
#     - ROBUST SCALER instead of StandardScaler
#     - Linear models with EXTREME REGULARIZATION
#     - Non-linear models with ULTRA-STRONG CONSTRAINTS
#     - Cross-validation with stability check
#     - REINFORCED prediction plausibility tests
#     - MEDIAN FALLBACK if all models fail
#     - MLFLOW REGISTRY REGISTRATION WITH VERSIONNING
#     - Formula extraction for linear models
#     - Feature importance and SHAP for non-linear models
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import warnings
import os
# Add root path to find openssl_patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from pricing__epac import openssl_patch
from pricing__epac.src.config.settings import settings
import time
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Linear models with CV and REINFORCED REGULARIZATION
from sklearn.linear_model import (
    RidgeCV,
    LassoCV,
    ElasticNetCV,
    Ridge
)

# Non-linear models with STRONG CONSTRAINTS
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
import xgboost as xgb
import lightgbm as lgb

# Constant model for fallback
from sklearn.dummy import DummyRegressor
# After imports, before the code starts
import os

# Configuration for MinIO (S3 compatible)
os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MLFLOW_S3_ENDPOINT_URL
os.environ['AWS_DEFAULT_REGION'] = settings.AWS_DEFAULT_REGION
# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
    print("[INFO] SHAP successfully installed")
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not installed. Install it with: pip install shap")

import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

# MLflow configuration
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ────────────────────────────────────────────────
# INITIAL FEATURE CONFIGURATION
# ────────────────────────────────────────────────
PROJECT_ROOT = settings.PROJECT_ROOT
MODELS_DIR = settings.MODELS_ARTIFACT_ROOT / "bindingtype_siren_regularized"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# NUM_VARS now includes boolean variables (0/1)
NUM_VARS = [
    "quantity", "production_page", "height", "thickness", "width",
    "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
    "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill"
]

TARGET = "unit_price"

CAT_VARS = [
    "text_paper_type", "text_color",
    "cover_finish_type", "cover_color", "cover_size", "cover_paper_type",
    "head_and_tail", "priority_level",
    "binding_type", "coil_type", "tab_color", "insert_paper_type",
    "case_finish_type", "spine_type", "label_type", "siren"
]

# Boolean columns to treat as integers
BOOL_COLS_AS_INT = [
    "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
    "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill"
]

# Maximum reasonable price for printing (beyond this, it's aberrant)
MAX_REASONABLE_PRICE = 5000

# Parameters
MIN_SAMPLES = 30
CV_FOLDS = 5


# ────────────────────────────────────────────────
# DYNAMIC ANALYSIS OF PRICE DISTRIBUTIONS
# ────────────────────────────────────────────────

def compute_dynamic_price_bounds(df, group_column=None, group_value=None, outlier_multiplier=3):
    """
    Dynamically calculates price bounds for a specific group or globally
    with outlier detection and exclusion
    """
    print(f"\n   📊 Calculating dynamic bounds for {group_value if group_value else 'global'}...")

    if group_column and group_value and group_column in df.columns:
        group_data = df[df[group_column] == group_value][TARGET].dropna()
        if len(group_data) < 10:
            print(f"      ⚠️ Not enough data ({len(group_data)}), using global")
            group_data = df[TARGET].dropna()
    else:
        group_data = df[TARGET].dropna()

    if len(group_data) == 0:
        return {
            "min": 0.5,
            "max": MAX_REASONABLE_PRICE,
            "default": 100.0,
            "stats": {"n_samples": 0}
        }

    # Outlier detection
    q1 = group_data.quantile(0.25)
    q3 = group_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - outlier_multiplier * iqr
    upper_bound = q3 + outlier_multiplier * iqr

    # Filter outliers
    clean_data = group_data[(group_data >= lower_bound) & (group_data <= upper_bound)]

    if len(clean_data) < 5:
        clean_data = group_data

    # Robust statistics
    p1 = clean_data.quantile(0.01)
    p5 = clean_data.quantile(0.05)
    p95 = clean_data.quantile(0.95)
    p99 = clean_data.quantile(0.99)
    median = clean_data.median()

    # Bounds with safety margins
    min_bound = max(0.5, p1 * 0.8)
    max_bound = min(MAX_REASONABLE_PRICE, p99 * 1.5)

    stats = {
        "n_samples": len(group_data),
        "n_clean": len(clean_data),
        "outliers_removed": len(group_data) - len(clean_data),
        "outlier_pct": (len(group_data) - len(clean_data)) / len(group_data) * 100 if len(group_data) > 0 else 0,
        "min": float(group_data.min()),
        "max": float(group_data.max()),
        "median": float(median),
        "mean": float(clean_data.mean()),
        "p1": float(p1),
        "p5": float(p5),
        "p95": float(p95),
        "p99": float(p99)
    }

    print(f"      Calculated bounds: [{min_bound:.2f}€, {max_bound:.2f}€] (default: {median:.2f}€)")
    print(f"      Outliers: {stats['outliers_removed']} ({stats['outlier_pct']:.1f}%)")

    return {
        "min": float(min_bound),
        "max": float(max_bound),
        "default": float(median),
        "stats": stats
    }


# ────────────────────────────────────────────────
# MLFLOW UTILITY FUNCTIONS
# ────────────────────────────────────────────────

def safe_mlflow_start_run(run_name):
    """Safely starts an MLflow run"""
    try:
        mlflow.end_run()
        print("   ✅ Previous MLflow run terminated")
    except:
        pass
    return mlflow.start_run(run_name=run_name)


def create_custom_signature(input_df: pd.DataFrame, features_num: list, features_cat: list) -> Schema:
    """
    Creates a custom signature where booleans are integers (long)
    instead of booleans
    """
    schema_list = []

    # Continuous numeric columns (height, thickness, width) → double
    for col in ["height", "thickness", "width"]:
        if col in input_df.columns and col in features_num:
            schema_list.append(ColSpec("double", col, required=True))

    # Integer columns (quantity, production_page) → long
    for col in ["quantity", "production_page"]:
        if col in input_df.columns and col in features_num:
            schema_list.append(ColSpec("long", col, required=True))

    # Booleans as integers → long (NOT boolean)
    for col in BOOL_COLS_AS_INT:
        if col in input_df.columns and col in features_num:
            schema_list.append(ColSpec("long", col, required=True))

    # Categorical columns → string
    for col in features_cat:
        if col in input_df.columns:
            schema_list.append(ColSpec("string", col, required=True))

    return Schema(schema_list)


def set_model_version_tags(model_name: str, version: int, tags: dict) -> bool:
    """Adds tags to a specific model version"""
    try:
        client = mlflow.tracking.MlflowClient()
        for key, value in tags.items():
            client.set_model_version_tag(model_name, str(version), key, str(value))
        print(f"   → Tags added to {model_name} v{version}")
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to add tags: {e}")
        return False


def set_production_alias(model_name: str, version: int) -> bool:
    """Assigns production alias to a version"""
    try:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, "production", str(version))
        print(f"   ✅ Alias 'production' assigned to {model_name} v{version}")
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to assign alias: {e}")
        return False


def extract_feature_importance(pipeline, feature_names):
    """Extracts feature importance from the best model"""
    try:
        model = pipeline.named_steps["model"] if "model" in pipeline.named_steps else pipeline.named_steps.get("reg")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_dict = {}
            for i, imp in enumerate(importances):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(imp)
                else:
                    importance_dict[f"feature_{i}"] = float(imp)

            # Sort and limit to 50
            sorted_imp = dict(sorted(importance_dict.items(),
                                     key=lambda x: x[1],
                                     reverse=True)[:50])

            print(f"\n📊 Top 5 important features:")
            for i, (feat, imp) in enumerate(list(sorted_imp.items())[:5]):
                print(f"   {i + 1}. {feat}: {imp:.4f}")

            return sorted_imp, json.dumps(sorted_imp)

        elif hasattr(model, "coef_"):
            coefs = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            abs_coefs = np.abs(coefs)
            coef_dict = {}
            for i, coef in enumerate(abs_coefs):
                if i < len(feature_names):
                    coef_dict[feature_names[i]] = float(coef)
                else:
                    coef_dict[f"feature_{i}"] = float(coef)

            sorted_coef = dict(sorted(coef_dict.items(),
                                      key=lambda x: x[1],
                                      reverse=True)[:50])
            return sorted_coef, json.dumps(sorted_coef)

        else:
            return {}, None

    except Exception as e:
        print(f"⚠️ Error extracting feature importance: {e}")
        return {}, None


# ────────────────────────────────────────────────
# REINFORCED REGULARIZATION FUNCTIONS
# ────────────────────────────────────────────────

def validate_predictions_plausible_reinforced(model, X, group_name=None, price_bounds=None):
    """
    Reinforced version of prediction validation with dynamic bounds
    """
    try:
        if price_bounds is None:
            max_price = MAX_REASONABLE_PRICE
            min_price = 0.5
            safe_min = 0.5
            safe_max = MAX_REASONABLE_PRICE * 1.2
        else:
            max_price = price_bounds["max"]
            min_price = price_bounds["min"]
            safe_min = min_price * 0.8
            safe_max = min(max_price * 1.2, MAX_REASONABLE_PRICE * 1.5)

        # Sample to test
        X_sample = X.sample(min(100, len(X)), random_state=42)
        y_pred = model.predict(X_sample)

        # Prediction statistics
        pred_max = y_pred.max()
        pred_min = y_pred.min()
        pred_mean = y_pred.mean()
        pred_median = np.median(y_pred)
        pred_std = y_pred.std()

        print(f"\n   🔍 Reinforced validation for {group_name if group_name else 'model'}:")
        print(f"      Min: {pred_min:.2f}€")
        print(f"      Max: {pred_max:.2f}€")
        print(f"      Mean: {pred_mean:.2f}€")
        print(f"      Median: {pred_median:.2f}€")
        print(f"      Std: {pred_std:.2f}€")
        print(f"      Dynamic bounds: [{min_price:.2f}€, {max_price:.2f}€]")
        print(f"      Safety bounds: [{safe_min:.2f}€, {safe_max:.2f}€]")

        # Reinforced plausibility tests
        issues = []

        if pred_max > safe_max:
            issues.append(f"max too high: {pred_max:.2f}€ > {safe_max:.2f}€")
        elif pred_max > max_price:
            issues.append(f"max high: {pred_max:.2f}€ > {max_price:.2f}€")

        if pred_min < safe_min:
            issues.append(f"min too low: {pred_min:.2f}€ < {safe_min:.2f}€")
        elif pred_min < min_price:
            issues.append(f"min low: {pred_min:.2f}€ < {min_price:.2f}€")

        if pred_mean > max_price * 1.5:
            issues.append(f"mean too high: {pred_mean:.2f}€")
        elif pred_mean < min_price * 0.5:
            issues.append(f"mean too low: {pred_mean:.2f}€")

        expected_range = max_price - min_price
        if pred_std > expected_range * 2:
            issues.append(f"std too high: {pred_std:.2f}€ (expected range: {expected_range:.2f}€)")

        if pred_min > 0 and pred_max / pred_min > 100:
            issues.append(f"max/min ratio too high: {pred_max / pred_min:.1f}")

        percentiles = np.percentile(y_pred, [1, 5, 95, 99])
        if percentiles[3] > max_price * 3:
            issues.append(f"distribution tail too long (p99={percentiles[3]:.2f}€)")
        if percentiles[0] < 0:
            issues.append(f"negative predictions detected (p1={percentiles[0]:.2f}€)")

        if issues:
            print(f"   ⚠️ Problems detected ({len(issues)}):")
            for issue in issues[:3]:
                print(f"      - {issue}")
            if len(issues) > 3:
                print(f"      - and {len(issues) - 3} other(s)")
            return False

        print(f"   ✅ Predictions plausible")
        return True

    except Exception as e:
        print(f"   ⚠️ Error during validation: {e}")
        return False


def get_regularized_models_config_reinforced():
    """
    Configuration of models with ULTRA-STRONG REGULARIZATION
    to avoid aberrant predictions
    """
    return [
        {
            "name": "RandomForest_UltraRegularized",
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 30,
                "min_samples_leaf": 20,
                "max_features": 0.2,
                "max_leaf_nodes": 50,
                "min_impurity_decrease": 0.05,
                "ccp_alpha": 0.05,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        {
            "name": "XGBoost_UltraRegularized",
            "model": xgb.XGBRegressor,
            "params": {
                "n_estimators": 80,
                "max_depth": 2,
                "learning_rate": 0.001,
                "subsample": 0.3,
                "colsample_bytree": 0.2,
                "colsample_bylevel": 0.2,
                "colsample_bynode": 0.2,
                "reg_alpha": 5.0,
                "reg_lambda": 10.0,
                "min_child_weight": 20,
                "gamma": 0.5,
                "max_delta_step": 1,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0
            }
        },
        {
            "name": "LightGBM_UltraRegularized",
            "model": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 80,
                "max_depth": 2,
                "num_leaves": 20,
                "learning_rate": 0.001,
                "subsample": 0.3,
                "subsample_freq": 1,
                "colsample_bytree": 0.2,
                "reg_alpha": 5.0,
                "reg_lambda": 10.0,
                "min_child_samples": 50,
                "min_child_weight": 20,
                "min_split_gain": 0.5,
                "lambda_l1": 5.0,
                "lambda_l2": 10.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }
        },
        {
            "name": "GradientBoosting_UltraRegularized",
            "model": GradientBoostingRegressor,
            "params": {
                "n_estimators": 80,
                "max_depth": 2,
                "learning_rate": 0.001,
                "subsample": 0.4,
                "min_samples_split": 30,
                "min_samples_leaf": 20,
                "max_features": 0.2,
                "max_leaf_nodes": 50,
                "min_impurity_decrease": 0.05,
                "ccp_alpha": 0.05,
                "random_state": 42
            }
        },
        {
            "name": "ExtraTrees_UltraRegularized",
            "model": ExtraTreesRegressor,
            "params": {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 30,
                "min_samples_leaf": 20,
                "max_features": 0.2,
                "max_leaf_nodes": 50,
                "min_impurity_decrease": 0.05,
                "ccp_alpha": 0.05,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        {
            "name": "Ridge_Extreme",
            "model": Ridge,
            "params": {
                "alpha": 100.0,
                "random_state": 42
            }
        }
    ]


def calculate_stability_metrics(model, X, y, cv_folds=5):
    """
    Calculates stability metrics to detect overfitting
    """
    try:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

        stability = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max(),
            'cv_range': cv_scores.max() - cv_scores.min()
        }

        if stability['cv_std'] > 0.2:
            print(f"   ⚠️  Unstable model: CV std = {stability['cv_std']:.3f}")
            stability['overfitting_risk'] = 'HIGH'
        elif stability['cv_std'] > 0.1:
            stability['overfitting_risk'] = 'MEDIUM'
        else:
            stability['overfitting_risk'] = 'LOW'

        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)

            stability['train_score'] = train_score
            stability['val_score'] = val_score
            stability['train_val_gap'] = train_score - val_score

            if stability['train_val_gap'] > 0.2:
                print(f"   ⚠️  Overfitting detected: train={train_score:.3f}, val={val_score:.3f}")
                stability['overfitting_detected'] = True
            else:
                stability['overfitting_detected'] = False

        except:
            pass

        return stability

    except Exception as e:
        print(f"   Error calculating stability: {e}")
        return None


def prepare_features(df_group, min_samples_feature=8):
    """Prepares features for a given group"""
    features_num = []
    for c in NUM_VARS:
        if c in df_group.columns:
            non_na = df_group[c].notna().sum()
            std_val = df_group[c].std() if non_na > 1 else 0
            if non_na >= min_samples_feature and std_val > 1e-6:
                features_num.append(c)

    features_cat = []
    for c in CAT_VARS:
        if c in df_group.columns:
            non_na = df_group[c].notna().sum()
            if non_na >= min_samples_feature:
                if df_group[c].dtype == "object" or pd.api.types.is_categorical_dtype(df_group[c]):
                    nunique = df_group[c].nunique()
                    if 1 < nunique <= 50:
                        features_cat.append(c)

    return features_num, features_cat


# ────────────────────────────────────────────────
# LINEAR MODELS WITH ULTRA-STRONG REGULARIZATION
# ────────────────────────────────────────────────

def extract_linear_formula(pipeline, features_num, features_cat):
    """Extracts formula from a linear model"""
    try:
        if pipeline is None:
            return None

        reg = pipeline.named_steps.get("reg") or pipeline.named_steps.get("model")
        if not hasattr(reg, "coef_"):
            return None

        prep = pipeline.named_steps.get("prep")
        if not prep or "cat" not in prep.named_transformers_:
            return None

        intercept = reg.intercept_
        if hasattr(intercept, '__len__') and len(intercept) > 0:
            intercept = intercept[0]

        coefs = reg.coef_.flatten() if len(reg.coef_.shape) > 1 else reg.coef_

        # Get feature names after encoding
        cat_encoder = prep.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(features_cat)
        var_names = np.concatenate([features_num, cat_names])

        # Adjust length
        min_len = min(len(coefs), len(var_names))
        coefs = coefs[:min_len]
        var_names = var_names[:min_len]

        # Build formula
        terms = [f"{intercept:.4f}"]
        significant_terms = 0

        for name, coef in zip(var_names, coefs):
            if abs(coef) > 0.0001:
                sign = " + " if coef >= 0 else " - "
                terms.append(f"{sign}{abs(coef):.4f} × {name}")
                significant_terms += 1
                if significant_terms >= 20:
                    terms.append(" + ...")
                    break

        formula = "unit_price = " + "".join(terms)
        formula = formula.replace("+ -", "- ").replace("  ", " ")

        print(f"\n📝 Extracted formula: {formula[:100]}...")
        return formula

    except Exception as e:
        print(f"⚠️ Error extracting formula: {e}")
        return None


def train_linear_models_regularized(df_group, group_name, min_samples=30, cv_folds=5, price_bounds=None):
    """Trains linear models with ULTRA-STRONG REGULARIZATION"""

    print(f"\n{'=' * 70}")
    print(f"[REGULARIZED LINEAR] {group_name}  |  {len(df_group)} rows")
    print(f"{'=' * 70}")

    if len(df_group) < min_samples:
        print(f"[WARNING] Not enough data ({len(df_group)} < {min_samples})")
        return None

    features_num, features_cat = prepare_features(df_group)
    all_features = features_num + features_cat

    if len(all_features) < 3:
        print("[WARNING] Not enough variables")
        return None

    print(f"Variables: {len(all_features)} including num={len(features_num)}, cat={len(features_cat)}")

    X = df_group[all_features].copy()
    y = df_group[TARGET].copy()

    common_idx = X.index.intersection(y.index)
    if len(common_idx) != len(X):
        X = X.loc[common_idx]
        y = y.loc[common_idx]

    # ========== LOG TRANSFORMATION OF PRICES ==========
    y_log = np.log1p(y)

    print(f"\n   📊 Price statistics:")
    print(f"      Raw prices: min={y.min():.2f}€, mean={y.mean():.2f}€, max={y.max():.2f}€")

    # ========== OUTLIER CLIPPING ==========
    if price_bounds:
        y_clipped = y.clip(lower=price_bounds["min"], upper=price_bounds["max"])
    else:
        p1 = y.quantile(0.01)
        p99 = y.quantile(0.99)
        y_clipped = y.clip(lower=max(0.5, p1 * 0.8), upper=min(MAX_REASONABLE_PRICE, p99 * 1.5))

    y_log_clipped = np.log1p(y_clipped)

    n_clipped = (y != y_clipped).sum()
    if n_clipped > 0:
        print(f"   ✂️ Clipping: {n_clipped} samples adjusted ({n_clipped / len(y) * 100:.1f}%)")

    # ========== PREPROCESSOR WITH ROBUST SCALER ==========
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(quantile_range=(5, 95)), features_num),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
             features_cat)
        ],
        remainder="drop"
    )

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # ========== REGULARIZED LINEAR MODELS ==========
    models = {
        "RidgeCV_Log": RidgeCV(
            alphas=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
            cv=kf,
            scoring="r2"
        ),
        "Ridge_Alpha100_Log": Ridge(alpha=100.0, random_state=42),
        "Ridge_Alpha500_Log": Ridge(alpha=500.0, random_state=42),
        "LassoCV_Log": LassoCV(
            alphas=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            cv=kf,
            max_iter=100000,
            random_state=42,
            tol=1e-4,
            selection='random'
        ),
        "ElasticNetCV_Log": ElasticNetCV(
            alphas=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=kf,
            max_iter=100000,
            tol=1e-4,
            random_state=42,
            n_jobs=-1
        )
    }

    best_cv_r2 = -np.inf
    best_name = None
    best_pipe = None
    best_params = {}
    all_results = []

    print(f"\nEvaluation {cv_folds}-fold CV on log(price)...")

    for name, model in models.items():
        try:
            pipe = Pipeline([("prep", preprocessor), ("reg", model)])

            cv_scores = cross_val_score(pipe, X, y_log_clipped, cv=kf, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            pipe.fit(X, y_log_clipped)

            y_pred_log = pipe.predict(X)
            y_pred = np.expm1(y_pred_log)

            if price_bounds:
                y_pred = np.clip(y_pred, price_bounds["min"] * 0.8, price_bounds["max"] * 1.2)

            n_aberrant = (y_pred > (price_bounds["max"] * 2 if price_bounds else MAX_REASONABLE_PRICE)).sum()

            print(f"{name:25} -> CV R2(log) = {cv_mean:.4f} (±{cv_std:.4f}) | aberrant={n_aberrant}")

            if hasattr(pipe.named_steps["reg"], "alpha_"):
                print(f"      alpha={pipe.named_steps['reg'].alpha_:.1f}")

            if n_aberrant == 0:
                all_results.append({
                    "model_name": name,
                    "cv_r2": cv_mean,
                    "cv_std": cv_std,
                })

                if cv_mean > best_cv_r2:
                    best_cv_r2 = cv_mean
                    best_name = name
                    best_pipe = pipe
                    best_params = {"alpha": getattr(pipe.named_steps["reg"], "alpha_", None)}

        except Exception as e:
            print(f"{name:25} -> ERROR: {str(e)[:50]}")
            continue

    # ========== FALLBACK IF NO VALID MODEL ==========
    if best_pipe is None:
        print("\n⚠️  No valid model, median fallback...")

        try:
            median_pipe = Pipeline([
                ("prep", preprocessor),
                ("reg", DummyRegressor(strategy='median'))
            ])
            median_pipe.fit(X, y_log_clipped)
            best_pipe = median_pipe
            best_name = "Median_Fallback"
            best_cv_r2 = 0
            print("   ✅ Median model accepted")
        except Exception as e:
            print(f"   ❌ Fallback failed: {e}")
            return None

    # Final predictions
    y_pred_log = best_pipe.predict(X)
    y_pred = np.expm1(y_pred_log)

    if price_bounds:
        y_pred = np.clip(y_pred, price_bounds["min"] * 0.8, price_bounds["max"] * 1.2)

    r2_train = r2_score(y, y_pred)
    mae_train = mean_absolute_error(y, y_pred)

    print(f"\n✅ Best model: {best_name}")
    print(f"   R2 = {r2_train:.4f}")
    print(f"   MAE = {mae_train:.2f}€")
    print(f"   Average prices: predicted={y_pred.mean():.2f}€, actual={y.mean():.2f}€")
    print(f"   Median prices: predicted={np.median(y_pred):.2f}€, actual={np.median(y):.2f}€")

    # Extract formula
    formula = None
    if best_name not in ["Median_Fallback", "Constant_Fallback"]:
        formula = extract_linear_formula(best_pipe, features_num, features_cat)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y, y_pred, alpha=0.6, s=45, edgecolor="none")
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1.3)
    axes[0].set_xlabel("Actual price (€)")
    axes[0].set_ylabel("Predicted price (€)")
    axes[0].set_title(f"{group_name} - {best_name}")
    axes[0].grid(True, alpha=0.25)

    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=45, edgecolor="none")
    axes[1].axhline(y=0, color='r', linestyle='--', lw=1.3)
    axes[1].set_xlabel("Predicted price (€)")
    axes[1].set_ylabel("Residuals (€)")
    axes[1].set_title(f"Residuals - MAE = {mae_train:.2f}€")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

    return {
        "group": group_name,
        "model_type": "linear",
        "best_model": best_name,
        "cv_r2": best_cv_r2,
        "r2_train": r2_train,
        "mae_train": mae_train,
        "n_samples": len(df_group),
        "n_features": len(all_features),
        "pipeline": best_pipe,
        "uses_log_transform": True,
        "formula": formula,
        "predicted_mean": float(y_pred.mean()),
        "actual_mean": float(y.mean()),
        "predicted_median": float(np.median(y_pred)),
        "actual_median": float(np.median(y)),
        "all_results": [
            {
                "model_name": r["model_name"],
                "cv_r2": float(r["cv_r2"]),
                "cv_std": float(r["cv_std"]),
            }
            for r in all_results
        ],
    }


# ────────────────────────────────────────────────
# REGULARIZED NON-LINEAR MODELS
# ────────────────────────────────────────────────

def train_nonlinear_family_regularized(df_group, group_name, min_samples=30, price_bounds=None):
    """Trains non-linear models WITH ULTRA-STRONG REGULARIZATION"""

    print(f"\n{'=' * 70}")
    print(f"[REGULARIZED NON-LINEAR] {group_name}  |  {len(df_group)} rows")
    print(f"{'=' * 70}")

    if len(df_group) < min_samples:
        print(f"[WARNING] Not enough data ({len(df_group)} < {min_samples})")
        return None

    features_num, features_cat = prepare_features(df_group)
    all_base_features = features_num + features_cat

    if len(all_base_features) < 3:
        print("[WARNING] Not enough variables")
        return None

    X = df_group[all_base_features].copy()
    y = df_group[TARGET].copy()

    common_idx = X.index.intersection(y.index)
    if len(common_idx) != len(X):
        X = X.loc[common_idx]
        y = y.loc[common_idx]

    # Statistics
    print(f"\n   📊 Prices: min={y.min():.2f}€, mean={y.mean():.2f}€, max={y.max():.2f}€")

    # ========== LOG TRANSFORMATION OF PRICES ==========
    y_log = np.log1p(y)

    # ========== OUTLIER CLIPPING ==========
    if price_bounds:
        y_clipped = y.clip(lower=price_bounds["min"], upper=price_bounds["max"])
    else:
        p1 = y.quantile(0.01)
        p99 = y.quantile(0.99)
        y_clipped = y.clip(lower=max(0.5, p1 * 0.5), upper=min(MAX_REASONABLE_PRICE, p99 * 1.5))

    y_log_clipped = np.log1p(y_clipped)

    print(f"   ✂️ Clipping: {(y != y_clipped).sum()} samples adjusted")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log_clipped, test_size=0.2, random_state=42
    )

    # ========== PREPROCESSOR WITH ROBUST SCALER ==========
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(quantile_range=(5, 95)), features_num),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
             features_cat)
        ],
        remainder="drop"
    )

    preprocessor.fit(X_train)

    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    try:
        feature_names = (
                features_num +
                list(preprocessor.named_transformers_["cat"].get_feature_names_out())
        )
    except:
        feature_names = [f"feature_{i}" for i in range(X_train_prep.shape[1])]

    expected_n_features = X_train_prep.shape[1]
    print(f"   🔢 Features after preprocessing: {expected_n_features}")

    # ========== REGULARIZED MODELS CONFIGURATION ==========
    models_config = get_regularized_models_config_reinforced()

    all_results = []
    best_r2 = -np.inf
    best_result = None
    best_model_obj = None
    best_model_name = None

    print(f"\nEvaluating {len(models_config)} models...")

    for config in models_config:
        name = config["name"]
        ModelClass = config["model"]
        params = config["params"].copy()

        try:
            model = ModelClass(**params)
            model.fit(X_train_prep, y_train)

            if hasattr(model, 'n_features_in_'):
                if model.n_features_in_ != expected_n_features:
                    print(f"{name:30} -> ⚠️ Features mismatch: {model.n_features_in_} vs {expected_n_features}")
                    continue

            # Predictions
            y_pred_test_log = model.predict(X_test_prep)
            y_pred_train_log = model.predict(X_train_prep)

            # Return to original scale
            y_pred_test = np.expm1(y_pred_test_log)
            y_pred_train = np.expm1(y_pred_train_log)
            y_test_orig = np.expm1(y_test)
            y_train_orig = np.expm1(y_train)

            # Validation
            valid = True
            issues = []

            if np.any(np.isnan(y_pred_test)) or np.any(np.isinf(y_pred_test)):
                issues.append("NaN/Inf")
                valid = False

            n_negative = (y_pred_test < 0).sum()
            if n_negative > len(y_pred_test) * 0.5:
                issues.append(f"{n_negative} negatives")
                valid = False
            elif n_negative > 0:
                y_pred_test = np.maximum(y_pred_test, 0)
                y_pred_train = np.maximum(y_pred_train, 0)

            if price_bounds:
                if y_pred_test.max() > price_bounds["max"] * 3:
                    issues.append(f"max too high: {y_pred_test.max():.1f}€")
                    valid = False
                if y_pred_test.min() < -price_bounds["min"]:
                    issues.append(f"min too low: {y_pred_test.min():.1f}€")
                    valid = False

            if not valid:
                print(f"{name:30} -> ❌ {', '.join(issues)} - REJECTED")
                continue

            # Metrics
            r2 = r2_score(y_test_orig, y_pred_test)
            mae = mean_absolute_error(y_test_orig, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))

            if r2 < -1.0:
                print(f"{name:30} -> ❌ R² too low ({r2:.3f})")
                continue

            print(f"{name:30} -> ✅ R2 = {r2:6.3f} | MAE = {mae:6.2f}€ | RMSE = {rmse:7.2f}€")

            result = {
                "model_name": name,
                "r2_test": r2,
                "r2_train": r2_score(y_train_orig, y_pred_train),
                "mae": mae,
                "rmse": rmse,
                "model": model
            }
            all_results.append(result)

            if r2 > best_r2:
                best_r2 = r2
                best_result = result
                best_model_obj = model
                best_model_name = name

        except Exception as e:
            print(f"{name:30} -> ERROR: {str(e)[:50]}")
            continue

    if best_result is None and all_results:
        best_result = max(all_results, key=lambda x: x['r2_test'])
        best_model_obj = best_result['model']
        best_model_name = best_result['model_name']
        best_r2 = best_result['r2_test']
        print(f"\n⚠️  Selecting best available: {best_model_name} (R2={best_r2:.3f})")
    elif best_result is None:
        print("\n❌ No valid non-linear model")
        return None

    print(f"\n✅ Best model: {best_model_name} (R2 = {best_result['r2_test']:.3f})")

    best_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", best_model_obj)
    ])

    # Feature Importance
    feature_importance = None
    if hasattr(best_model_obj, "feature_importances_"):
        importances = best_model_obj.feature_importances_
        if len(importances) == len(feature_names):
            imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            feature_importance = imp_df.to_dict()
            print("\n📊 Top 5 important variables:")
            print(imp_df.head(5))

    # SHAP Analysis
    shap_success = False
    if SHAP_AVAILABLE and hasattr(best_model_obj, "feature_importances_"):
        try:
            print("\n  [INFO] Calculating SHAP values...")
            X_sample = X_test_prep[:min(50, len(X_test_prep))]
            explainer = shap.TreeExplainer(best_model_obj)
            shap_values = explainer.shap_values(X_sample)
            shap_success = True
            print("  ✅ SHAP successfully calculated")
        except Exception as e:
            print(f"  ⚠️ SHAP error: {e}")

    return {
        "group": group_name,
        "model_type": "nonlinear",
        "best_model": best_model_name,
        "r2_test": best_result["r2_test"],
        "r2_train": best_result.get("r2_train", 0),
        "r2": best_result["r2_test"],
        "mae": best_result["mae"],
        "rmse": best_result["rmse"],
        "n_samples": len(df_group),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(all_base_features),
        "n_features_after_encoding": expected_n_features,
        "pipeline": best_pipe,
        "feature_importance": feature_importance,
        "shap_success": shap_success,
        "uses_log_transform": True,
        "all_results": [
            {
                "model_name": r["model_name"],
                "r2_test": r["r2_test"],
                "mae": r["mae"],
                "rmse": r["rmse"]
            }
            for r in all_results
        ]
    }


# ────────────────────────────────────────────────
# MAIN TRAINING FUNCTION WITH MLFLOW
# ────────────────────────────────────────────────

def train_by_bindingtype_siren_regularized(file_path: str | Path = None,
                                          run_linear: bool = True,
                                          run_nonlinear: bool = True,
                                          min_samples: int = 30,
                                          save_pipelines: bool = False,
                                          output_dir: Path | str = None,
                                          register_to_mlflow: bool = True):
    """
    Trains REGULARIZED models for each (binding_type, siren) couple
    and registers them in MLflow Registry with versionning
    """
    print("\n" + "=" * 100)
    print("REGULARIZED TRAINING BY COUPLE (BINDING_TYPE * SIREN)".center(100))
    print("=" * 100)

    start_time = time.time()

    if file_path is None:
        file_path = PROJECT_ROOT / "data" / "processed" / "pricing_fully_cleaned.xlsx"
    else:
        file_path = Path(file_path)

    if save_pipelines and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Data loading
    print("\n📂 Loading data...")
    df = pd.read_excel(file_path, engine="openpyxl")

    # Type conversion
    for col in NUM_VARS + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if 'siren' not in df.columns:
        print("[ERROR] Column 'siren' missing")
        return {"linear": [], "nonlinear": [], "created_models": []}

    df['siren'] = df['siren'].astype(str).replace(['nan', 'None', ''], 'UNKNOWN')
    df_model = df[df[TARGET].notna()].copy()

    print(f"📊 Rows with price: {len(df_model)}")

    # Create couple
    df_model['bindingtype_siren'] = df_model.apply(
        lambda row: f"{row.get('binding_type', 'UNKNOWN')}__{row.get('siren', 'UNKNOWN')}",
        axis=1
    )

    # DYNAMIC CALCULATION OF GLOBAL PRICE BOUNDS
    print("\n" + "=" * 60)
    print("📊 DYNAMIC PRICE ANALYSIS")
    print("=" * 60)

    global_price_bounds = compute_dynamic_price_bounds(df_model)

    # Filter couples
    couple_counts = df_model['bindingtype_siren'].value_counts()
    valid_couples = couple_counts[couple_counts >= min_samples].index.tolist()

    print(f"\n🎯 Couples analyzed (≥{min_samples} rows): {len(valid_couples)}")

    results = {
        "linear": [],
        "nonlinear": [],
        "created_models": [],
        "price_bounds": {}
    }

    # MLflow configuration
    if register_to_mlflow:
        mlflow.set_experiment("Pricing_Couple_Models")
        run_name = f"couple-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        mlflow_run = safe_mlflow_start_run(run_name)
        mlflow.set_tag("model_type", "couple_training")
        mlflow.set_tag("run_name", run_name)

        mlflow.log_params({
            "min_samples": min_samples,
            "n_couples": len(valid_couples)
        })

    for i, couple in enumerate(valid_couples, 1):
        df_couple = df_model[df_model['bindingtype_siren'] == couple].copy()
        safe_couple = couple.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('*', '_')

        print("\n" + "#" * 70)
        print(f"# COUPLE {i}/{len(valid_couples)}: {couple} ({len(df_couple)} rows)")
        print("#" * 70)

        # Dynamic bounds for this couple
        price_bounds = compute_dynamic_price_bounds(df_model, 'bindingtype_siren', couple)
        results["price_bounds"][couple] = price_bounds

        # Linear models with log transform
        if run_linear:
            linear_res = train_linear_models_regularized(
                df_couple, couple,
                min_samples=min_samples,
                price_bounds=price_bounds
            )
            if linear_res and isinstance(linear_res, dict):
                results["linear"].append(linear_res)

                if save_pipelines and output_dir and linear_res.get("pipeline"):
                    pipeline_path = output_dir / f"{safe_couple}_linear_regularized.joblib"
                    joblib.dump(linear_res["pipeline"], pipeline_path)
                    print(f"   ✅ Linear pipeline saved: {pipeline_path}")

                if register_to_mlflow and linear_res.get("pipeline"):
                    try:
                        features_num, features_cat = prepare_features(df_couple)
                        model_name = f"PricingModel_{safe_couple}_Linear"

                        sample_input = df_couple[features_num + features_cat].head(5)
                        input_schema = create_custom_signature(sample_input, features_num, features_cat)

                        sample_output = linear_res["pipeline"].predict(sample_input)
                        output_schema = infer_signature(sample_output).inputs
                        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                        metadata = {
                            "group": couple,
                            "model_type": "linear",
                            "best_model": linear_res["best_model"],
                            "cv_r2": float(linear_res["cv_r2"]),
                            "n_samples": int(linear_res["n_samples"]),
                            "n_features": int(linear_res["n_features"]),
                            "formula": linear_res.get("formula"),
                            "timestamp": datetime.now().isoformat(),
                            "uses_log_transform": linear_res.get("uses_log_transform", False),
                            "price_bounds": {
                                "min": price_bounds["min"],
                                "max": price_bounds["max"],
                                "default": price_bounds["default"]
                            }
                        }

                        model_info = mlflow.sklearn.log_model(
                            sk_model=linear_res["pipeline"],
                        name=model_name,
                            signature=signature,
                            registered_model_name=model_name,
                            metadata=metadata
                        )

                        version = getattr(model_info, 'registered_model_version', None)

                        if version:
                            tags = {
                                "model_type": "linear",
                                "group": couple,
                                "best_model": linear_res["best_model"],
                                "cv_r2": str(linear_res["cv_r2"]),
                                "n_samples": str(linear_res["n_samples"]),
                                "training_date": datetime.now().isoformat(),
                                "run_id": mlflow.active_run().info.run_id,
                                "run_name": run_name,
                                "lifecycle_status": "new",
                                "uses_log_transform": str(linear_res.get("uses_log_transform", False)),
                                "price_min": str(price_bounds["min"]),
                                "price_max": str(price_bounds["max"]),
                                "linear_formula": linear_res.get("formula", "")[:500] if linear_res.get("formula") else ""
                            }

                            set_model_version_tags(model_name, int(version), tags)

                            results["created_models"].append({
                                "model_name": model_name,
                                "version": int(version),
                                "group": couple,
                                "type": "linear",
                                "metrics": {"cv_r2": linear_res["cv_r2"], "n_samples": linear_res["n_samples"]},
                                "formula": linear_res.get("formula")
                            })

                            print(f"\n   ✅ Linear model registered in MLflow: {model_name} v{version}")

                    except Exception as e:
                        print(f"   ⚠️ Error registering linear model in MLflow for {couple}: {e}")

        # Non-linear models with log transform
        if run_nonlinear:
            nonlinear_res = train_nonlinear_family_regularized(
                df_couple, couple,
                min_samples=min_samples,
                price_bounds=price_bounds
            )
            if nonlinear_res and isinstance(nonlinear_res, dict):
                results["nonlinear"].append(nonlinear_res)

                if save_pipelines and output_dir and nonlinear_res.get("pipeline"):
                    pipeline_path = output_dir / f"{safe_couple}_nonlinear_regularized.joblib"
                    joblib.dump(nonlinear_res["pipeline"], pipeline_path)
                    print(f"   ✅ Non-linear pipeline saved: {pipeline_path}")

                if register_to_mlflow and nonlinear_res.get("pipeline"):
                    try:
                        features_num, features_cat = prepare_features(df_couple)
                        model_name = f"PricingModel_{safe_couple}_NonLinear"

                        sample_input = df_couple[features_num + features_cat].head(5)
                        input_schema = create_custom_signature(sample_input, features_num, features_cat)

                        sample_output = nonlinear_res["pipeline"].predict(sample_input)
                        output_schema = infer_signature(sample_output).inputs
                        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                        feature_importance_dict, feature_importance_json = extract_feature_importance(
                            nonlinear_res["pipeline"],
                            features_num + features_cat
                        )

                        metadata = {
                            "group": couple,
                            "model_type": "nonlinear",
                            "best_model": nonlinear_res["best_model"],
                            "r2_test": float(nonlinear_res["r2_test"]),
                            "n_samples": int(nonlinear_res["n_samples"]),
                            "n_features": int(nonlinear_res["n_features"]),
                            "timestamp": datetime.now().isoformat(),
                            "uses_log_transform": nonlinear_res.get("uses_log_transform", False),
                            "price_bounds": {
                                "min": price_bounds["min"],
                                "max": price_bounds["max"],
                                "default": price_bounds["default"]
                            }
                        }

                        if feature_importance_dict:
                            metadata["feature_importance"] = feature_importance_dict

                        model_info = mlflow.sklearn.log_model(
                            sk_model=nonlinear_res["pipeline"],
                        name=model_name,
                            signature=signature,
                            registered_model_name=model_name,
                            metadata=metadata
                        )

                        version = getattr(model_info, 'registered_model_version', None)

                        if version:
                            tags = {
                                "model_type": "nonlinear",
                                "group": couple,
                                "best_model": nonlinear_res["best_model"],
                                "r2_test": str(nonlinear_res["r2_test"]),
                                "n_samples": str(nonlinear_res["n_samples"]),
                                "training_date": datetime.now().isoformat(),
                                "run_id": mlflow.active_run().info.run_id,
                                "run_name": run_name,
                                "lifecycle_status": "new",
                                "uses_log_transform": str(nonlinear_res.get("uses_log_transform", False)),
                                "price_min": str(price_bounds["min"]),
                                "price_max": str(price_bounds["max"])
                            }

                            if feature_importance_json:
                                tags["feature_importance"] = feature_importance_json

                            set_model_version_tags(model_name, int(version), tags)

                            results["created_models"].append({
                                "model_name": model_name,
                                "version": int(version),
                                "group": couple,
                                "type": "nonlinear",
                                "metrics": {"r2_test": nonlinear_res["r2_test"], "n_samples": nonlinear_res["n_samples"]},
                                "feature_importance": feature_importance_dict,
                                "shap_success": nonlinear_res.get("shap_success", False)
                            })

                            print(f"\n   ✅ Non-linear model registered in MLflow: {model_name} v{version}")

                    except Exception as e:
                        print(f"   ⚠️ Error registering non-linear model in MLflow for {couple}: {e}")

    # Assign production alias to best models
    if register_to_mlflow and results["created_models"]:
        print("\n" + "=" * 60)
        print("🏆 PRODUCTION ALIAS CONFIGURATION")
        print("=" * 60)

        for model_info in results["created_models"]:
            try:
                client = mlflow.tracking.MlflowClient()
                model_name = model_info["model_name"]
                version = model_info["version"]

                latest_version = None
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    latest_version = max([int(v.version) for v in versions])

                if latest_version and version == latest_version:
                    set_production_alias(model_name, version)

                    set_model_version_tags(model_name, version, {
                        "lifecycle_status": "production",
                        "deployment_date": datetime.now().isoformat()
                    })

                    print(f"   ✅ Production alias configured for {model_name} v{version}")
                else:
                    print(f"   ⚠️ {model_name} v{version} is not the latest version (latest = v{latest_version})")

            except Exception as e:
                print(f"   ⚠️ Error configuring alias for {model_info['model_name']}: {e}")

    elapsed = time.time() - start_time

    if register_to_mlflow:
        mlflow.log_metrics({
            "n_linear_models": len(results["linear"]),
            "n_nonlinear_models": len(results["nonlinear"]),
            "n_registered_models": len(results["created_models"]),
            "training_time_minutes": elapsed / 60
        })

        print(
            f"\n📊 MLflow Run: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")

    print("\n" + "=" * 100)
    print("SUMMARY".center(100))
    print("=" * 100)
    print(f"⏱️  Time: {elapsed / 60:.1f} minutes")
    print(f"📊 Linear models: {len(results['linear'])}")
    print(f"📈 Non-linear models: {len(results['nonlinear'])}")
    print(f"📦 MLflow registered models: {len(results['created_models'])}")
    print("=" * 100)

    if register_to_mlflow:
        return {
            "linear": results["linear"],
            "nonlinear": results["nonlinear"],
            "created_models": results["created_models"],
            "price_bounds": results["price_bounds"],
            "run_id": mlflow.active_run().info.run_id,
            "experiment_id": mlflow.active_run().info.experiment_id,
            "run_name": run_name
        }
    else:
        return results


# ────────────────────────────────────────────────
# RESULTS ANALYSIS AND DISPLAY FUNCTION
# ────────────────────────────────────────────────

def display_couple_results_regularized(couple_results):
    """
    Displays detailed results by couple
    """
    print("\n" + "=" * 120)
    print("DETAILED RESULTS BY COUPLE".center(120))
    print("=" * 120)

    linear_dict = {r["group"]: r for r in couple_results["linear"]}
    nonlinear_dict = {r["group"]: r for r in couple_results["nonlinear"]}
    created_models_dict = {f"{m['group']}_{m['type']}": m for m in couple_results.get("created_models", [])}
    price_bounds = couple_results.get("price_bounds", {})

    all_groups = sorted(set(linear_dict.keys()) | set(nonlinear_dict.keys()))

    for group in all_groups[:20]:  # Limit display to first 20
        print(f"\n{'─' * 80}")
        print(f"📌 COUPLE: {group}")
        print(f"{'─' * 80}")

        bounds = price_bounds.get(group, {})
        if bounds:
            outlier_pct = bounds.get("stats", {}).get("outlier_pct", 0)
            outlier_info = f" (outliers: {outlier_pct:.1f}%)" if outlier_pct > 0 else ""
            print(f"   Dynamic bounds: [{bounds['min']:.2f}€, {bounds['max']:.2f}€] (default: {bounds['default']:.2f}€){outlier_info}")

        if group in linear_dict:
            lin = linear_dict[group]
            mlflow_info = created_models_dict.get(f"{group}_linear", {})
            mlflow_version = mlflow_info.get("version", "N/A")

            print(f"\n  ✅ REGULARIZED LINEAR - {lin['best_model']}")
            print(f"     R² (train): {lin.get('r2_train', 0):.4f}")
            print(f"     MAE: {lin.get('mae_train', 0):.2f}€")
            print(f"     Samples: {lin['n_samples']}")
            print(f"     Log transform: {lin.get('uses_log_transform', False)}")
            print(f"     Average predicted price: {lin.get('predicted_mean', 0):.2f}€ (actual: {lin.get('actual_mean', 0):.2f}€)")

            if lin.get('formula'):
                print(f"     📝 Formula: {lin['formula'][:100]}...")

            if mlflow_version != "N/A":
                print(f"     ✅ MLflow Registry: v{mlflow_version}")
        else:
            print(f"\n  ❌ LINEAR: No valid model")

        if group in nonlinear_dict:
            nl = nonlinear_dict[group]
            mlflow_info = created_models_dict.get(f"{group}_nonlinear", {})
            mlflow_version = mlflow_info.get("version", "N/A")

            print(f"\n  🔷 REGULARIZED NON-LINEAR - {nl['best_model']}")
            print(f"     R² (test): {nl['r2_test']:.4f}")
            print(f"     MAE: {nl.get('mae', 0):.2f}€")
            print(f"     Samples: {nl['n_samples']} (train={nl.get('n_train', 0)}, test={nl.get('n_test', 0)})")
            print(f"     Log transform: {nl.get('uses_log_transform', False)}")
            print(f"     SHAP: {'✅' if nl.get('shap_success', False) else '❌'}")
            if nl.get('feature_importance'):
                print(f"     Feature importance: ✅")
            if mlflow_version != "N/A":
                print(f"     ✅ MLflow Registry: v{mlflow_version}")
        else:
            print(f"\n  ❌ NON-LINEAR: No valid model")

        print()

    if len(all_groups) > 20:
        print(f"\n... and {len(all_groups) - 20} other couples")

    return


# ────────────────────────────────────────────────
# MAIN FUNCTION TO LAUNCH THE ANALYSIS
# ────────────────────────────────────────────────

def run_couple_analysis_regularized(file_path: str | Path = None,
                                   min_samples_couple: int = 30,
                                   register_to_mlflow: bool = True):
    """
    Runs couple analysis WITH REGULARIZATION and MLflow registration
    """
    print("\n" + "=" * 120)
    print("EPAC - COUPLE ANALYSIS (BINDING_TYPE * SIREN) WITH REGULARIZATION".center(120))
    print("=" * 120)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    couple_dir = MODELS_DIR / f"couple_regularized_{timestamp}"
    couple_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "*" * 60)
    print("REGULARIZED MODELS BY COUPLE".center(60))
    print("*" * 60)

    couple_results = train_by_bindingtype_siren_regularized(
        file_path=file_path,
        run_linear=True,
        run_nonlinear=True,
        min_samples=min_samples_couple,
        save_pipelines=True,
        output_dir=couple_dir / "pipelines",
        register_to_mlflow=register_to_mlflow
    )

    # Display results
    display_couple_results_regularized(couple_results)

    # Save results
    summary = {
        "timestamp": timestamp,
        "min_samples": min_samples_couple,
        "n_couples_total": len(couple_results.get("price_bounds", {})),
        "n_linear": len(couple_results["linear"]),
        "n_nonlinear": len(couple_results["nonlinear"]),
        "n_registered": len(couple_results.get("created_models", [])),
        "linear_results": [
            {
                "group": r["group"],
                "best_model": r["best_model"],
                "r2_train": r.get("r2_train", 0),
                "mae_train": r.get("mae_train", 0),
                "n_samples": r["n_samples"],
                "has_formula": r.get("formula") is not None
            }
            for r in couple_results["linear"]
        ],
        "nonlinear_results": [
            {
                "group": r["group"],
                "best_model": r["best_model"],
                "r2_test": r["r2_test"],
                "mae": r.get("mae", 0),
                "n_samples": r["n_samples"],
                "shap_success": r.get("shap_success", False),
                "has_feature_importance": r.get("feature_importance") is not None
            }
            for r in couple_results["nonlinear"]
        ]
    }

    with open(couple_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save registered models
    if couple_results.get("created_models"):
        mlflow_models_df = pd.DataFrame(couple_results["created_models"])
        mlflow_models_df.to_csv(couple_dir / "mlflow_registered_models.csv", index=False)

    print(f"\n✅ Results saved to: {couple_dir}")
    print("\n" + "=" * 120)
    print("COUPLE ANALYSIS COMPLETED SUCCESSFULLY!".center(120))
    print("=" * 120)

    return couple_results


# ────────────────────────────────────────────────
# SIMPLIFIED FUNCTION FOR INTEGRATION
# ────────────────────────────────────────────────

def train_by_bindingtype_siren(file_path=None, save_pipelines=False, output_dir=None):
    """
    Couple training function - Complete regularized version
    """
    return train_by_bindingtype_siren_regularized(
        file_path=file_path,
        run_linear=True,
        run_nonlinear=True,
        min_samples=MIN_SAMPLES,
        save_pipelines=save_pipelines,
        output_dir=output_dir,
        register_to_mlflow=True
    )


# ────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EPAC couple analysis with ultra-strong regularization")
    parser.add_argument("--file", type=str, help="Path to data file")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum samples per couple")
    parser.add_argument("--no-register", action="store_true", help="Do not register in MLflow")

    args = parser.parse_args()

    # Main execution with regularization
    results = run_couple_analysis_regularized(
        file_path=args.file,
        min_samples_couple=args.min_samples,
        register_to_mlflow=not args.no_register
    )

