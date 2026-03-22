# ============================================================
# EPAC – HYBRID TRAINING WITH ULTRA-STRONG REGULARIZATION
#     - Family-based approach (BindingType)
#     - LOG TRANSFORMATION of prices
#     - OUTLIER CLIPPING
#     - ROBUST SCALER instead of StandardScaler
#     - Linear models with EXTREME REGULARIZATION
#     - Non-linear models with ULTRA-STRONG CONSTRAINTS
#     - Cross-validation with stability check
#     - REINFORCED prediction plausibility tests
#     - MEDIAN FALLBACK if all models fail
#     - MLFLOW REGISTRY REGISTRATION WITH VERSIONNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import warnings
import os
import sys
# Add root path to find openssl_patch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pricing__epac import openssl_patch
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, \
    StandardScaler  # Changed: RobustScaler instead of StandardScaler
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

# For VIF (info only)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Disable pyOpenSSL in urllib3 (even if not installed)
os.environ['URLLIB3_USE_PYOPENSSL'] = '0'
warnings.filterwarnings('ignore', module='urllib3.contrib.pyopenssl')
# Configuration for MinIO (S3 compatible)
os.environ['AWS_ACCESS_KEY_ID'] = 'minio_admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio_password'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
# Explainability
try:
    import shap

    SHAP_AVAILABLE = True
    print("[INFO] SHAP successfully installed")
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not installed. Install it with: pip install shap")

import joblib
# After imports, before the code starts
import os

# Configuration for MinIO (S3 compatible)
os.environ['AWS_ACCESS_KEY_ID'] = 'minio_admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio_password'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ────────────────────────────────────────────────
# INITIAL FEATURE CONFIGURATION
# ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "hybrid_approach_regularized"
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


# ────────────────────────────────────────────────
# DYNAMIC ANALYSIS OF PRICE DISTRIBUTIONS
# ────────────────────────────────────────────────

def compute_dynamic_price_bounds(df, family_column='binding_type', outlier_multiplier=3):
    """
    Dynamically calculates price bounds for each family from data
    with outlier detection and exclusion
    """
    print("\n" + "=" * 60)
    print("📊 DYNAMIC CALCULATION OF PRICE BOUNDS PER FAMILY")
    print("=" * 60)

    price_bounds = {}
    all_prices = df[TARGET].dropna()

    # Global statistics for DEFAULT family
    global_q1 = all_prices.quantile(0.25)
    global_q3 = all_prices.quantile(0.75)
    global_iqr = global_q3 - global_q1
    global_lower = global_q1 - outlier_multiplier * global_iqr
    global_upper = global_q3 + outlier_multiplier * global_iqr

    # Filter global outliers for base statistics
    clean_global_prices = all_prices[(all_prices >= global_lower) & (all_prices <= global_upper)]

    price_bounds["DEFAULT"] = {
        "min": float(max(0.5, clean_global_prices.quantile(0.01))),
        "max": float(min(MAX_REASONABLE_PRICE, clean_global_prices.quantile(0.99) * 1.5)),
        "default": float(clean_global_prices.median()),
        "stats": {
            "n_samples": len(clean_global_prices),
            "min": float(clean_global_prices.min()),
            "max": float(clean_global_prices.max()),
            "mean": float(clean_global_prices.mean()),
            "median": float(clean_global_prices.median()),
            "q1": float(clean_global_prices.quantile(0.25)),
            "q3": float(clean_global_prices.quantile(0.75)),
            "p5": float(clean_global_prices.quantile(0.05)),
            "p95": float(clean_global_prices.quantile(0.95)),
            "p99": float(clean_global_prices.quantile(0.99))
        }
    }

    print(f"\n📌 DEFAULT (all families, {len(clean_global_prices)} clean samples):")
    print(f"   Calculated bounds: [{price_bounds['DEFAULT']['min']:.2f}€, {price_bounds['DEFAULT']['max']:.2f}€]")
    print(f"   Default price: {price_bounds['DEFAULT']['default']:.2f}€")

    # Get all unique families
    families = df[family_column].unique() if family_column in df.columns else []

    for family in families:
        family_data = df[df[family_column] == family][TARGET].dropna()

        if len(family_data) < 10:  # Not enough data for this family
            print(f"\n   ⚠️ {family}: {len(family_data)} samples < 10, using global bounds")
            price_bounds[family] = price_bounds["DEFAULT"].copy()
            price_bounds[family]["stats"]["n_samples"] = len(family_data)
            continue

        # Outlier detection in the family
        q1 = family_data.quantile(0.25)
        q3 = family_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - outlier_multiplier * iqr
        upper_bound = q3 + outlier_multiplier * iqr

        # Filter outliers
        clean_family_data = family_data[(family_data >= lower_bound) & (family_data <= upper_bound)]

        if len(clean_family_data) < 5:  # Too little data after filtering
            print(f"\n   ⚠️ {family}: too many outliers, using global bounds")
            price_bounds[family] = price_bounds["DEFAULT"].copy()
            price_bounds[family]["stats"]["n_samples"] = len(family_data)
            price_bounds[family]["stats"]["n_clean"] = len(clean_family_data)
            continue

        # Robust statistics on cleaned data
        p1 = clean_family_data.quantile(0.01)
        p5 = clean_family_data.quantile(0.05)
        p95 = clean_family_data.quantile(0.95)
        p99 = clean_family_data.quantile(0.99)
        median = clean_family_data.median()
        mean = clean_family_data.mean()

        # Calculation of dynamic bounds with safety margins
        min_bound = max(0.5, p1 * 0.8)  # 20% below percentile 1
        max_bound = min(MAX_REASONABLE_PRICE, p99 * 1.5)  # 50% above percentile 99, capped

        # Default price: median (more robust than mean)
        default_price = median

        price_bounds[family] = {
            "min": float(min_bound),
            "max": float(max_bound),
            "default": float(default_price),
            "stats": {
                "n_samples": len(family_data),
                "n_clean": len(clean_family_data),
                "outliers_removed": len(family_data) - len(clean_family_data),
                "outlier_pct": (len(family_data) - len(clean_family_data)) / len(family_data) * 100,
                "min_raw": float(family_data.min()),
                "max_raw": float(family_data.max()),
                "min_clean": float(clean_family_data.min()),
                "max_clean": float(clean_family_data.max()),
                "mean": float(mean),
                "median": float(median),
                "q1": float(clean_family_data.quantile(0.25)),
                "q3": float(clean_family_data.quantile(0.75)),
                "p1": float(p1),
                "p5": float(p5),
                "p95": float(p95),
                "p99": float(p99)
            }
        }

        print(f"\n📌 {family}:")
        print(
            f"   Samples: {len(family_data)} including {price_bounds[family]['stats']['outliers_removed']} outliers ({price_bounds[family]['stats']['outlier_pct']:.1f}%)")
        print(f"   Raw prices: [{family_data.min():.2f}€, {family_data.max():.2f}€]")
        print(f"   Clean prices: [{clean_family_data.min():.2f}€, {clean_family_data.max():.2f}€]")
        print(f"   Mean: {mean:.2f}€, Median: {median:.2f}€")
        print(f"   P1: {p1:.2f}€, P99: {p99:.2f}€")
        print(f"   Calculated bounds: [{min_bound:.2f}€, {max_bound:.2f}€]")
        print(f"   Default price: {default_price:.2f}€")

    return price_bounds


def analyze_family_price_distribution(df, family_name=None):
    """
    Analyzes price distribution for a family and returns statistics
    and recommended bounds
    """
    print(f"\n📊 PRICE ANALYSIS FOR {family_name if family_name else 'ALL FAMILIES'}")
    print("-" * 50)

    if family_name and 'binding_type' in df.columns:
        family_data = df[df['binding_type'] == family_name]['unit_price'].dropna()
    else:
        family_data = df['unit_price'].dropna()

    if len(family_data) == 0:
        print("   ⚠️ No data available")
        return None

    # Basic statistics
    stats = {
        "n_samples": len(family_data),
        "min": float(family_data.min()),
        "max": float(family_data.max()),
        "mean": float(family_data.mean()),
        "median": float(family_data.median()),
        "std": float(family_data.std()),
        "q1": float(family_data.quantile(0.25)),
        "q3": float(family_data.quantile(0.75)),
        "p5": float(family_data.quantile(0.05)),
        "p95": float(family_data.quantile(0.95)),
        "p99": float(family_data.quantile(0.99))
    }

    # Outlier detection (IQR x 3 method)
    IQR = stats["q3"] - stats["q1"]
    lower_bound = stats["q1"] - 3 * IQR
    upper_bound = stats["q3"] + 3 * IQR
    outliers = family_data[(family_data < lower_bound) | (family_data > upper_bound)]

    stats["outliers"] = {
        "count": len(outliers),
        "percentage": len(outliers) / len(family_data) * 100,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "min_outlier": float(outliers.min()) if len(outliers) > 0 else None,
        "max_outlier": float(outliers.max()) if len(outliers) > 0 else None
    }

    # Recommended bounds for clipping
    stats["recommended_bounds"] = {
        "min": max(0.5, stats["p5"] * 0.5),
        "max": min(MAX_REASONABLE_PRICE, stats["p95"] * 2)
    }

    # Display
    print(f"\n   📈 Statistics:")
    print(f"      Samples: {stats['n_samples']}")
    print(f"      Min: {stats['min']:.2f}€")
    print(f"      Max: {stats['max']:.2f}€")
    print(f"      Mean: {stats['mean']:.2f}€")
    print(f"      Median: {stats['median']:.2f}€")
    print(f"      Std: {stats['std']:.2f}€")
    print(f"      P5: {stats['p5']:.2f}€")
    print(f"      P95: {stats['p95']:.2f}€")
    print(f"      P99: {stats['p99']:.2f}€")
    print(f"      Outliers: {stats['outliers']['count']} ({stats['outliers']['percentage']:.1f}%)")
    print(
        f"      Recommended bounds: [{stats['recommended_bounds']['min']:.2f}€, {stats['recommended_bounds']['max']:.2f}€]")

    return stats


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
        model = pipeline.named_steps["model"]

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

def validate_predictions_plausible_reinforced(model, X, family_name=None, price_bounds=None):
    """
    Reinforced version of prediction validation with dynamic bounds
    """
    try:
        # If price_bounds is not provided, use very wide default bounds
        if price_bounds is None:
            max_price = MAX_REASONABLE_PRICE
            min_price = 0.5
            safe_min = 0.5
            safe_max = MAX_REASONABLE_PRICE * 1.2
        else:
            # Get bounds for this family
            bounds = price_bounds.get(family_name,
                                      price_bounds.get("DEFAULT",
                                                       {"min": 0.5, "max": MAX_REASONABLE_PRICE, "default": 100}))
            max_price = bounds["max"]
            min_price = bounds["min"]
            safe_min = min_price * 0.8  # 20% margin
            safe_max = max_price * 1.2  # 20% margin
            safe_max = min(safe_max, MAX_REASONABLE_PRICE * 1.5)  # Never exceed absolute cap

        # Sample to test
        X_sample = X.sample(min(100, len(X)), random_state=42)
        y_pred = model.predict(X_sample)

        # Prediction statistics
        pred_max = y_pred.max()
        pred_min = y_pred.min()
        pred_mean = y_pred.mean()
        pred_median = np.median(y_pred)  # FIX: using np.median() instead of .median()
        pred_std = y_pred.std()

        print(f"\n   🔍 Reinforced validation for {family_name if family_name else 'model'}:")
        print(f"      Min: {pred_min:.2f}€")
        print(f"      Max: {pred_max:.2f}€")
        print(f"      Mean: {pred_mean:.2f}€")
        print(f"      Median: {pred_median:.2f}€")
        print(f"      Std: {pred_std:.2f}€")
        print(f"      Dynamic bounds: [{min_price:.2f}€, {max_price:.2f}€]")
        print(f"      Safety bounds: [{safe_min:.2f}€, {safe_max:.2f}€]")

        # Reinforced plausibility tests
        issues = []

        # 1. Extremes test with margins
        if pred_max > safe_max:
            issues.append(f"max too high: {pred_max:.2f}€ > {safe_max:.2f}€")
        elif pred_max > max_price:
            issues.append(f"max high: {pred_max:.2f}€ > {max_price:.2f}€")

        if pred_min < safe_min:
            issues.append(f"min too low: {pred_min:.2f}€ < {safe_min:.2f}€")
        elif pred_min < min_price:
            issues.append(f"min low: {pred_min:.2f}€ < {min_price:.2f}€")

        # 2. Mean test
        if pred_mean > max_price * 1.5:
            issues.append(f"mean too high: {pred_mean:.2f}€")
        elif pred_mean < min_price * 0.5:
            issues.append(f"mean too low: {pred_mean:.2f}€")

        # 3. Standard deviation test
        expected_range = max_price - min_price
        if pred_std > expected_range * 2:
            issues.append(f"std too high: {pred_std:.2f}€ (expected range: {expected_range:.2f}€)")

        # 4. Max/min ratio test
        if pred_min > 0 and pred_max / pred_min > 100:
            issues.append(f"max/min ratio too high: {pred_max / pred_min:.1f}")

        # 5. Extreme percentiles test
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
    models = [
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
        # Ridge with extreme regularization (for comparison)
        {
            "name": "Ridge_Extreme",
            "model": Ridge,
            "params": {
                "alpha": 100.0,
                "random_state": 42
            }
        }
    ]

    return models


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


def calculate_vif(X_num):
    """Calculates VIF for information only"""
    if len(X_num.columns) <= 1:
        return pd.Series()

    X_num = X_num.select_dtypes(include=[np.number])
    if X_num.shape[1] <= 1:
        return pd.Series()

    X_num = X_num.loc[:, X_num.std() > 1e-6]
    if X_num.shape[1] <= 1:
        return pd.Series()

    X_num = X_num.dropna()
    if len(X_num) < X_num.shape[1] + 1:
        return pd.Series()

    try:
        X_vif = add_constant(X_num)
        vif_data = {}
        for i in range(1, X_vif.shape[1]):
            try:
                vif = variance_inflation_factor(X_vif.values, i)
                if np.isfinite(vif) and vif < 1000:
                    vif_data[X_vif.columns[i]] = vif
            except:
                continue
        return pd.Series(vif_data).sort_values(ascending=False)
    except Exception:
        return pd.Series()


def prepare_features(df_family):
    """Prepares features for a given family"""
    features_num = []
    for c in NUM_VARS:
        if c in df_family.columns:
            non_na = df_family[c].notna().sum()
            std_val = df_family[c].std() if non_na > 1 else 0
            if non_na >= 15 and std_val > 1e-6:
                features_num.append(c)

    features_cat = []
    for c in CAT_VARS:
        if c in df_family.columns:
            non_na = df_family[c].notna().sum()
            if non_na >= 8:
                if df_family[c].dtype == "object" or pd.api.types.is_categorical_dtype(df_family[c]):
                    nunique = df_family[c].nunique()
                    if 1 < nunique <= 50:
                        features_cat.append(c)

    return features_num, features_cat


# ────────────────────────────────────────────────
# LINEAR MODELS WITH ULTRA-STRONG REGULARIZATION
# ────────────────────────────────────────────────

def extract_linear_formula(pipeline, features_num, features_cat):
    """Extracts formula from a linear model"""
    try:
        # Check that pipeline has necessary steps
        if pipeline is None:
            return None

        if "reg" not in pipeline.named_steps:
            return None

        reg = pipeline.named_steps["reg"]

        # Check that it's a linear model with coefficients
        if not hasattr(reg, "coef_"):
            return None

        # Check that preprocessor is present
        if "prep" not in pipeline.named_steps:
            return None

        prep = pipeline.named_steps["prep"]

        # Check that categorical encoder is present
        if "cat" not in prep.named_transformers_:
            return None

        intercept = reg.intercept_
        if hasattr(intercept, '__len__') and len(intercept) > 0:
            intercept = intercept[0]  # Take the first if it's an array

        coefs = reg.coef_.flatten() if len(reg.coef_.shape) > 1 else reg.coef_

        # Get feature names after encoding
        cat_encoder = prep.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(features_cat)
        var_names = np.concatenate([features_num, cat_names])

        # Ensure number of coefficients matches
        if len(coefs) != len(var_names):
            print(f"   ⚠️ Number of coefficients ({len(coefs)}) different from number of variables ({len(var_names)})")
            # Truncate or complete
            min_len = min(len(coefs), len(var_names))
            coefs = coefs[:min_len]
            var_names = var_names[:min_len]

        # Build formula
        terms = []

        # Add intercept
        if intercept != 0:
            terms.append(f"{intercept:.4f}")

        # Add significant coefficients
        significant_terms = 0
        for name, coef in zip(var_names, coefs):
            if abs(coef) > 0.0001:  # Ignore very small coefficients
                if coef > 0:
                    if terms:  # If there are already terms
                        terms.append(f"+ {abs(coef):.4f} × {name}")
                    else:
                        terms.append(f"{abs(coef):.4f} × {name}")
                else:
                    terms.append(f"- {abs(coef):.4f} × {name}")
                significant_terms += 1

                # Limit to 20 terms to avoid overly long formulas
                if significant_terms >= 20:
                    terms.append("+ ...")
                    break

        if not terms:
            return "unit_price = 0 (no significant coefficients)"

        # Build final formula
        if terms and terms[0].startswith('+'):
            terms[0] = terms[0][2:]  # Remove initial +

        formula = "unit_price = " + " ".join(terms)

        # Clean formula
        formula = formula.replace("+ -", "- ")
        formula = formula.replace("  ", " ")

        print(f"\n📝 Extracted formula: {formula}")
        return formula

    except Exception as e:
        print(f"⚠️ Error extracting formula: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_linear_models_regularized(df_family, family_name, min_samples=40, cv_folds=5, price_bounds=None):
    """Trains linear models with ULTRA-STRONG REGULARIZATION and reinforced validation"""

    print(f"\n{'=' * 100}")
    print(f"[REGULARIZED LINEAR] {family_name}  |  {len(df_family)} rows")
    print(f"{'=' * 100}\n")

    if len(df_family) < min_samples:
        print("[WARNING] Not enough data\n")
        return None

    features_num, features_cat = prepare_features(df_family)
    all_features = features_num + features_cat

    if len(all_features) < 3:
        print("[WARNING] Not enough variables\n")
        return None

    print("Variables:", ', '.join(all_features), f"({len(all_features)})\n")

    X = df_family[all_features].copy()
    y = df_family[TARGET].copy()

    common_idx = X.index.intersection(y.index)
    if len(common_idx) != len(X):
        X = X.loc[common_idx]
        y = y.loc[common_idx]

    # ========== LOG TRANSFORMATION OF PRICES ==========
    # Prices are often log-normal, predicting log(price) stabilizes
    y_log = np.log1p(y)  # log(1+price) to handle small prices

    print(f"\n   📊 Price statistics:")
    print(f"      Raw prices: min={y.min():.2f}€, mean={y.mean():.2f}€, max={y.max():.2f}€")
    print(
        f"      Log prices: min={np.expm1(y_log.min()):.2f}€, mean={np.expm1(y_log.mean()):.2f}€, max={np.expm1(y_log.max()):.2f}€")

    # ========== OUTLIER CLIPPING ==========
    bounds = price_bounds.get(family_name, price_bounds.get("DEFAULT", {"min": 0.5, "max": MAX_REASONABLE_PRICE}))

    # Moderate clipping (keep data within reasonable limits)
    p1 = y.quantile(0.01)
    p99 = y.quantile(0.99)

    y_clipped = y.clip(lower=max(0.5, p1 * 0.8), upper=min(MAX_REASONABLE_PRICE, p99 * 1.5))
    y_log_clipped = np.log1p(y_clipped)

    n_clipped = (y != y_clipped).sum()
    if n_clipped > 0:
        print(f"\n   ✂️ Clipping: {n_clipped} samples adjusted ({n_clipped / len(y) * 100:.1f}%)")
        print(f"      Bounds: [{max(0.5, p1 * 0.8):.2f}€, {min(MAX_REASONABLE_PRICE, p99 * 1.5):.2f}€]")

    # ========== PREPROCESSOR WITH ROBUST SCALER ==========
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(quantile_range=(5, 95)), features_num),  # More robust to outliers
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

    print(f"\nEvaluation {cv_folds}-fold CV on log(price)...\n")

    for name, model in models.items():
        try:
            pipe = Pipeline([("prep", preprocessor), ("reg", model)])

            # CV on log(price)
            cv_scores = cross_val_score(pipe, X, y_log_clipped, cv=kf, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Final training
            pipe.fit(X, y_log_clipped)

            # Predictions (return to original scale)
            y_pred_log = pipe.predict(X)
            y_pred = np.expm1(y_pred_log)  # expm1 to reverse log1p

            # Safety clipping
            y_pred = np.clip(y_pred, 0.5, bounds['max'] * 1.2)

            n_negative = (y_pred < 0).sum()
            n_aberrant = (y_pred > bounds['max'] * 2).sum()

            print(f"{name:25} -> CV R2(log) = {cv_mean:.4f} (±{cv_std:.4f})", end="")

            if hasattr(pipe.named_steps["reg"], "alpha_"):
                print(f"   alpha={pipe.named_steps['reg'].alpha_:.1f}", end="")

            print(f" | aberrant={n_aberrant}")

            # Reinforced validation
            if n_aberrant == 0 and n_negative == 0:
                # Test on sample for validation
                X_sample = X.sample(min(50, len(X)), random_state=42)
                y_sample_pred = np.expm1(pipe.predict(X_sample))

                # FIX: using np.median() instead of .median()
                if np.median(y_sample_pred) < bounds['max'] * 1.2:
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
            else:
                print(f"      ❌ Rejected ({n_aberrant} aberrant, {n_negative} negative)")

        except Exception as e:
            print(f"{name:25} -> ERROR: {str(e)[:50]}")
            continue

    # ========== FALLBACK IF NO VALID MODEL ==========
    if best_pipe is None:
        print("\n❌ No valid linear model found")

        # FALLBACK SOLUTION 1: median model
        print("\n⚠️  Attempting with median model as fallback...")

        try:
            median_pipe = Pipeline([
                ("prep", preprocessor),
                ("reg", DummyRegressor(strategy='median'))
            ])
            median_pipe.fit(X, y_log_clipped)

            y_pred_median = np.expm1(median_pipe.predict(X))
            median_price = np.median(y_pred_median)  # FIX: using np.median()

            print(f"      Median price: {median_price:.2f}€")

            if median_price < bounds['max'] * 1.2:
                best_pipe = median_pipe
                best_name = "Median_Fallback"
                best_cv_r2 = 0  # R2 null because it's just the median
                print("   ✅ Median model accepted")
            else:
                print("   ❌ Median model rejected (price too high)")

                # FALLBACK SOLUTION 2: reasonable constant price
                print("\n⚠️  Using constant price as last resort...")

                class ConstantModel:
                    def __init__(self, value):
                        self.value = value

                    def fit(self, X, y):
                        return self

                    def predict(self, X):
                        return np.full(X.shape[0] if hasattr(X, 'shape') else 1, self.value)

                    def get_params(self, deep=True):
                        return {}

                fixed_price = min(bounds['default'], 1000)  # Reasonable fixed price
                const_pipe = Pipeline([
                    ("prep", "passthrough"),
                    ("reg", ConstantModel(np.log1p(fixed_price)))
                ])
                const_pipe.fit(X, y_log_clipped)
                best_pipe = const_pipe
                best_name = "Constant_Fallback"
                best_cv_r2 = -np.inf
                print(f"   ✅ Constant model accepted (fixed price = {fixed_price:.2f}€)")

        except Exception as e:
            print(f"   ❌ Fallback models failed: {e}")
            return None

    # Final predictions
    y_pred_log = best_pipe.predict(X)
    y_pred = np.expm1(y_pred_log)

    # Final safety clipping
    y_pred = np.clip(y_pred, 0.5, bounds['max'] * 1.2)

    r2_train = r2_score(y, y_pred)
    mae_train = mean_absolute_error(y, y_pred)

    print(f"\n✅ Best model: {best_name}")
    print(f"   R2 = {r2_train:.4f}")
    print(f"   MAE = {mae_train:.2f}€")
    print(f"   Average prices: predicted={y_pred.mean():.2f}€, actual={y.mean():.2f}€")
    print(f"   Median prices: predicted={np.median(y_pred):.2f}€, actual={np.median(y):.2f}€")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y, y_pred, alpha=0.6, s=45, edgecolor="none")
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1.3)
    axes[0].set_xlabel("Actual price (€)")
    axes[0].set_ylabel("Predicted price (€)")
    axes[0].set_title(f"{family_name} - {best_name}")
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
        "family": family_name,
        "model_type": "linear",
        "best_model": best_name,
        "cv_r2": best_cv_r2,
        "r2_train": r2_train,
        "mae_train": mae_train,
        "n_samples": len(df_family),
        "n_features": len(all_features),
        "pipeline": best_pipe,
        "uses_log_transform": True,
        "predicted_mean": float(y_pred.mean()),
        "predicted_median": float(np.median(y_pred)),
        "actual_mean": float(y.mean()),
        "actual_median": float(np.median(y))
    }


# ────────────────────────────────────────────────
# REGULARIZED NON-LINEAR MODELS
# ────────────────────────────────────────────────

def train_nonlinear_family_regularized(df_family, family_name, min_samples=50, price_bounds=None):
    """Trains non-linear models WITH ULTRA-STRONG REGULARIZATION - Relaxed version"""

    print(f"\n{'=' * 100}")
    print(f"[REGULARIZED NON-LINEAR] {family_name}  |  {len(df_family)} rows")
    print(f"{'=' * 100}\n")

    if len(df_family) < min_samples:
        print("[WARNING] Not enough data")
        return None

    features_num, features_cat = prepare_features(df_family)
    all_base_features = features_num + features_cat

    if len(all_base_features) < 3:
        print("[WARNING] Not enough variables")
        return None

    X = df_family[all_base_features].copy()
    y = df_family[TARGET].copy()

    common_idx = X.index.intersection(y.index)
    if len(common_idx) != len(X):
        X = X.loc[common_idx]
        y = y.loc[common_idx]

    # ========== PRICE ANALYSIS FOR THE FAMILY ==========
    print(f"\n📊 PRICE ANALYSIS FOR {family_name}")
    print("-" * 50)

    y_clean = y[~np.isnan(y) & (y > 0)]

    stats = {
        'min': float(y_clean.min()),
        'max': float(y_clean.max()),
        'mean': float(y_clean.mean()),
        'median': float(y_clean.median()),
        'std': float(y_clean.std()),
        'p5': float(y_clean.quantile(0.05)),
        'p95': float(y_clean.quantile(0.95)),
        'p99': float(y_clean.quantile(0.99)),
        'n_outliers': int((y_clean > y_clean.quantile(0.99)).sum() + (y_clean < y_clean.quantile(0.01)).sum())
    }

    print(f"\n   📈 Statistics:")
    print(f"       Samples: {len(y_clean)}")
    print(f"       Min: {stats['min']:.2f}€")
    print(f"       Max: {stats['max']:.2f}€")
    print(f"       Mean: {stats['mean']:.2f}€")
    print(f"       Median: {stats['median']:.2f}€")
    print(f"       Std: {stats['std']:.2f}€")
    print(f"       P5: {stats['p5']:.2f}€")
    print(f"       P95: {stats['p95']:.2f}€")
    print(f"       P99: {stats['p99']:.2f}€")
    print(f"       Outliers: {stats['n_outliers']} ({stats['n_outliers'] / len(y_clean) * 100:.1f}%)")

    # Adaptive bounds for the family
    if price_bounds and family_name in price_bounds:
        bounds = price_bounds[family_name]
    else:
        # Default bounds adapted to data
        bounds = {
            'min': max(0.1, stats['p5'] * 0.5),  # 50% of percentile 5
            'max': min(1000, stats['p95'] * 2.0)  # 2x percentile 95, max 1000€
        }

    print(f"       Recommended bounds: [{bounds['min']:.2f}€, {bounds['max']:.2f}€]")

    # ========== LOG TRANSFORMATION OF PRICES ==========
    y_log = np.log1p(y.clip(lower=0.01))  # Avoid log(0)

    # ========== LIGHT OUTLIER CLIPPING ==========
    # Only extreme values
    p1 = y.quantile(0.01)
    p99 = y.quantile(0.99)

    y_clipped = y.clip(
        lower=max(0.01, p1 * 0.5),
        upper=min(1000, p99 * 1.5)
    )
    y_log_clipped = np.log1p(y_clipped)

    print(f"\n   ✂️ Clipping: {stats['n_outliers']} samples adjusted ({stats['n_outliers'] / len(y) * 100:.1f}%)")
    print(f"       Bounds: [{y_clipped.min():.2f}€, {y_clipped.max():.2f}€]")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log_clipped, test_size=0.2, random_state=42
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_num),
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
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_prep.shape[1])]

    expected_n_features = X_train_prep.shape[1]
    print(f"\n   🔢 Number of features after preprocessing: {expected_n_features}")

    # ============================================================
    # RELAXED CONFIGURATION - Thresholds adapted to data
    # ============================================================
    models_config = get_regularized_models_config_reinforced()

    # Validation thresholds adapted to the family
    MIN_ACCEPTABLE_R2 = max(-1.0, -len(y) / 100)  # More permissive with little data
    MAX_ACCEPTABLE_RMSE_FACTOR = 2.0  # 2x standard deviation

    y_std = y.std()
    expected_range = bounds["max"] - bounds["min"]

    print(f"\n   🔧 Validation thresholds:")
    print(f"       Minimum R²: {MIN_ACCEPTABLE_R2:.3f}")
    print(f"       Maximum RMSE: {y_std * MAX_ACCEPTABLE_RMSE_FACTOR:.2f}€")

    all_results = []
    best_r2 = -np.inf
    best_result = None
    best_model_obj = None
    best_model_name = None

    print(f"\nEvaluating {len(models_config)} models...\n")

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

            # ADAPTIVE VERIFICATION
            valid_model = True
            issues = []

            # 1. Check NaN/Inf
            if np.any(np.isnan(y_pred_test)) or np.any(np.isinf(y_pred_test)):
                issues.append("NaN/Inf")
                valid_model = False

            # 2. Check negative predictions (very permissive)
            n_negative = (y_pred_test < 0).sum()
            # FIX: n_positive doesn't exist, check percentage of negatives
            if n_negative > len(y_pred_test) * 0.5:  # If more than 50% are negative
                issues.append(f"{n_negative} negatives ({n_negative / len(y_pred_test) * 100:.1f}%)")
                valid_model = False
            elif n_negative > 0:
                print(f"   ⚠️  {name} - {n_negative} negative predictions (forced to 0)")
                y_pred_test = np.maximum(y_pred_test, 0)
                y_pred_train = np.maximum(y_pred_train, 0)

            # 3. Check bounds (very permissive)
            if y_pred_test.max() > bounds['max'] * 3:  # 3x the max
                issues.append(f"max too high: {y_pred_test.max():.1f}€ > {bounds['max'] * 3:.1f}€")
                valid_model = False
            elif y_pred_test.max() > bounds['max']:
                print(f"   ⚠️  {name} - max > bound: {y_pred_test.max():.1f}€ > {bounds['max']:.1f}€")

            if y_pred_test.min() < -bounds['min']:  # Negative but not too much
                issues.append(f"min too low: {y_pred_test.min():.1f}€ < {-bounds['min']:.1f}€")
                valid_model = False
            elif y_pred_test.min() < 0:
                print(f"   ⚠️  {name} - negative min: {y_pred_test.min():.1f}€")

            if not valid_model:
                print(f"{name:30} -> ❌ {', '.join(issues)} - REJECTED")
                continue

            # Metric calculation
            r2 = r2_score(y_test_orig, y_pred_test)
            mae = mean_absolute_error(y_test_orig, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))

            # Metric validation (more permissive)
            if r2 < MIN_ACCEPTABLE_R2:
                print(f"{name:30} -> ⚠️ R² too low ({r2:.3f} < {MIN_ACCEPTABLE_R2}) - REJECTED")
                continue

            if rmse > y_std * MAX_ACCEPTABLE_RMSE_FACTOR:
                print(
                    f"{name:30} -> ⚠️ RMSE too high ({rmse:.2f} > {y_std * MAX_ACCEPTABLE_RMSE_FACTOR:.2f}) - REJECTED")
                continue

            # Stability calculation (info only)
            try:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, X_train_prep, y_train, cv=3, scoring='r2')
                cv_std = cv_scores.std()
                if cv_std > 0.3:
                    print(f"   ⚠️  {name} - unstable: cv_std={cv_std:.3f}")
            except:
                cv_std = None

            print(f"{name:30} -> ✅ R2 = {r2:6.3f} | MAE = {mae:6.2f}€ | RMSE = {rmse:7.2f}€")

            result = {
                "model_name": name,
                "r2_test": r2,
                "r2_train": r2_score(y_train_orig, y_pred_train),
                "mae": mae,
                "rmse": rmse,
                "model": model,
                "n_features_expected": expected_n_features,
                "validation_issues": issues
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

    # If no model found, take the best available even if bad
    if best_result is None and all_results:
        best_result = max(all_results, key=lambda x: x['r2_test'])
        best_model_obj = best_result['model']
        best_model_name = best_result['model_name']
        best_r2 = best_result['r2_test']
        print(f"\n⚠️  No good model, selecting best available: {best_model_name} (R2={best_r2:.3f})")

    elif best_result is None:
        print("\n❌ No valid non-linear model found")
        return None

    print(f"\n✅ Best model: {best_model_name} (R2 = {best_result['r2_test']:.3f})")

    best_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", best_model_obj)
    ])

    # Final predictions
    y_pred_train_log = best_pipe.predict(X_train)
    y_pred_test_log = best_pipe.predict(X_test)

    y_pred_train = np.expm1(y_pred_train_log)
    y_pred_test = np.expm1(y_pred_test_log)
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_train_orig, y_pred_train, alpha=0.6, s=45, edgecolor="none")
    axes[0].plot([y_train_orig.min(), y_train_orig.max()], [y_train_orig.min(), y_train_orig.max()], "r--", lw=2.5)
    axes[0].set_xlabel("Actual price (€)")
    axes[0].set_ylabel("Predicted price (€)")
    axes[0].set_title(f"Train - {best_model_name}")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test_orig, y_pred_test, alpha=0.6, s=45, edgecolor="none")
    axes[1].plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], "r--", lw=2.5)
    axes[1].set_xlabel("Actual price (€)")
    axes[1].set_ylabel("Predicted price (€)")
    axes[1].set_title(f"Test - R2 = {best_result['r2_test']:.3f}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Feature Importance
    feature_importance = None
    if hasattr(best_model_obj, "feature_importances_"):
        importances = best_model_obj.feature_importances_
        if len(importances) == len(feature_names):
            imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            feature_importance = imp_df.to_dict()
            print("\n📊 Top 10 important variables:")
            print(imp_df.head(10))

    # 🔴 IMPORTANT: Include 'r2' for compatibility
    return {
        "family": family_name,
        "model_type": "nonlinear",
        "best_model": best_model_name,
        "r2_test": best_result["r2_test"],
        "r2_train": best_result.get("r2_train", 0),
        "r2": best_result["r2_test"],  # For compatibility
        "mae": best_result["mae"],
        "rmse": best_result["rmse"],
        "n_samples": len(df_family),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(all_base_features),
        "n_features_after_encoding": expected_n_features,
        "pipeline": best_pipe,
        "feature_importance": feature_importance,
        "stability_metrics": best_result.get("stability"),
        "all_results": [
            {
                "model_name": r["model_name"],
                "r2_test": r["r2_test"],
                "mae": r["mae"],
                "rmse": r["rmse"]
            }
            for r in all_results
        ],
        "uses_log_transform": True
    }
# ────────────────────────────────────────────────
# MAIN TRAINING FUNCTION WITH MLFLOW
# ────────────────────────────────────────────────

def train_by_bindingtype_regularized(file_path: str | Path = None,
                                     run_linear: bool = True,
                                     run_nonlinear: bool = True,
                                     min_samples: int = 50,
                                     save_pipelines: bool = False,
                                     output_dir: Path | str = None,
                                     register_to_mlflow: bool = True):
    """
    Trains REGULARIZED models for each binding_type
    and registers them in MLflow Registry with versionning
    """
    print("\n" + "=" * 100)
    print("REGULARIZED TRAINING BY BINDING TYPE".center(100))
    print("=" * 100)

    if file_path is None:
        file_path = PROJECT_ROOT / "data" / "processed" / "pricing_fully_cleaned.xlsx"
    else:
        file_path = Path(file_path)

    if save_pipelines and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Data loading
    df = pd.read_excel(file_path, engine="openpyxl")

    # Type conversion
    for col in NUM_VARS + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_model = df[df[TARGET].notna()].copy()
    print(f"Total rows with price: {len(df_model)}")

    # DYNAMIC CALCULATION OF PRICE BOUNDS
    print("\n" + "=" * 60)
    print("📊 DYNAMIC PRICE ANALYSIS PER FAMILY")
    print("=" * 60)

    price_bounds = compute_dynamic_price_bounds(df_model, outlier_multiplier=3)

    # Save calculated bounds
    bounds_file = MODELS_DIR / "dynamic_price_bounds.json"
    with open(bounds_file, 'w') as f:
        # Convert to JSON serializable
        bounds_serializable = {}
        for family, bounds in price_bounds.items():
            bounds_serializable[family] = {
                "min": bounds["min"],
                "max": bounds["max"],
                "default": bounds["default"]
            }
        json.dump(bounds_serializable, f, indent=2)
    print(f"\n✅ Dynamic bounds saved to: {bounds_file}")

    # Binding types to analyze
    binding_types = [
        ("binding_type", "CASEBIND", "CASEBIND"),
        ("binding_type", "PERFECT", "PERFECT"),
        ("binding_type", "COILHARD", "COILHARD"),
        ("binding_type", "SS", "SS"),
        ("binding_type", "COILSOFT", "COILSOFT"),
        ("binding_type", "COILHARD-TAB", "COILHARD-TAB"),
        ("binding_type", "CASEBIND-ES", "CASEBIND-ES"),
        ("binding_type", "LOOSELEAF-NC", "LOOSELEAF-NC"),
    ]

    results = {
        "linear": [],
        "nonlinear": [],
        "created_models": [],
        "price_bounds": price_bounds
    }

    # MLflow configuration
    if register_to_mlflow:
        mlflow.set_experiment("Pricing_Family_Models")
        run_name = f"family-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        mlflow_run = safe_mlflow_start_run(run_name)
        mlflow.set_tag("model_type", "family_training")
        mlflow.set_tag("git_commit", "unknown")
        mlflow.set_tag("run_name", run_name)

        mlflow.log_params({
            "min_samples": min_samples,
            "n_binding_types": len(
                [bt for bt in binding_types if len(df_model[df_model[bt[0]] == bt[1]]) >= min_samples])
        })

        for family, bounds in price_bounds.items():
            if family != "DEFAULT" and "stats" in bounds:
                mlflow.log_param(f"{family}_price_min", bounds["min"])
                mlflow.log_param(f"{family}_price_max", bounds["max"])
                mlflow.log_param(f"{family}_price_default", bounds["default"])
                mlflow.log_param(f"{family}_n_samples", bounds["stats"].get("n_samples", 0))
                mlflow.log_param(f"{family}_outlier_pct", bounds["stats"].get("outlier_pct", 0))

    for col, val, name in binding_types:
        df_family = df_model[df_model[col] == val].copy()

        if len(df_family) < min_samples:
            print(f"\n[INFO] {name}: {len(df_family)} rows < {min_samples} -> ignored")
            continue

        print(f"\n{'#' * 100}")
        print(f"# FAMILY: {name} ({len(df_family)} samples)")
        print(f"{'#' * 100}")

        family_stats = analyze_family_price_distribution(df_family, name)
        family_features_num, family_features_cat = prepare_features(df_family)

        # Linear models with log transform
        if run_linear:
            linear_res = train_linear_models_regularized(df_family, name, min_samples=min_samples,
                                                         price_bounds=price_bounds)
            if linear_res and isinstance(linear_res, dict):
                # ADD: Extract formula for linear models
                formula = None
                if linear_res.get("pipeline") and linear_res.get("best_model") not in ["Median_Fallback",
                                                                                       "Constant_Fallback"]:
                    try:
                        formula = extract_linear_formula(
                            linear_res["pipeline"],
                            family_features_num,
                            family_features_cat
                        )
                        print(f"   📝 Extracted formula for {name}: {formula[:100]}...")
                    except Exception as e:
                        print(f"   ⚠️ Error extracting formula for {name}: {e}")

                # ADD: Add formula to results dictionary
                linear_res["formula"] = formula

                results["linear"].append(linear_res)

                if save_pipelines and output_dir and linear_res.get("pipeline"):
                    pipeline_path = output_dir / f"{name}_linear_regularized.joblib"
                    joblib.dump(linear_res["pipeline"], pipeline_path)
                    print(f"   ✅ Linear pipeline saved: {pipeline_path}")

                if register_to_mlflow and linear_res.get("pipeline"):
                    try:
                        model_name = f"PricingModel_{name}_Linear"
                        pipeline = linear_res["pipeline"]

                        sample_input = df_family[family_features_num + family_features_cat].head(5)
                        input_schema = create_custom_signature(sample_input, family_features_num, family_features_cat)

                        sample_output = pipeline.predict(sample_input)
                        output_schema = infer_signature(sample_output).inputs
                        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                        metadata = {
                            "family": name,
                            "model_type": "linear",
                            "best_model": linear_res["best_model"],
                            "cv_r2": float(linear_res["cv_r2"]),
                            "n_samples": int(linear_res["n_samples"]),
                            "n_features": int(linear_res["n_features"]),
                            "formula": linear_res.get("formula"),  # ADD: Include formula in metadata
                            "timestamp": datetime.now().isoformat(),
                            "uses_log_transform": linear_res.get("uses_log_transform", False),
                            "price_bounds": {
                                "min": price_bounds.get(name, price_bounds["DEFAULT"])["min"],
                                "max": price_bounds.get(name, price_bounds["DEFAULT"])["max"],
                                "default": price_bounds.get(name, price_bounds["DEFAULT"])["default"]
                            }
                        }

                        model_info = mlflow.sklearn.log_model(
                            sk_model=pipeline,
                            artifact_path=model_name,
                            signature=signature,
                            registered_model_name=model_name,
                            metadata=metadata
                        )

                        version = getattr(model_info, 'registered_model_version', None)

                        if version:
                            tags = {
                                "model_type": "linear",
                                "family": name,
                                "best_model": linear_res["best_model"],
                                "cv_r2": str(linear_res["cv_r2"]),
                                "n_samples": str(linear_res["n_samples"]),
                                "training_date": datetime.now().isoformat(),
                                "run_id": mlflow.active_run().info.run_id,
                                "run_name": run_name,
                                "lifecycle_status": "new",
                                "uses_log_transform": str(linear_res.get("uses_log_transform", False)),
                                "price_min": str(price_bounds.get(name, price_bounds["DEFAULT"])["min"]),
                                "price_max": str(price_bounds.get(name, price_bounds["DEFAULT"])["max"]),
                                "linear_formula": linear_res.get("formula", "")[:500] if linear_res.get(
                                    "formula") else ""  # ADD: Add formula to tags
                            }

                            set_model_version_tags(model_name, int(version), tags)

                            # ADD: Add formula to created_models
                            results["created_models"].append({
                                "model_name": model_name,
                                "version": int(version),
                                "family": name,
                                "type": "linear",
                                "metrics": {"cv_r2": linear_res["cv_r2"], "n_samples": linear_res["n_samples"]},
                                "formula": linear_res.get("formula")  # ADD: Store formula
                            })

                            print(f"\n   ✅ Linear model registered in MLflow: {model_name} v{version}")

                    except Exception as e:
                        print(f"   ⚠️ Error registering linear model in MLflow for {name}: {e}")

        # Non-linear models with log transform
        if run_nonlinear:
            nonlinear_res = train_nonlinear_family_regularized(df_family, name, min_samples=min_samples,
                                                               price_bounds=price_bounds)
            if nonlinear_res and isinstance(nonlinear_res, dict):
                results["nonlinear"].append(nonlinear_res)

                if save_pipelines and output_dir and nonlinear_res.get("pipeline"):
                    pipeline_path = output_dir / f"{name}_nonlinear_regularized.joblib"
                    joblib.dump(nonlinear_res["pipeline"], pipeline_path)
                    print(f"   ✅ Non-linear pipeline saved: {pipeline_path}")

                if register_to_mlflow and nonlinear_res.get("pipeline"):
                    try:
                        model_name = f"PricingModel_{name}_NonLinear"
                        pipeline = nonlinear_res["pipeline"]

                        sample_input = df_family[family_features_num + family_features_cat].head(5)
                        input_schema = create_custom_signature(sample_input, family_features_num, family_features_cat)

                        sample_output = pipeline.predict(sample_input)
                        output_schema = infer_signature(sample_output).inputs
                        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                        feature_importance_dict, feature_importance_json = extract_feature_importance(
                            pipeline,
                            family_features_num + family_features_cat
                        )

                        metadata = {
                            "family": name,
                            "model_type": "nonlinear",
                            "best_model": nonlinear_res["best_model"],
                            "r2_test": float(nonlinear_res["r2_test"]),
                            "n_samples": int(nonlinear_res["n_samples"]),
                            "n_features": int(nonlinear_res["n_features"]),
                            "timestamp": datetime.now().isoformat(),
                            "uses_log_transform": nonlinear_res.get("uses_log_transform", False),
                            "price_bounds": {
                                "min": price_bounds.get(name, price_bounds["DEFAULT"])["min"],
                                "max": price_bounds.get(name, price_bounds["DEFAULT"])["max"],
                                "default": price_bounds.get(name, price_bounds["DEFAULT"])["default"]
                            }
                        }

                        if feature_importance_dict:
                            metadata["feature_importance"] = feature_importance_dict

                        model_info = mlflow.sklearn.log_model(
                            sk_model=pipeline,
                            artifact_path=model_name,
                            signature=signature,
                            registered_model_name=model_name,
                            metadata=metadata
                        )

                        version = getattr(model_info, 'registered_model_version', None)

                        if version:
                            tags = {
                                "model_type": "nonlinear",
                                "family": name,
                                "best_model": nonlinear_res["best_model"],
                                "r2_test": str(nonlinear_res["r2_test"]),
                                "n_samples": str(nonlinear_res["n_samples"]),
                                "training_date": datetime.now().isoformat(),
                                "run_id": mlflow.active_run().info.run_id,
                                "run_name": run_name,
                                "lifecycle_status": "new",
                                "uses_log_transform": str(nonlinear_res.get("uses_log_transform", False)),
                                "price_min": str(price_bounds.get(name, price_bounds["DEFAULT"])["min"]),
                                "price_max": str(price_bounds.get(name, price_bounds["DEFAULT"])["max"])
                            }

                            if feature_importance_json:
                                tags["feature_importance"] = feature_importance_json

                            set_model_version_tags(model_name, int(version), tags)

                            results["created_models"].append({
                                "model_name": model_name,
                                "version": int(version),
                                "family": name,
                                "type": "nonlinear",
                                "metrics": {"r2_test": nonlinear_res["r2_test"],
                                            "n_samples": nonlinear_res["n_samples"]},
                                "feature_importance": feature_importance_dict
                            })

                            print(f"\n   ✅ Non-linear model registered in MLflow: {model_name} v{version}")

                    except Exception as e:
                        print(f"   ⚠️ Error registering non-linear model in MLflow for {name}: {e}")

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

    if register_to_mlflow:
        mlflow.log_metrics({
            "n_linear_models": len(results["linear"]),
            "n_nonlinear_models": len(results["nonlinear"]),
            "n_registered_models": len(results["created_models"])
        })

        print(
            f"\n📊 MLflow Run: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")

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


def display_family_results_regularized(family_results):
    """
    Displays results with regularization metrics
    """
    print("\n" + "=" * 120)
    print("DETAILED RESULTS BY FAMILY (REGULARIZED MODELS)".center(120))
    print("=" * 120)

    linear_dict = {r["family"]: r for r in family_results["linear"]}
    nonlinear_dict = {r["family"]: r for r in family_results["nonlinear"]}
    created_models_dict = {f"{m['family']}_{m['type']}": m for m in family_results.get("created_models", [])}
    price_bounds = family_results.get("price_bounds", {})

    all_families = sorted(set(linear_dict.keys()) | set(nonlinear_dict.keys()))

    for family in all_families:
        print(f"\n{'─' * 80}")
        print(f"📌 FAMILY: {family}")
        print(f"{'─' * 80}")

        bounds = price_bounds.get(family, price_bounds.get("DEFAULT", {}))
        if bounds:
            outlier_pct = bounds.get("stats", {}).get("outlier_pct", 0)
            outlier_info = f" (outliers: {outlier_pct:.1f}%)" if outlier_pct > 0 else ""
            print(
                f"   Dynamic bounds: [{bounds['min']:.2f}€, {bounds['max']:.2f}€] (default: {bounds['default']:.2f}€){outlier_info}")

        if family in linear_dict:
            lin = linear_dict[family]
            mlflow_info = created_models_dict.get(f"{family}_linear", {})
            mlflow_version = mlflow_info.get("version", "N/A")

            print(f"\n  ✅ REGULARIZED LINEAR - {lin['best_model']}")
            print(f"     R² (train): {lin.get('r2_train', 0):.4f}")
            print(f"     MAE: {lin.get('mae_train', 0):.2f}€")
            print(f"     Samples: {lin['n_samples']}")
            print(f"     Log transform: {lin.get('uses_log_transform', False)}")
            print(
                f"     Average predicted price: {lin.get('predicted_mean', 0):.2f}€ (actual: {lin.get('actual_mean', 0):.2f}€)")

            # ADD: Display formula
            if lin.get('formula'):
                print(f"     📝 Formula: {lin['formula'][:100]}...")

            if mlflow_version != "N/A":
                print(f"     ✅ MLflow Registry: v{mlflow_version}")
        else:
            print(f"\n  ❌ LINEAR: No valid model")

        if family in nonlinear_dict:
            nl = nonlinear_dict[family]
            mlflow_info = created_models_dict.get(f"{family}_nonlinear", {})
            mlflow_version = mlflow_info.get("version", "N/A")

            print(f"\n  🔷 REGULARIZED NON-LINEAR - {nl['best_model']}")
            print(f"     R² (test): {nl['r2_test']:.4f}")
            print(f"     R² (train): {nl.get('r2_train', 0):.4f}")
            print(f"     MAE: {nl.get('mae', 0):.2f}€")
            print(f"     Samples: {nl['n_samples']} (train={nl.get('n_train', 0)}, test={nl.get('n_test', 0)})")
            print(f"     Log transform: {nl.get('uses_log_transform', False)}")
            if nl.get('stability_metrics'):
                stab = nl['stability_metrics']
                print(f"     CV Stability: {stab.get('cv_std', 0):.3f}")
                print(f"     Overfitting risk: {stab.get('overfitting_risk', 'N/A')}")
            if mlflow_version != "N/A":
                print(f"     ✅ MLflow Registry: v{mlflow_version}")
        else:
            print(f"\n  ❌ NON-LINEAR: No valid model")

        print()

    # Summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE - REGULARIZED MODELS".center(120))
    print("=" * 120)

    summary_data = []
    for family in all_families:
        row = {"Family": family}

        bounds = price_bounds.get(family, price_bounds.get("DEFAULT", {}))
        if bounds:
            row["Min"] = f"{bounds['min']:.2f}€"
            row["Max"] = f"{bounds['max']:.2f}€"
        else:
            row["Min"] = ""
            row["Max"] = ""

        if family in linear_dict:
            lin = linear_dict[family]
            mlflow_info = created_models_dict.get(f"{family}_linear", {})
            mlflow_version = mlflow_info.get("version", "")

            row["Linear - Model"] = lin['best_model']
            row["Linear - R2"] = f"{lin.get('r2_train', 0):.4f}"
            row["Linear - MAE"] = f"{lin.get('mae_train', 0):.2f}€"
            row["Linear - Version"] = f"v{mlflow_version}" if mlflow_version else ""
            row["Linear - Formula"] = lin.get('formula', '')[:50] + "..." if lin.get('formula') else ""
        else:
            row["Linear - Model"] = "N/A"
            row["Linear - R2"] = "N/A"
            row["Linear - MAE"] = "N/A"
            row["Linear - Version"] = ""
            row["Linear - Formula"] = ""

        if family in nonlinear_dict:
            nl = nonlinear_dict[family]
            mlflow_info = created_models_dict.get(f"{family}_nonlinear", {})
            mlflow_version = mlflow_info.get("version", "")

            row["Non-linear - Model"] = nl['best_model']
            # FIX: using 'r2_test' instead of 'r2'
            row["Non-linear - R2_test"] = f"{nl['r2_test']:.4f}"
            row["Non-linear - MAE"] = f"{nl.get('mae', 0):.2f}€"
            row["Non-linear - Version"] = f"v{mlflow_version}" if mlflow_version else ""
        else:
            row["Non-linear - Model"] = "N/A"
            row["Non-linear - R2_test"] = "N/A"
            row["Non-linear - MAE"] = "N/A"
            row["Non-linear - Version"] = ""

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    return summary_df


def run_family_analysis_regularized(file_path: str | Path = None,
                                    min_samples_family: int = 50,
                                    register_to_mlflow: bool = True):
    """
    Runs family analysis WITH REGULARIZATION and MLflow registration
    """
    print("\n" + "=" * 120)
    print("EPAC - FAMILY ANALYSIS WITH REGULARIZATION".center(120))
    print("=" * 120)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    family_dir = MODELS_DIR / f"family_regularized_{timestamp}"
    family_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "*" * 60)
    print("REGULARIZED MODELS BY FAMILY".center(60))
    print("*" * 60)

    family_results = train_by_bindingtype_regularized(
        file_path=file_path,
        run_linear=True,
        run_nonlinear=True,
        min_samples=min_samples_family,
        save_pipelines=True,
        output_dir=family_dir / "pipelines",
        register_to_mlflow=register_to_mlflow
    )

    # Display results
    summary_df = display_family_results_regularized(family_results)

    # Save
    summary_df.to_csv(family_dir / "family_summary_regularized.csv", index=False)

    # Save registered models in MLflow
    if family_results.get("created_models"):
        mlflow_models_df = pd.DataFrame(family_results["created_models"])
        mlflow_models_df.to_csv(family_dir / "mlflow_registered_models.csv", index=False)

    # Save dynamic bounds
    if family_results.get("price_bounds"):
        bounds_data = []
        for family, bounds in family_results["price_bounds"].items():
            row = {
                "family": family,
                "min": bounds["min"],
                "max": bounds["max"],
                "default": bounds["default"],
            }
            if "stats" in bounds:
                row["n_samples"] = bounds["stats"].get("n_samples", 0)
                row["n_clean"] = bounds["stats"].get("n_clean", 0)
                row["outliers_removed"] = bounds["stats"].get("outliers_removed", 0)
                row["outlier_pct"] = bounds["stats"].get("outlier_pct", 0)
            bounds_data.append(row)

        bounds_df = pd.DataFrame(bounds_data)
        bounds_df.to_csv(family_dir / "dynamic_price_bounds.csv", index=False)

    # Save JSON summary
    summary = {
        "timestamp": timestamp,
        "n_families_linear": len(family_results["linear"]),
        "n_families_nonlinear": len(family_results["nonlinear"]),
        "n_registered_models": len(family_results.get("created_models", [])),
        "price_bounds": {
            k: {"min": v["min"], "max": v["max"], "default": v["default"]}
            for k, v in family_results.get("price_bounds", {}).items()
        },
        "linear_results": [
            {
                "family": lin["family"],
                "best_model": lin["best_model"],
                "r2_train": lin.get("r2_train", 0),
                "mae_train": lin.get("mae_train", 0),
                "n_samples": lin["n_samples"],
                "uses_log_transform": lin.get("uses_log_transform", False),
                "formula": lin.get("formula")  # ADD: Include formula in JSON summary
            }
            for lin in family_results["linear"]
        ],
        "nonlinear_results": [
            {
                "family": nl["family"],
                "best_model": nl["best_model"],
                "r2_test": nl["r2_test"],  # FIX: using 'r2_test'
                "r2_train": nl.get("r2_train", 0),
                "mae": nl.get("mae", 0),
                "n_samples": nl["n_samples"],
                "uses_log_transform": nl.get("uses_log_transform", False),
                "stability": nl.get("stability_metrics", {}).get("overfitting_risk", "UNKNOWN")
            }
            for nl in family_results["nonlinear"]
        ],
        "registered_models": [
            {
                "model_name": m["model_name"],
                "version": m["version"],
                "family": m["family"],
                "type": m["type"],
                "metrics": m["metrics"],
                "formula": m.get("formula")  # ADD: Include formula in registered_models
            }
            for m in family_results.get("created_models", [])
        ]
    }

    with open(family_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {family_dir}")
    print("\n" + "=" * 120)
    print("REGULARIZED ANALYSIS COMPLETED SUCCESSFULLY!".center(120))
    print("=" * 120)

    return family_results


# ────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EPAC family analysis with regularization")
    parser.add_argument("--file", type=str, help="Path to data file")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples per family")
    parser.add_argument("--no-register", action="store_true", help="Do not register in MLflow")

    args = parser.parse_args()

    # Main execution with regularization
    results = run_family_analysis_regularized(
        file_path=args.file,
        min_samples_family=args.min_samples,
        register_to_mlflow=not args.no_register
    )