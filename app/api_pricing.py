import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pricing_epac import openssl_patch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import uvicorn
import traceback
import re
import time
# After imports, before the code starts
import os

# Configuration for MinIO (S3 compatible)
os.environ['AWS_ACCESS_KEY_ID'] = 'minio_admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio_password'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Constants
ALIAS_PRODUCTION = "production"
MODEL_NAME_GLOBAL = "PricingModelGlobal"
MODEL_NAME_CLIENT_FEATURES = "ClientFeatures"

# Tags for additional information
TAG_FORMULA = "linear_formula"
TAG_FEATURE_IMPORTANCE = "feature_importance"
TAG_SHAP_SUCCESS = "shap_success"

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pricing MLOps API",
    description="API for pricing predictions with MLflow models",
    version="6.0.0"
)

# ==============================================
# GLOBAL MODEL TRAINING COLUMNS
# ==============================================

# In the model, booleans are in NUM_COLS and treated as numeric (0/1)
NUM_COLS = [
    "quantity",
    "production_page",
    "height",
    "thickness",
    "width",
    # Booleans treated as numeric (0/1)
    "security_label",
    "has_coil",
    "has_insert",
    "has_tab",
    "has_backcover",
    "perf",
    "double_sided_cover",
    "shrinkwrap",
    "three_hole_drill"
]

CAT_COLS = [
    "text_paper_type",
    "text_color",
    "cover_finish_type",
    "cover_color",
    "cover_size",
    "cover_paper_type",
    "head_and_tail",
    "priority_level",
    "binding_type",
    "coil_type",
    "tab_color",
    "insert_paper_type",
    "case_finish_type",
    "spine_type",
    "label_type",
    "siren"
]

ALL_FEATURES = NUM_COLS + CAT_COLS

# Types for conversions - FIXED HERE
# Now all booleans are in INT_COLS because the model expects 0/1 integers
BOOL_COLS = []  # No longer used, kept for compatibility

INT_COLS = [
    "quantity",
    "production_page",
    "security_label",
    "has_coil",
    "has_insert",
    "has_tab",
    "has_backcover",
    "perf",
    "double_sided_cover",
    "shrinkwrap",
    "three_hole_drill"
]

FLOAT_COLS = ["height", "thickness", "width"]

# ==============================================
# KNOWN CATEGORIES VOCABULARY
# ==============================================

KNOWN_CATEGORIES = {
    "text_paper_type": ["BIRCH_W40_TB", "80_GLOSS_TEXT", "70_OFFSET", "NONE", "missing", "unknown"],
    "text_color": ["4/4", "4/0", "1/1", "NONE", "missing", "unknown"],
    "cover_finish_type": ["LAYFLAT-GLOSS", "GLOSS", "MATTE", "NONE", "missing", "unknown"],
    "cover_color": ["4/0", "4/4", "1/0", "NONE", "missing", "unknown"],
    "cover_size": ["L", "M", "S", "XL", "NONE", "missing", "unknown"],
    "cover_paper_type": ["12PT_C1S", "100_GLOSS_TEXT", "80_GLOSS_TEXT", "NONE", "missing", "unknown"],
    "head_and_tail": ["NONE", "BLACK", "WHITE", "BLACK & WHITE", "missing", "unknown"],
    "priority_level": ["NORMAL", "HIGH1", "HIGH2", "LOW", "missing", "unknown"],
    "binding_type": ["SS", "CASEBIND", "PERFECT", "SPIRAL", "missing", "unknown"],
    "coil_type": ["NONE", "METAL", "PLASTIC", "missing", "unknown"],
    "tab_color": ["NONE", "WHITE", "COLOR", "missing", "unknown"],
    "insert_paper_type": ["NONE", "80_GLOSS_TEXT", "70_OFFSET", "missing", "unknown"],
    "case_finish_type": ["NONE", "LAYFLAT-GLOSS", "GLOSS", "missing", "unknown"],
    "spine_type": ["NONE", "ROUND", "SQUARE", "missing", "unknown"],
    "label_type": ["NONE", "STANDARD", "CUSTOM", "missing", "unknown"],
    "siren": ["SAV", "missing", "unknown"]
}

FALLBACK_CATEGORY = "missing"


# ==============================================
# PYDANTIC MODELS
# ==============================================

class PricingRequest(BaseModel):
    """Pricing prediction request"""
    siren: Optional[str] = Field(None, description="Client SIREN (optional)")
    binding_type: Optional[str] = Field(None, description="Binding type (optional)")
    features: Dict[str, Any] = Field(..., description="Product technical features")

    class Config:
        json_schema_extra = {
            "example": {
                "siren": "SAV",
                "binding_type": "SS",
                "features": {
                    "quantity": 500,
                    "production_page": 252,
                    "height": 276.22,
                    "thickness": 13.578,
                    "width": 209.55,
                    "security_label": 0,
                    "has_coil": 0,
                    "has_insert": 0,
                    "has_tab": 0,
                    "has_backcover": 1,
                    "perf": 1,
                    "double_sided_cover": 0,
                    "shrinkwrap": 0,
                    "three_hole_drill": 0,
                    "text_paper_type": "BIRCH_W40_TB",
                    "text_color": "4/4",
                    "cover_finish_type": "LAYFLAT-GLOSS",
                    "cover_color": "4/0",
                    "cover_size": "L",
                    "cover_paper_type": "12PT_C1S",
                    "head_and_tail": "NONE",
                    "priority_level": "NORMAL",
                    "coil_type": "NONE",
                    "tab_color": "NONE",
                    "insert_paper_type": "NONE",
                    "case_finish_type": "NONE",
                    "spine_type": "NONE",
                    "label_type": "NONE"
                }
            }
        }


class ModelMetrics(BaseModel):
    r2: Optional[float] = None


class PredictionGlobal(BaseModel):
    model_name: str
    model_version: int
    prediction: float
    metrics: ModelMetrics = ModelMetrics()
    features_used: List[str] = ALL_FEATURES
    feature_importance: Dict[str, float] = {}  # ADDED: Feature importance for global model
    available: bool = True
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionFamilyLinear(BaseModel):
    model_name: str
    model_version: int
    family: str
    prediction: float
    formula: Optional[str] = None
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionFamilyNonLinear(BaseModel):
    model_name: str
    model_version: int
    family: str
    prediction: float
    feature_importance: Dict[str, float] = {}
    shap_available: bool = False
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionCoupleLinear(BaseModel):
    model_name: str
    model_version: int
    couple: str
    prediction: float
    formula: Optional[str] = None
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionCoupleNonLinear(BaseModel):
    model_name: str
    model_version: int
    couple: str
    prediction: float
    feature_importance: Dict[str, float] = {}
    shap_available: bool = False
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class ClientFeaturesInfo(BaseModel):
    model_name: str
    model_version: int
    siren: str
    client_found: bool
    elasticity: Optional[float] = None
    seniority_years: Optional[float] = None
    recency_days: Optional[float] = None
    avg_price_ht: Optional[float] = None
    n_orders: Optional[int] = None
    price_volatility: Optional[float] = None
    relative_price: Optional[float] = None
    metadata: Dict[str, Any] = {}


class PricingResponse(BaseModel):
    request_id: str
    timestamp: str
    input: Dict[str, Any]
    global_prediction: Optional[PredictionGlobal] = None
    family_linear: Optional[PredictionFamilyLinear] = None
    family_nonlinear: Optional[PredictionFamilyNonLinear] = None
    couple_linear: Optional[PredictionCoupleLinear] = None
    couple_nonlinear: Optional[PredictionCoupleNonLinear] = None
    client_features: Optional[ClientFeaturesInfo] = None
    summary: Dict[str, Any] = {}


# ==============================================
# UTILITY FUNCTIONS
# ==============================================

def sanitize_name(name: str) -> str:
    """Cleans a name for file / MLflow registry / model usage."""
    # Replace all non-alphanumeric characters with underscore
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def extract_formula_from_tags(tags: Dict) -> Optional[str]:
    """Extracts formula from MLflow tags"""
    return tags.get(TAG_FORMULA, None)


def extract_feature_importance_from_tags(tags: Dict) -> Dict[str, float]:
    """Extracts feature importance from MLflow tags"""
    if TAG_FEATURE_IMPORTANCE in tags:
        try:
            return json.loads(tags[TAG_FEATURE_IMPORTANCE])
        except:
            return {}
    return {}


def extract_shap_success_from_tags(tags: Dict) -> bool:
    """Extracts SHAP status from MLflow tags"""
    return tags.get(TAG_SHAP_SUCCESS, "false").lower() == "true"


def get_production_model_info(model_name: str) -> Optional[Dict]:
    """
    Retrieves the production model directly from MLflow
    """
    try:
        # Get the version with production alias
        mv = client.get_model_version_by_alias(model_name, ALIAS_PRODUCTION)
        if not mv:
            logger.warning(f"No production alias found for {model_name}")
            return None

        # Load the model
        model_uri = f"models:/{model_name}@{ALIAS_PRODUCTION}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Get version information
        model_version = client.get_model_version(model_name, mv.version)

        # Extract tags
        tags = model_version.tags if hasattr(model_version, 'tags') else {}

        # Extract metadata (if present)
        metadata = {}
        if "metadata" in tags:
            try:
                metadata = json.loads(tags["metadata"])
            except:
                pass

        # Extract metrics from tags
        metrics = {}

        # Look for R² in tags
        r2_keys = ['r2_test', 'r2', 'test_r2', 'cv_r2', 'performance_r2']
        found_r2 = False

        for key in r2_keys:
            if key in tags:
                try:
                    metrics['r2'] = float(tags[key])
                    logger.info(f"Found R²={metrics['r2']} from tag '{key}' for {model_name}")
                    found_r2 = True
                    break
                except (ValueError, TypeError):
                    continue

        # Look in metadata if not found
        if not found_r2 and metadata:
            for key in r2_keys:
                if key in metadata:
                    try:
                        metrics['r2'] = float(metadata[key])
                        logger.info(f"Found R²={metrics['r2']} from metadata.{key} for {model_name}")
                        found_r2 = True
                        break
                    except (ValueError, TypeError):
                        continue

        # Look for other metrics
        for key, value in tags.items():
            if key.startswith("performance_"):
                metric_name = key.replace("performance_", "")
                try:
                    metrics[metric_name] = float(value)
                except:
                    pass
            elif key in ["rmse", "mae", "mape", "cv_r2"] and key not in metrics:
                try:
                    metrics[key] = float(value)
                except:
                    pass

        # Extract additional information
        formula = extract_formula_from_tags(tags)
        feature_importance = extract_feature_importance_from_tags(tags)
        shap_available = extract_shap_success_from_tags(tags)

        model_info = {
            "name": model_name,
            "version": int(mv.version),
            "alias": ALIAS_PRODUCTION,
            "model": model,
            "run_id": model_version.run_id,
            "tags": tags,
            "metadata": metadata,
            "description": getattr(model_version, 'description', ''),
            "metrics": metrics,  # ← THIS IS WHERE METRICS ARE STORED
            "formula": formula,
            "feature_importance": feature_importance,
            "shap_available": shap_available
        }

        logger.info(f"Loaded production model: {model_name} v{mv.version} with metrics: {metrics}")
        return model_info

    except Exception as e:
        logger.error(f"Error loading production model {model_name}: {e}")
        return None


def get_client_features_data(siren: str) -> Optional[Dict]:
    """Retrieves client features from MLflow"""
    try:
        client_features_info = get_production_model_info(MODEL_NAME_CLIENT_FEATURES)

        if not client_features_info:
            logger.warning("Client features model not available")
            return None

        client_model = client_features_info["model"]

        if hasattr(client_model, 'get_client_features'):
            return client_model.get_client_features(siren)
        else:
            result = client_model.predict(pd.DataFrame({"siren": [siren]}))
            if isinstance(result, pd.DataFrame) and not result.empty:
                return result.iloc[0].to_dict()

        return None
    except Exception as e:
        logger.error(f"Error getting client features: {e}")
        return None


def safe_categorical_value(value: Any, column: str) -> str:
    """Converts a categorical value to a value known by the model"""
    if pd.isna(value) or value is None:
        return FALLBACK_CATEGORY

    str_value = str(value).strip()

    if column in KNOWN_CATEGORIES:
        if str_value in KNOWN_CATEGORIES[column]:
            return str_value
        else:
            logger.warning(f"Unknown category '{str_value}' for column {column}, using fallback '{FALLBACK_CATEGORY}'")
            return FALLBACK_CATEGORY

    return str_value


def build_features_for_model(request: PricingRequest) -> pd.DataFrame:
    """
    Builds a DataFrame with EXACTLY the columns expected by the model
    All boolean columns are treated as integers (0/1)
    """
    data = {}
    warnings = []

    # Add all features with their values (or default values)
    for feature in ALL_FEATURES:
        if feature in request.features:
            value = request.features[feature]

            # Type handling according to column
            if feature in INT_COLS:
                # Convert to integer (0/1 for booleans)
                try:
                    if isinstance(value, bool):
                        data[feature] = 1 if value else 0
                    elif isinstance(value, (int, float)):
                        data[feature] = 1 if float(value) != 0 else 0
                    elif isinstance(value, str):
                        str_val = value.lower().strip()
                        if str_val in ['true', '1', 'yes', 'oui', 'vrai']:
                            data[feature] = 1
                        else:
                            data[feature] = 0
                    else:
                        data[feature] = 0
                        warnings.append(f"Could not convert {feature}={value} to int, using 0")
                except (ValueError, TypeError):
                    data[feature] = 0
                    warnings.append(f"Could not convert {feature}={value} to int, using 0")

            elif feature in FLOAT_COLS:
                # Keep as float
                try:
                    data[feature] = float(value)
                except (ValueError, TypeError):
                    data[feature] = 0.0
                    warnings.append(f"Could not convert {feature}={value} to float, using 0.0")

            else:  # Categorical
                data[feature] = safe_categorical_value(value, feature)

        elif feature == "siren" and request.siren:
            data[feature] = safe_categorical_value(request.siren, feature)

        elif feature == "binding_type" and request.binding_type:
            data[feature] = safe_categorical_value(request.binding_type, feature)

        else:
            # Default values
            if feature in INT_COLS:
                data[feature] = 0
            elif feature in FLOAT_COLS:
                data[feature] = 0.0
            else:
                data[feature] = FALLBACK_CATEGORY

    # Create the DataFrame
    df = pd.DataFrame([data])

    # Ensure columns are in the correct order
    df = df[ALL_FEATURES]

    # Final type conversion - all numeric, NO booleans
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

    logger.info(f"Built DataFrame with {len(df.columns)} columns")
    logger.info(f"Data types: {df.dtypes.to_dict()}")

    if warnings:
        logger.warning(f"Conversion warnings: {warnings}")

    return df, warnings


def extract_metrics_from_tags(tags: Dict) -> ModelMetrics:
    """Extracts metrics from MLflow tags"""
    metrics = {}

    # Exhaustive list of possible keys for R²
    r2_keys = [
        'r2_test', 'r2', 'test_r2', 'cv_r2', 'performance_r2',
        'r2_score', 'R2', 'R²', 'r2_train'
    ]

    # Look for R² in all possible keys
    for key in r2_keys:
        if key in tags:
            try:
                metrics['r2'] = float(tags[key])
                logger.info(f"Found R²={metrics['r2']} from tag '{key}'")
                break
            except (ValueError, TypeError):
                continue

    # If still no R², look in keys containing 'r2'
    if 'r2' not in metrics:
        for key, value in tags.items():
            if 'r2' in key.lower() or 'rsquared' in key.lower():
                try:
                    metrics['r2'] = float(value)
                    logger.info(f"Found R²={metrics['r2']} from tag '{key}'")
                    break
                except (ValueError, TypeError):
                    continue

    # Look for other metrics
    for key, value in tags.items():
        if key.startswith("performance_"):
            metric_name = key.replace("performance_", "")
            try:
                metrics[metric_name] = float(value)
            except:
                pass
        elif key in ["rmse", "mae", "mape"]:
            try:
                metrics[key] = float(value)
            except:
                pass

    return ModelMetrics(**metrics)


def extract_feature_importance(model_info: Dict) -> Dict[str, float]:
    """Extracts feature importance from MLflow tags"""
    try:
        # Priority 1: Already extracted in model_info (from get_production_model_info)
        if "feature_importance" in model_info and model_info["feature_importance"]:
            return model_info["feature_importance"]

        # Priority 2: In tags
        if "tags" in model_info and TAG_FEATURE_IMPORTANCE in model_info["tags"]:
            return json.loads(model_info["tags"][TAG_FEATURE_IMPORTANCE])

        # Priority 3: In metadata
        if "metadata" in model_info and "feature_importance" in model_info["metadata"]:
            return model_info["metadata"]["feature_importance"]
    except Exception as e:
        logger.warning(f"Error extracting feature importance: {e}")

    return {}


def get_model_formula(model_info: Dict) -> Optional[str]:
    """Retrieves model formula from MLflow tags"""
    try:
        # Priority 1: Already extracted in model_info (from get_production_model_info)
        if "formula" in model_info and model_info["formula"]:
            return model_info["formula"]

        # Priority 2: In tags
        if "tags" in model_info and TAG_FORMULA in model_info["tags"]:
            return model_info["tags"][TAG_FORMULA]

        # Priority 3: In metadata
        if "metadata" in model_info and "formula" in model_info["metadata"]:
            return model_info["metadata"]["formula"]
    except Exception as e:
        logger.warning(f"Error extracting formula: {e}")

    return None


# ==============================================
# ENDPOINTS
# ==============================================

@app.get("/")
async def root():
    return {
        "service": "Pricing MLOps API v6",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_features": {
            "num_cols": len(NUM_COLS),
            "cat_cols": len(CAT_COLS),
            "total_features": len(ALL_FEATURES)
        },
        "endpoints": ["/predict", "/health", "/models", "/features/client/{siren}"]
    }


@app.get("/health")
async def health_check():
    try:
        start = time.time()
        experiments = client.search_experiments()
        global_model = get_production_model_info(MODEL_NAME_GLOBAL)
        client_features = get_production_model_info(MODEL_NAME_CLIENT_FEATURES)

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": round((time.time() - start) * 1000, 2),
            "mlflow_connected": True,
            "models": {
                "global": global_model is not None,
                "client_features": client_features is not None
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/models")
async def list_production_models():
    try:
        registered_models = client.search_registered_models()
        production_models = []

        for model in registered_models:
            try:
                mv = client.get_model_version_by_alias(model.name, ALIAS_PRODUCTION)
                if mv:
                    production_models.append({
                        "name": model.name,
                        "version": int(mv.version),
                        "alias": ALIAS_PRODUCTION,
                        "tags": mv.tags if hasattr(mv, 'tags') else {}
                    })
            except:
                continue

        return {"count": len(production_models), "models": production_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PricingResponse)
async def predict(request: PricingRequest):
    import uuid

    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    logger.info(f"Request {request_id}: siren={request.siren}, binding_type={request.binding_type}")

    # Build features for the model with type conversion
    input_df, warnings = build_features_for_model(request)
    logger.info(f"Built DataFrame with columns: {list(input_df.columns)}")
    logger.info(f"Data types: {input_df.dtypes.to_dict()}")

    # Initialize response
    response = PricingResponse(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        input=request.model_dump(),
        summary={}
    )

    # 1. Client features
    if request.siren:
        try:
            client_data = get_client_features_data(request.siren)
            if client_data:
                client_features_info = get_production_model_info(MODEL_NAME_CLIENT_FEATURES)
                response.client_features = ClientFeaturesInfo(
                    model_name=MODEL_NAME_CLIENT_FEATURES,
                    model_version=client_features_info["version"] if client_features_info else 0,
                    siren=request.siren,
                    client_found=True,
                    elasticity=client_data.get("client_price_elasticity"),
                    seniority_years=client_data.get("client_seniority_years"),
                    recency_days=client_data.get("client_recency_days"),
                    avg_price_ht=client_data.get("client_avg_price_ht"),
                    n_orders=client_data.get("client_n_orders"),
                    price_volatility=client_data.get("client_price_volatility"),
                    relative_price=client_data.get("client_relative_price"),
                    metadata={k: v for k, v in client_data.items() if k.startswith("client_")}
                )
        except Exception as e:
            logger.error(f"Client features error: {e}")

    # 2. Global model
    try:
        global_info = get_production_model_info(MODEL_NAME_GLOBAL)
        if global_info:
            # Prediction
            pred = global_info["model"].predict(input_df)[0]

            # Target is log-transformed in the model, so we apply expm1
            # Check if prediction is in log or original scale
            if pred < 0 or pred < 100:  # If it's small, it's probably in log
                pred_original = np.expm1(pred)
            else:
                pred_original = pred

            # ADDED: Extract feature importance for global model
            feature_importance = extract_feature_importance(global_info)

            response.global_prediction = PredictionGlobal(
                model_name=global_info["name"],
                model_version=global_info["version"],
                prediction=float(pred_original),
                metrics=extract_metrics_from_tags(global_info.get("tags", {})),
                feature_importance=feature_importance,  # ADDED: Add feature importance
                warnings=warnings if warnings else None
            )
            logger.info(f"Global prediction: {pred_original:.2f}")
            if feature_importance:
                logger.info(f"   📊 Feature importance available: {len(feature_importance)} features")
    except Exception as e:
        logger.error(f"Global prediction error: {e}")
        logger.error(traceback.format_exc())
        response.global_prediction = PredictionGlobal(
            model_name=MODEL_NAME_GLOBAL,
            model_version=0,
            prediction=0.0,
            available=False,
            error=str(e)
        )

    # 3. Family models
    if request.binding_type:
        # Linear
        try:
            family_linear = get_family_model(request.binding_type, "Linear")
            if family_linear:
                pred = family_linear["model"].predict(input_df)[0]
                # Apply expm1 if necessary
                if pred < 0 or pred < 100:
                    pred_original = np.expm1(pred)
                else:
                    pred_original = pred

                response.family_linear = PredictionFamilyLinear(
                    model_name=family_linear["name"],
                    model_version=family_linear["version"],
                    family=request.binding_type,
                    prediction=float(pred_original),
                    formula=get_model_formula(family_linear),
                    metrics=extract_metrics_from_tags(family_linear.get("tags", {})),
                    available=True,
                    warnings=warnings if warnings else None
                )
        except Exception as e:
            response.family_linear = PredictionFamilyLinear(
                model_name=f"PricingModel_{sanitize_name(request.binding_type)}_Linear",
                model_version=0,
                family=request.binding_type,
                prediction=0.0,
                available=False,
                reason=str(e)[:100]
            )

        # Non-linear
        try:
            family_nonlinear = get_family_model(request.binding_type, "NonLinear")
            if family_nonlinear:
                pred = family_nonlinear["model"].predict(input_df)[0]
                if pred < 0 or pred < 100:
                    pred_original = np.expm1(pred)
                else:
                    pred_original = pred

                response.family_nonlinear = PredictionFamilyNonLinear(
                    model_name=family_nonlinear["name"],
                    model_version=family_nonlinear["version"],
                    family=request.binding_type,
                    prediction=float(pred_original),
                    feature_importance=extract_feature_importance(family_nonlinear),
                    shap_available=family_nonlinear.get("shap_available", False),
                    metrics=extract_metrics_from_tags(family_nonlinear.get("tags", {})),
                    available=True,
                    warnings=warnings if warnings else None
                )
        except Exception as e:
            response.family_nonlinear = PredictionFamilyNonLinear(
                model_name=f"PricingModel_{sanitize_name(request.binding_type)}_NonLinear",
                model_version=0,
                family=request.binding_type,
                prediction=0.0,
                available=False,
                reason=str(e)[:100]
            )

    # 4. Couple models
    if request.siren and request.binding_type:
        couple_key = f"{request.binding_type} × {request.siren}"

        # Linear
        try:
            couple_linear = get_couple_model(request.binding_type, request.siren, "Linear")
            if couple_linear:
                pred = couple_linear["model"].predict(input_df)[0]
                if pred < 0 or pred < 100:
                    pred_original = np.expm1(pred)
                else:
                    pred_original = pred

                response.couple_linear = PredictionCoupleLinear(
                    model_name=couple_linear["name"],
                    model_version=couple_linear["version"],
                    couple=couple_key,
                    prediction=float(pred_original),
                    formula=get_model_formula(couple_linear),
                    metrics=extract_metrics_from_tags(couple_linear.get("tags", {})),
                    available=True,
                    warnings=warnings if warnings else None
                )
        except Exception as e:
            response.couple_linear = PredictionCoupleLinear(
                model_name=f"PricingModel_{sanitize_name(request.binding_type)}__{sanitize_name(request.siren)}_Linear",
                model_version=0,
                couple=couple_key,
                prediction=0.0,
                available=False,
                reason=str(e)[:100]
            )

        # Non-linear
        try:
            couple_nonlinear = get_couple_model(request.binding_type, request.siren, "NonLinear")
            if couple_nonlinear:
                pred = couple_nonlinear["model"].predict(input_df)[0]
                if pred < 0 or pred < 100:
                    pred_original = np.expm1(pred)
                else:
                    pred_original = pred

                response.couple_nonlinear = PredictionCoupleNonLinear(
                    model_name=couple_nonlinear["name"],
                    model_version=couple_nonlinear["version"],
                    couple=couple_key,
                    prediction=float(pred_original),
                    feature_importance=extract_feature_importance(couple_nonlinear),
                    shap_available=couple_nonlinear.get("shap_available", False),
                    metrics=extract_metrics_from_tags(couple_nonlinear.get("tags", {})),
                    available=True,
                    warnings=warnings if warnings else None
                )
        except Exception as e:
            response.couple_nonlinear = PredictionCoupleNonLinear(
                model_name=f"PricingModel_{sanitize_name(request.binding_type)}__{sanitize_name(request.siren)}_NonLinear",
                model_version=0,
                couple=couple_key,
                prediction=0.0,
                available=False,
                reason=str(e)[:100]
            )

    # Summary
    response.summary = {
        "predictions_available": sum([
            1 if response.global_prediction and response.global_prediction.available else 0,
            1 if response.family_linear and response.family_linear.available else 0,
            1 if response.family_nonlinear and response.family_nonlinear.available else 0,
            1 if response.couple_linear and response.couple_linear.available else 0,
            1 if response.couple_nonlinear and response.couple_nonlinear.available else 0
        ]),
        "client_features_found": response.client_features.client_found if response.client_features else False,
        "models_used": [],
        "warnings": warnings if warnings else None
    }

    if response.global_prediction and response.global_prediction.available:
        response.summary["models_used"].append("global")
    if response.family_linear and response.family_linear.available:
        response.summary["models_used"].append("family_linear")
    if response.family_nonlinear and response.family_nonlinear.available:
        response.summary["models_used"].append("family_nonlinear")
    if response.couple_linear and response.couple_linear.available:
        response.summary["models_used"].append("couple_linear")
    if response.couple_nonlinear and response.couple_nonlinear.available:
        response.summary["models_used"].append("couple_nonlinear")

    logger.info(f"Request {request_id} completed - {response.summary}")
    return response


def get_family_model(binding_type: str, model_type: str) -> Optional[Dict]:
    safe = sanitize_name(binding_type)
    return get_production_model_info(f"PricingModel_{safe}_{model_type}")


def get_couple_model(binding_type: str, siren: str, model_type: str) -> Optional[Dict]:
    """
    Retrieves the couple model with the correct format: PricingModel_SS__SAV_Linear
    Note: double underscore between binding_type and siren
    """
    safe_binding = sanitize_name(binding_type)
    safe_siren = sanitize_name(siren)
    model_name = f"PricingModel_{safe_binding}__{safe_siren}_{model_type}"
    return get_production_model_info(model_name)


@app.get("/features/client/{siren}")
async def get_client_features_endpoint(siren: str):
    """Retrieves features for a client"""
    data = get_client_features_data(siren)
    if not data:
        return {"siren": siren, "found": False}
    return {"siren": siren, "found": True, "features": data}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    args = parser.parse_args()

    if args.mlflow_uri:
        MLFLOW_TRACKING_URI = args.mlflow_uri
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

    print("\n" + "=" * 60)
    print("🚀 PRICING MLOPS API V6".center(60))
    print("=" * 60)
    print(f"\n📡 MLflow: {MLFLOW_TRACKING_URI}")
    print(f"🌐 API: http://{args.host}:{args.port}")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")
    print(f"\n📊 Model features:")
    print(f"   - Numeric (including booleans): {len(NUM_COLS)}")
    print(f"   - Categorical: {len(CAT_COLS)}")
    print(f"   - Total: {len(ALL_FEATURES)}")
    print("=" * 60)

    uvicorn.run("api_pricing_final_v6:app", host=args.host, port=args.port)


@app.get("/debug/mlflow")
async def debug_mlflow():
    """MLflow diagnostic endpoint"""
    results = {}

    try:
        # 1. Check connection
        results["mlflow_uri"] = MLFLOW_TRACKING_URI
        results["mlflow_connected"] = False

        experiments = client.search_experiments()
        results["mlflow_connected"] = True
        results["experiments_count"] = len(experiments)

        # 2. Check PricingModelGlobal
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME_GLOBAL, ALIAS_PRODUCTION)
            if mv:
                results["global_model"] = {
                    "name": MODEL_NAME_GLOBAL,
                    "version": mv.version,
                    "alias": ALIAS_PRODUCTION,
                    "run_id": mv.run_id,
                    "status": mv.status
                }

                # Test loading
                model_uri = f"models:/{MODEL_NAME_GLOBAL}@{ALIAS_PRODUCTION}"
                model = mlflow.pyfunc.load_model(model_uri)
                results["global_model"]["loaded"] = True
            else:
                results["global_model"] = {"error": "No production alias"}
        except Exception as e:
            results["global_model"] = {"error": str(e)}

        # 3. Check ClientFeatures
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME_CLIENT_FEATURES, ALIAS_PRODUCTION)
            results["client_features_model"] = {
                "exists": mv is not None,
                "version": mv.version if mv else None
            }
        except:
            results["client_features_model"] = {"exists": False}

        return results
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}