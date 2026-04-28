from fastapi import FastAPI, HTTPException
import logging
import time
from datetime import datetime
import pandas as pd

from pricing__epac.src.api.schemas.pricing_models import PricingRequest, PricingResponse
from pricing__epac.src.api.services.pricing_service import PricingService
from pricing__epac.src.api.services.mlflow_service import MLflowService
from pricing__epac.src.api.ml.model_loader import ModelLoader
from pricing__epac.src.api.ml.model_registry import ModelRegistry
from pricing__epac.src.config.settings import settings
from pricing__epac.src.config.feature_config import NUM_COLS, CAT_COLS, ALL_FEATURES
from pricing__epac.src.shared.logging import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Pricing MLOps API",
    description="API for pricing predictions with MLflow models",
    version="7.0.0"
)

# Initialize services
pricing_service = PricingService()
mlflow_service = MLflowService()
model_loader = ModelLoader()
model_registry = ModelRegistry()

# Set pandas options
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Pricing MLOps API v7",
        "mlflow_uri": settings.MLFLOW_TRACKING_URI,
        "model_features": {
            "num_cols": len(NUM_COLS),
            "cat_cols": len(CAT_COLS),
            "total_features": len(ALL_FEATURES)
        },
        "endpoints": [
            "/predict",
            "/health",
            "/models",
            "/features/client/{siren}",
            "/debug/mlflow",
            "/admin/reload-model/{model_name}"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        start = time.time()

        mlflow_connected = mlflow_service.check_mlflow_connection()
        global_model = model_loader.get_production_model_info(settings.MODEL_NAME_GLOBAL)
        client_features_model = model_loader.get_production_model_info(
            settings.MODEL_NAME_CLIENT_FEATURES
        )

        return {
            "status": "healthy" if mlflow_connected else "degraded",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": round((time.time() - start) * 1000, 2),
            "mlflow_connected": mlflow_connected,
            "models": {
                "global": global_model is not None,
                "client_features": client_features_model is not None
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}


@app.get("/models")
async def list_production_models():
    """List all production models"""
    try:
        models = model_registry.list_production_models()
        return {"count": len(models), "models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PricingResponse)
async def predict(request: PricingRequest):
    """Main prediction endpoint"""
    return await pricing_service.predict(request)


@app.get("/features/client/{siren}")
async def get_client_features_endpoint(siren: str):
    """Retrieve features for a client"""
    try:
        data = mlflow_service.get_client_features_data(siren)
        if not data:
            return {"siren": siren, "found": False}
        return {"siren": siren, "found": True, "features": data}
    except Exception as e:
        logger.error(f"Error retrieving client features for {siren}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/mlflow")
async def debug_mlflow():
    """MLflow diagnostic endpoint"""
    results = {}

    try:
        # 1. Check connection
        results["mlflow_uri"] = settings.MLFLOW_TRACKING_URI
        results["mlflow_connected"] = mlflow_service.check_mlflow_connection()
        results["experiments_count"] = mlflow_service.get_experiments_count()

        # 2. Check global model
        global_model = model_loader.get_production_model_info(settings.MODEL_NAME_GLOBAL)
        if global_model:
            results["global_model"] = {
                "name": settings.MODEL_NAME_GLOBAL,
                "version": global_model["version"],
                "alias": settings.ALIAS_PRODUCTION,
                "run_id": global_model["run_id"],
                "loaded": True,
                "metrics": global_model.get("metrics", {}),
                "description": global_model.get("description", ""),
                "shap_available": global_model.get("shap_available", False)
            }
        else:
            mv = model_registry.get_model_version_by_alias(settings.MODEL_NAME_GLOBAL)
            results["global_model"] = mv if mv else {"error": "No production alias found"}

        # 3. Check client features model
        client_model = model_loader.get_production_model_info(
            settings.MODEL_NAME_CLIENT_FEATURES
        )
        if client_model:
            results["client_features_model"] = {
                "name": settings.MODEL_NAME_CLIENT_FEATURES,
                "version": client_model["version"],
                "alias": settings.ALIAS_PRODUCTION,
                "run_id": client_model["run_id"],
                "loaded": True,
                "metrics": client_model.get("metrics", {}),
                "description": client_model.get("description", "")
            }
        else:
            mv = model_registry.get_model_version_by_alias(settings.MODEL_NAME_CLIENT_FEATURES)
            results["client_features_model"] = mv if mv else {
                "error": "No production alias found"
            }

        # 4. Loader behavior
        results["loader_mode"] = "no_cache"
        results["loaded_models"] = []

        return results

    except Exception as e:
        logger.error(f"MLflow debug endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reload-model/{model_name}")
async def reload_model(model_name: str):
    """
    Admin endpoint to force a fresh fetch of a model from MLflow.

    Note:
    ModelLoader is no-cache, so there is no true reload operation.
    This endpoint simply re-fetches the production model info.
    """
    try:
        model_info = model_loader.get_production_model_info(model_name)

        if model_info:
            return {
                "status": "success",
                "message": "Model fetched fresh from MLflow",
                "model": model_name,
                "version": model_info["version"],
                "alias": model_info["alias"],
                "run_id": model_info["run_id"]
            }

        return {
            "status": "error",
            "message": f"Model {model_name} not found in MLflow with alias '{settings.ALIAS_PRODUCTION}'"
        }

    except Exception as e:
        logger.error(f"Error reloading model {model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
