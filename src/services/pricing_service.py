import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import traceback
import numpy as np
import pandas as pd

from src.ml.model_loader import ModelLoader
from src.services.feature_service import FeatureService
from src.api.models.pricing_models import (
    PricingRequest, PricingResponse, PredictionGlobal,
    PredictionFamilyLinear, PredictionFamilyNonLinear,
    PredictionCoupleLinear, PredictionCoupleNonLinear,
    ClientFeaturesInfo, ModelMetrics
)
from src.config.settings import settings

logger = logging.getLogger(__name__)


class PricingService:
    """Business logic for pricing predictions - loads models fresh each request"""

    def __init__(self):
        self.model_loader = ModelLoader()  # Plus de singleton, nouvelle instance à chaque requête
        self.feature_service = FeatureService()

    async def predict(self, request: PricingRequest) -> PricingResponse:
        """Main prediction logic - loads models fresh from MLflow"""
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        logger.info(f"Request {request_id}: siren={request.siren}, binding_type={request.binding_type}")

        # Build features
        input_df, warnings = self.feature_service.build_features_for_request(request)
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
            response.client_features = await self._get_client_features(request.siren)

        # 2. Global model prediction
        response.global_prediction = await self._predict_global(input_df, warnings)

        # 3. Family models
        if request.binding_type:
            response.family_linear = await self._predict_family_linear(
                request.binding_type, input_df, warnings
            )
            response.family_nonlinear = await self._predict_family_nonlinear(
                request.binding_type, input_df, warnings
            )

        # 4. Couple models
        if request.siren and request.binding_type:
            response.couple_linear = await self._predict_couple_linear(
                request.binding_type, request.siren, input_df, warnings
            )
            response.couple_nonlinear = await self._predict_couple_nonlinear(
                request.binding_type, request.siren, input_df, warnings
            )

        # Build summary
        response.summary = self._build_summary(response, warnings)

        logger.info(f"Request {request_id} completed - {response.summary}")
        return response

    async def _get_client_features(self, siren: str) -> Optional[ClientFeaturesInfo]:
        """Get client features if available - fresh from MLflow"""
        try:
            client_data = self.model_loader.get_client_features_data(siren)
            if client_data:
                model_info = self.model_loader.get_production_model_info(settings.MODEL_NAME_CLIENT_FEATURES)

                return ClientFeaturesInfo(
                    model_name=settings.MODEL_NAME_CLIENT_FEATURES,
                    model_version=model_info["version"] if model_info else 0,
                    siren=siren,
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

        return None

    async def _predict_global(self, input_df, warnings) -> Optional[PredictionGlobal]:
        """Global model prediction - fresh from MLflow"""
        try:
            model_info = self.model_loader.get_production_model_info(settings.MODEL_NAME_GLOBAL)
            if not model_info:
                return PredictionGlobal(
                    model_name=settings.MODEL_NAME_GLOBAL,
                    model_version=0,
                    prediction=0.0,
                    available=False,
                    error="Model not found in MLflow"
                )

            pred = model_info["model"].predict(input_df)[0]
            pred_original = self.feature_service.transform_prediction(pred)

            logger.info(f"Global prediction: {pred_original:.2f}")
            if model_info.get("feature_importance"):
                logger.info(f"   📊 Feature importance available: {len(model_info['feature_importance'])} features")

            return PredictionGlobal(
                model_name=model_info["name"],
                model_version=model_info["version"],
                prediction=pred_original,
                metrics=ModelMetrics(**model_info.get("metrics", {})),
                feature_importance=model_info.get("feature_importance", {}),
                warnings=warnings if warnings else None,
                available=True
            )
        except Exception as e:
            logger.error(f"Global prediction error: {e}")
            logger.error(traceback.format_exc())
            return PredictionGlobal(
                model_name=settings.MODEL_NAME_GLOBAL,
                model_version=0,
                prediction=0.0,
                available=False,
                error=str(e)
            )

    async def _predict_family_linear(self, binding_type, input_df, warnings):
        """Family linear model prediction - fresh from MLflow"""
        try:
            model_info = self.model_loader.get_family_model(binding_type, "Linear")
            if not model_info:
                return self._create_empty_family_response(
                    f"PricingModel_{self.model_loader._sanitize_name(binding_type)}_Linear",
                    binding_type, True, "Model not found in MLflow"
                )

            pred = model_info["model"].predict(input_df)[0]
            pred_original = self.feature_service.transform_prediction(pred)

            return PredictionFamilyLinear(
                model_name=model_info["name"],
                model_version=model_info["version"],
                family=binding_type,
                prediction=pred_original,
                formula=model_info.get("formula"),
                metrics=ModelMetrics(**model_info.get("metrics", {})),
                warnings=warnings if warnings else None,
                available=True
            )
        except Exception as e:
            return self._create_empty_family_response(
                f"PricingModel_{self.model_loader._sanitize_name(binding_type)}_Linear",
                binding_type, True, str(e)[:100]
            )

    async def _predict_family_nonlinear(self, binding_type, input_df, warnings):
        """Family nonlinear model prediction - fresh from MLflow"""
        try:
            model_info = self.model_loader.get_family_model(binding_type, "NonLinear")
            if not model_info:
                return self._create_empty_family_response(
                    f"PricingModel_{self.model_loader._sanitize_name(binding_type)}_NonLinear",
                    binding_type, False, "Model not found in MLflow"
                )

            pred = model_info["model"].predict(input_df)[0]
            pred_original = self.feature_service.transform_prediction(pred)

            return PredictionFamilyNonLinear(
                model_name=model_info["name"],
                model_version=model_info["version"],
                family=binding_type,
                prediction=pred_original,
                feature_importance=model_info.get("feature_importance", {}),
                shap_available=model_info.get("shap_available", False),
                metrics=ModelMetrics(**model_info.get("metrics", {})),
                warnings=warnings if warnings else None,
                available=True
            )
        except Exception as e:
            return self._create_empty_family_response(
                f"PricingModel_{self.model_loader._sanitize_name(binding_type)}_NonLinear",
                binding_type, False, str(e)[:100]
            )

    async def _predict_couple_linear(self, binding_type, siren, input_df, warnings):
        """Couple linear model prediction - fresh from MLflow"""
        couple_key = f"{binding_type} × {siren}"
        try:
            model_info = self.model_loader.get_couple_model(binding_type, siren, "Linear")
            if not model_info:
                return self._create_empty_couple_response(
                    f"PricingModel_{self.model_loader._sanitize_name(binding_type)}__{self.model_loader._sanitize_name(siren)}_Linear",
                    couple_key, True, "Model not found in MLflow"
                )

            pred = model_info["model"].predict(input_df)[0]
            pred_original = self.feature_service.transform_prediction(pred)

            return PredictionCoupleLinear(
                model_name=model_info["name"],
                model_version=model_info["version"],
                couple=couple_key,
                prediction=pred_original,
                formula=model_info.get("formula"),
                metrics=ModelMetrics(**model_info.get("metrics", {})),
                warnings=warnings if warnings else None,
                available=True
            )
        except Exception as e:
            return self._create_empty_couple_response(
                f"PricingModel_{self.model_loader._sanitize_name(binding_type)}__{self.model_loader._sanitize_name(siren)}_Linear",
                couple_key, True, str(e)[:100]
            )

    async def _predict_couple_nonlinear(self, binding_type, siren, input_df, warnings):
        """Couple nonlinear model prediction - fresh from MLflow"""
        couple_key = f"{binding_type} × {siren}"
        try:
            model_info = self.model_loader.get_couple_model(binding_type, siren, "NonLinear")
            if not model_info:
                return self._create_empty_couple_response(
                    f"PricingModel_{self.model_loader._sanitize_name(binding_type)}__{self.model_loader._sanitize_name(siren)}_NonLinear",
                    couple_key, False, "Model not found in MLflow"
                )

            pred = model_info["model"].predict(input_df)[0]
            pred_original = self.feature_service.transform_prediction(pred)

            return PredictionCoupleNonLinear(
                model_name=model_info["name"],
                model_version=model_info["version"],
                couple=couple_key,
                prediction=pred_original,
                feature_importance=model_info.get("feature_importance", {}),
                shap_available=model_info.get("shap_available", False),
                metrics=ModelMetrics(**model_info.get("metrics", {})),
                warnings=warnings if warnings else None,
                available=True
            )
        except Exception as e:
            return self._create_empty_couple_response(
                f"PricingModel_{self.model_loader._sanitize_name(binding_type)}__{self.model_loader._sanitize_name(siren)}_NonLinear",
                couple_key, False, str(e)[:100]
            )

    def _create_empty_family_response(self, model_name, family, is_linear, reason=None):
        """Create empty family response"""
        if is_linear:
            return PredictionFamilyLinear(
                model_name=model_name,
                model_version=0,
                family=family,
                prediction=0.0,
                available=False,
                reason=reason
            )
        else:
            return PredictionFamilyNonLinear(
                model_name=model_name,
                model_version=0,
                family=family,
                prediction=0.0,
                available=False,
                reason=reason
            )

    def _create_empty_couple_response(self, model_name, couple_key, is_linear, reason=None):
        """Create empty couple response"""
        if is_linear:
            return PredictionCoupleLinear(
                model_name=model_name,
                model_version=0,
                couple=couple_key,
                prediction=0.0,
                available=False,
                reason=reason
            )
        else:
            return PredictionCoupleNonLinear(
                model_name=model_name,
                model_version=0,
                couple=couple_key,
                prediction=0.0,
                available=False,
                reason=reason
            )

    def _build_summary(self, response: PricingResponse, warnings: List[str]) -> Dict:
        """Build response summary"""
        summary = {
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
            summary["models_used"].append("global")
        if response.family_linear and response.family_linear.available:
            summary["models_used"].append("family_linear")
        if response.family_nonlinear and response.family_nonlinear.available:
            summary["models_used"].append("family_nonlinear")
        if response.couple_linear and response.couple_linear.available:
            summary["models_used"].append("couple_linear")
        if response.couple_nonlinear and response.couple_nonlinear.available:
            summary["models_used"].append("couple_nonlinear")

        return summary