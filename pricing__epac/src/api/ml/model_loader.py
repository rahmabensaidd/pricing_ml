import json
import logging
from typing import Dict, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

from pricing__epac.src.config.settings import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Model loader that fetches models fresh from MLflow on each request.
    No caching - always gets the latest production version.
    """

    def __init__(self):
        """Initialize MLflow client without blocking on remote checks."""
        logger.info(f"Initializing ModelLoader with MLflow URI: {settings.MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        try:
            self.client = MlflowClient()
            logger.info("MLflow client created successfully")
        except Exception as e:
            logger.error(f"Failed to create MLflow client: {e}")
            self.client = None

    def get_production_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Retrieves the production model directly from MLflow at request time.
        No caching - always fetches fresh from MLflow.

        Args:
            model_name: Name of the model to load

        Returns:
            Model info dictionary or None if not found
        """
        if not self.client:
            logger.error("MLflow client not initialized")
            return None

        try:
            logger.info(f"Loading model '{model_name}' with alias '{settings.ALIAS_PRODUCTION}' from MLflow...")

            try:
                model_version = self.client.get_model_version_by_alias(
                    model_name, settings.ALIAS_PRODUCTION
                )

                if not model_version:
                    logger.warning(f"No production alias found for {model_name}")
                    versions = self.client.search_model_versions(f"name='{model_name}'")
                    if versions:
                        logger.info(f"Available versions for {model_name}:")
                        for version in versions:
                            aliases = version.aliases if hasattr(version, "aliases") else []
                            logger.info(f"   v{version.version} - aliases: {aliases}")
                    return None

            except Exception as e:
                logger.error(f"Error getting model version by alias: {e}")
                return None

            logger.info(
                f"Found {model_name} v{model_version.version} with alias '{settings.ALIAS_PRODUCTION}'"
            )

            model_uri = f"models:/{model_name}@{settings.ALIAS_PRODUCTION}"
            logger.info(f"Loading model from: {model_uri}")

            try:
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model from {model_uri}: {e}")
                return None

            version_info = self.client.get_model_version(model_name, model_version.version)
            logger.info(f"Run ID: {version_info.run_id}")

            tags = version_info.tags if hasattr(version_info, "tags") else {}
            logger.info(f"Tags found: {list(tags.keys()) if tags else 'None'}")

            metadata = {}
            if "metadata" in tags:
                try:
                    metadata = json.loads(tags["metadata"])
                    logger.info(f"Metadata found with keys: {list(metadata.keys())}")
                except Exception as e:
                    logger.warning(f"Could not parse metadata: {e}")

            metrics = self._extract_metrics(tags, metadata, model_name)
            formula = self._extract_formula(tags, metadata)
            feature_importance = self._extract_feature_importance(tags, metadata)
            shap_available = self._extract_shap_success(tags)
            algorithm = self._extract_algorithm(tags, metadata, model)

            return {
                "name": model_name,
                "version": int(model_version.version),
                "alias": settings.ALIAS_PRODUCTION,
                "model": model,
                "run_id": version_info.run_id,
                "tags": tags,
                "metadata": metadata,
                "description": getattr(version_info, "description", ""),
                "metrics": metrics,
                "formula": formula,
                "feature_importance": feature_importance,
                "shap_available": shap_available,
                "algorithm": algorithm,
            }

        except Exception as e:
            logger.error(f"Unexpected error loading model {model_name}: {e}", exc_info=True)
            return None

    def get_family_model(self, binding_type: str, model_type: str) -> Optional[Dict]:
        safe = self._sanitize_name(binding_type)
        model_name = f"PricingModel_{safe}_{model_type}"
        return self.get_production_model_info(model_name)

    def get_couple_model(self, binding_type: str, siren: str, model_type: str) -> Optional[Dict]:
        safe_binding = self._sanitize_name(binding_type)
        safe_siren = self._sanitize_name(siren)
        model_name = f"PricingModel_{safe_binding}__{safe_siren}_{model_type}"
        return self.get_production_model_info(model_name)

    def get_client_features_data(self, siren: str) -> Optional[Dict]:
        try:
            model_info = self.get_production_model_info(settings.MODEL_NAME_CLIENT_FEATURES)

            if not model_info:
                logger.warning("Client features model not available")
                return None

            client_model = model_info["model"]
            if hasattr(client_model, "get_client_features"):
                return client_model.get_client_features(siren)

            result = client_model.predict(pd.DataFrame({"siren": [siren]}))
            if isinstance(result, pd.DataFrame) and not result.empty:
                return result.iloc[0].to_dict()
            if isinstance(result, dict):
                return result
            if isinstance(result, list) and result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Error getting client features for siren {siren}: {e}")
            return None

    def check_mlflow_connection(self) -> bool:
        """Check if MLflow tracking server is accessible."""
        try:
            if not self.client:
                return False
            self.client.search_experiments(max_results=1)
            return True
        except Exception:
            return False

    def _extract_metrics(self, tags: Dict, metadata: Dict, model_name: str) -> Dict:
        metrics = {}
        r2_keys = ["r2_test", "r2", "test_r2", "cv_r2", "performance_r2", "r2_score", "R2", "RÂ²", "r2_train"]
        found_r2 = False

        for key in r2_keys:
            if key in tags:
                try:
                    metrics["r2"] = float(tags[key])
                    logger.info(f"Found R2={metrics['r2']} from tag '{key}' for {model_name}")
                    found_r2 = True
                    break
                except (ValueError, TypeError):
                    continue

        if not found_r2 and metadata:
            for key in r2_keys:
                if key in metadata:
                    try:
                        metrics["r2"] = float(metadata[key])
                        logger.info(f"Found R2={metrics['r2']} from metadata.{key} for {model_name}")
                        break
                    except (ValueError, TypeError):
                        continue

        for key, value in tags.items():
            if key.startswith("performance_"):
                metric_name = key.replace("performance_", "")
                try:
                    metrics[metric_name] = float(value)
                except (ValueError, TypeError):
                    continue
            elif key in ["rmse", "mae", "mape", "cv_r2"] and key not in metrics:
                try:
                    metrics[key] = float(value)
                except (ValueError, TypeError):
                    continue

        return metrics

    def _extract_formula(self, tags: Dict, metadata: Dict) -> Optional[str]:
        if settings.TAG_FORMULA in tags:
            return tags[settings.TAG_FORMULA]
        if metadata and "formula" in metadata:
            return metadata["formula"]
        return None

    def _extract_feature_importance(self, tags: Dict, metadata: Dict) -> Dict:
        if settings.TAG_FEATURE_IMPORTANCE in tags:
            try:
                return json.loads(tags[settings.TAG_FEATURE_IMPORTANCE])
            except Exception:
                pass
        if metadata and "feature_importance" in metadata:
            return metadata["feature_importance"]
        return {}

    def _extract_shap_success(self, tags: Dict) -> bool:
        return tags.get(settings.TAG_SHAP_SUCCESS, "false").lower() == "true"

    def _extract_algorithm(self, tags: Dict, metadata: Dict, model) -> Optional[str]:
        for key in ("best_model", "algorithm", "model_algorithm"):
            value = tags.get(key)
            if value:
                return str(value)

        if metadata:
            for key in ("best_model", "algorithm", "model_algorithm"):
                value = metadata.get(key)
                if value:
                    return str(value)

        try:
            unwrap = getattr(model, "unwrap_python_model", None)
            if callable(unwrap):
                py_model = unwrap()
                if py_model is not None:
                    class_name = py_model.__class__.__name__
                    if class_name:
                        return class_name
        except Exception:
            pass

        class_name = model.__class__.__name__ if model is not None else ""
        if class_name:
            return class_name
        return None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        import re

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)
