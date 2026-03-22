import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import logging
import json
from typing import Optional, Dict, List
from pricing__epac.src.config.settings import settings
from pricing__epac.src.api.ml.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class MLflowService:
    """Service for MLflow interactions - handles all MLflow operations"""

    def __init__(self):
        self.model_loader = ModelLoader()
        self.client = MlflowClient()
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def get_client_features_data(self, siren: str) -> Optional[Dict]:
        """
        Retrieves client features from MLflow ClientFeatures model
        """
        try:
            model_info = self.model_loader.get_model(settings.MODEL_NAME_CLIENT_FEATURES)

            if not model_info:
                logger.warning("Client features model not available")
                return None

            client_model = model_info["model"]

            # Try different methods to get client features
            if hasattr(client_model, 'get_client_features'):
                # Custom method if available
                return client_model.get_client_features(siren)
            else:
                # Standard prediction approach
                result = client_model.predict(pd.DataFrame({"siren": [siren]}))
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result.iloc[0].to_dict()
                elif isinstance(result, dict):
                    return result
                elif isinstance(result, list) and len(result) > 0:
                    return result[0]

            return None
        except Exception as e:
            logger.error(f"Error getting client features for siren {siren}: {e}")
            return None

    def get_family_model(self, binding_type: str, model_type: str) -> Optional[Dict]:
        """
        Get family model by binding type and model type (Linear/NonLinear)

        Args:
            binding_type: Type of binding (e.g., "SS", "CASEBIND")
            model_type: "Linear" or "NonLinear"

        Returns:
            Model info dictionary or None if not found
        """
        safe = self._sanitize_name(binding_type)
        model_name = f"PricingModel_{safe}_{model_type}"
        return self.model_loader.get_model(model_name)

    def get_couple_model(self, binding_type: str, siren: str, model_type: str) -> Optional[Dict]:
        """
        Retrieves the couple model with the correct format: PricingModel_SS__SAV_Linear

        Note: double underscore between binding_type and siren

        Args:
            binding_type: Type of binding (e.g., "SS", "CASEBIND")
            siren: Client SIREN (e.g., "SAV")
            model_type: "Linear" or "NonLinear"

        Returns:
            Model info dictionary or None if not found
        """
        safe_binding = self._sanitize_name(binding_type)
        safe_siren = self._sanitize_name(siren)
        model_name = f"PricingModel_{safe_binding}__{safe_siren}_{model_type}"
        return self.model_loader.get_model(model_name)

    def get_model_by_name(self, model_name: str) -> Optional[Dict]:
        """
        Get any model by its exact name

        Args:
            model_name: Full model name as registered in MLflow

        Returns:
            Model info dictionary or None if not found
        """
        return self.model_loader.get_model(model_name)

    def list_available_models(self) -> List[str]:
        """
        List all models currently loaded in memory

        Returns:
            List of model names
        """
        return list(self.model_loader._models.keys())

    def get_model_metrics(self, model_name: str) -> Dict:
        """
        Get metrics for a specific model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of metrics
        """
        model_info = self.model_loader.get_model(model_name)
        if model_info:
            return model_info.get("metrics", {})
        return {}

    def get_model_feature_importance(self, model_name: str) -> Dict[str, float]:
        """
        Get feature importance for a specific model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of feature importance
        """
        model_info = self.model_loader.get_model(model_name)
        if model_info:
            return model_info.get("feature_importance", {})
        return {}

    def check_mlflow_connection(self) -> bool:
        """
        Check if MLflow tracking server is accessible

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.search_experiments(max_results=1)
            return True
        except Exception as e:
            logger.error(f"MLflow connection failed: {e}")
            return False

    def get_experiments_count(self) -> int:
        """
        Get number of experiments in MLflow

        Returns:
            Number of experiments or 0 if error
        """
        try:
            return len(self.client.search_experiments())
        except Exception as e:
            logger.error(f"Error getting experiments count: {e}")
            return 0

    def get_registered_models(self) -> List[Dict]:
        """
        Get all registered models from MLflow

        Returns:
            List of registered models with their details
        """
        try:
            registered_models = self.client.search_registered_models()
            models_list = []

            for model in registered_models:
                models_list.append({
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id
                        }
                        for v in model.latest_versions
                    ]
                })

            return models_list
        except Exception as e:
            logger.error(f"Error getting registered models: {e}")
            return []

    def get_model_version_by_alias(self, model_name: str, alias: str = None) -> Optional[Dict]:
        """
        Get model version by alias

        Args:
            model_name: Name of the model
            alias: Alias to look for (defaults to production alias from settings)

        Returns:
            Model version info or None
        """
        try:
            alias_to_use = alias or settings.ALIAS_PRODUCTION
            mv = self.client.get_model_version_by_alias(model_name, alias_to_use)

            if mv:
                return {
                    "name": model_name,
                    "version": int(mv.version),
                    "alias": alias_to_use,
                    "run_id": mv.run_id,
                    "status": mv.status,
                    "tags": mv.tags if hasattr(mv, 'tags') else {}
                }
            return None
        except Exception as e:
            logger.error(f"Error getting model version for {model_name} with alias {alias_to_use}: {e}")
            return None

    def get_model_run_info(self, model_name: str) -> Optional[Dict]:
        """
        Get run information for a model

        Args:
            model_name: Name of the model

        Returns:
            Run information dictionary or None
        """
        try:
            model_info = self.model_loader.get_model(model_name)
            if model_info and model_info.get("run_id"):
                run = self.client.get_run(model_info["run_id"])
                return {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                    "tags": run.data.tags
                }
            return None
        except Exception as e:
            logger.error(f"Error getting run info for {model_name}: {e}")
            return None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Cleans a name for file / MLflow registry / model usage.
        Replaces all non-alphanumeric characters with underscore

        Args:
            name: Raw name to sanitize

        Returns:
            Sanitized name
        """
        import re
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def extract_metrics_from_tags(self, tags: Dict) -> Dict:
        """
        Extract metrics from MLflow tags

        Args:
            tags: Dictionary of tags from MLflow

        Returns:
            Dictionary of metrics
        """
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

        return metrics

    def extract_formula_from_tags(self, tags: Dict) -> Optional[str]:
        """Extracts formula from MLflow tags"""
        return tags.get(settings.TAG_FORMULA, None)

    def extract_feature_importance_from_tags(self, tags: Dict) -> Dict[str, float]:
        """Extracts feature importance from MLflow tags"""
        if settings.TAG_FEATURE_IMPORTANCE in tags:
            try:
                return json.loads(tags[settings.TAG_FEATURE_IMPORTANCE])
            except:
                return {}
        return {}

    def extract_shap_success_from_tags(self, tags: Dict) -> bool:
        """Extracts SHAP status from MLflow tags"""
        return tags.get(settings.TAG_SHAP_SUCCESS, "false").lower() == "true"