import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import logging
import json
import pandas as pd
from typing import Optional, Dict, Any, List
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Model loader that fetches models fresh from MLflow on each request.
    No caching - always gets the latest production version.
    """

    def __init__(self):
        """Initialize MLflow client - no model loading at startup"""
        logger.info(f"🔧 Initializing ModelLoader with MLflow URI: {settings.MLFLOW_TRACKING_URI}")

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # Create client
        try:
            self.client = MlflowClient()
            logger.info("✅ MLflow client created successfully")

            # Test connection
            experiments = self.client.search_experiments(max_results=1)
            logger.info(f"✅ MLflow connection OK - found {len(experiments)} experiments")

        except Exception as e:
            logger.error(f"❌ Failed to create MLflow client: {e}")
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
            logger.error("❌ MLflow client not initialized")
            return None

        try:
            logger.info(f"🔍 Loading model '{model_name}' with alias '{settings.ALIAS_PRODUCTION}' from MLflow...")

            # Get model version by alias
            try:
                model_version = self.client.get_model_version_by_alias(
                    model_name, settings.ALIAS_PRODUCTION
                )

                if not model_version:
                    logger.warning(f"❌ No production alias found for {model_name}")

                    # Try to find any version as fallback
                    versions = self.client.search_model_versions(f"name='{model_name}'")
                    if versions:
                        logger.info(f"📦 Available versions for {model_name}:")
                        for v in versions:
                            aliases = v.aliases if hasattr(v, 'aliases') else []
                            logger.info(f"   v{v.version} - aliases: {aliases}")
                    return None

            except Exception as e:
                logger.error(f"❌ Error getting model version by alias: {e}")
                return None

            logger.info(f"✅ Found {model_name} v{model_version.version} with alias '{settings.ALIAS_PRODUCTION}'")

            # Load the model
            model_uri = f"models:/{model_name}@{settings.ALIAS_PRODUCTION}"
            logger.info(f"📦 Loading model from: {model_uri}")

            try:
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"✅ Model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load model from {model_uri}: {e}")
                return None

            # Get full version info for tags and metadata
            version_info = self.client.get_model_version(model_name, model_version.version)
            logger.info(f"ℹ️ Run ID: {version_info.run_id}")

            # Extract tags
            tags = version_info.tags if hasattr(version_info, 'tags') else {}
            logger.info(f"📌 Tags found: {list(tags.keys()) if tags else 'None'}")

            # Extract metadata
            metadata = {}
            if "metadata" in tags:
                try:
                    metadata = json.loads(tags["metadata"])
                    logger.info(f"📦 Metadata found with keys: {list(metadata.keys())}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not parse metadata: {e}")

            # Extract metrics
            metrics = self._extract_metrics(tags, metadata, model_name)
            logger.info(f"📊 Metrics extracted: {metrics}")

            # Extract additional information
            formula = self._extract_formula(tags, metadata)
            feature_importance = self._extract_feature_importance(tags, metadata)
            shap_available = self._extract_shap_success(tags)

            model_info = {
                "name": model_name,
                "version": int(model_version.version),
                "alias": settings.ALIAS_PRODUCTION,
                "model": model,
                "run_id": version_info.run_id,
                "tags": tags,
                "metadata": metadata,
                "description": getattr(version_info, 'description', ''),
                "metrics": metrics,
                "formula": formula,
                "feature_importance": feature_importance,
                "shap_available": shap_available
            }

            logger.info(f"✅ Production model {model_name} v{model_version.version} loaded successfully")
            return model_info

        except Exception as e:
            logger.error(f"❌ Unexpected error loading model {model_name}: {e}", exc_info=True)
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
        return self.get_production_model_info(model_name)

    def get_couple_model(self, binding_type: str, siren: str, model_type: str) -> Optional[Dict]:
        """
        Retrieves the couple model with the correct format: PricingModel_SS__SAV_Linear

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
        return self.get_production_model_info(model_name)

    def get_client_features_data(self, siren: str) -> Optional[Dict]:
        """
        Retrieves client features from MLflow ClientFeatures model

        Args:
            siren: Client SIREN

        Returns:
            Client features dictionary or None
        """
        try:
            model_info = self.get_production_model_info(settings.MODEL_NAME_CLIENT_FEATURES)

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

    def check_mlflow_connection(self) -> bool:
        """Check if MLflow tracking server is accessible"""
        try:
            if not self.client:
                return False
            self.client.search_experiments(max_results=1)
            return True
        except Exception:
            return False

    def _extract_metrics(self, tags: Dict, metadata: Dict, model_name: str) -> Dict:
        """Extract metrics from tags and metadata"""
        metrics = {}

        # Look for R² in tags
        r2_keys = ['r2_test', 'r2', 'test_r2', 'cv_r2', 'performance_r2', 'r2_score', 'R2', 'R²', 'r2_train']
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

        return metrics

    def _extract_formula(self, tags: Dict, metadata: Dict) -> Optional[str]:
        """Extract formula from tags or metadata"""
        if settings.TAG_FORMULA in tags:
            return tags[settings.TAG_FORMULA]
        if metadata and "formula" in metadata:
            return metadata["formula"]
        return None

    def _extract_feature_importance(self, tags: Dict, metadata: Dict) -> Dict:
        """Extract feature importance from tags or metadata"""
        if settings.TAG_FEATURE_IMPORTANCE in tags:
            try:
                return json.loads(tags[settings.TAG_FEATURE_IMPORTANCE])
            except:
                pass
        if metadata and "feature_importance" in metadata:
            return metadata["feature_importance"]
        return {}

    def _extract_shap_success(self, tags: Dict) -> bool:
        """Extract SHAP success status from tags"""
        return tags.get(settings.TAG_SHAP_SUCCESS, "false").lower() == "true"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Cleans a name for file / MLflow registry / model usage."""
        import re
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)