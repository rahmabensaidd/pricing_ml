from mlflow.tracking import MlflowClient
import logging
from typing import List, Dict, Optional
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Helper class for MLflow model registry operations"""

    def __init__(self):
        self.client = MlflowClient()

    def list_production_models(self) -> List[Dict]:
        """List all models with production alias"""
        try:
            registered_models = self.client.search_registered_models()
            production_models = []

            for model in registered_models:
                try:
                    mv = self.client.get_model_version_by_alias(model.name, settings.ALIAS_PRODUCTION)
                    if mv:
                        production_models.append({
                            "name": model.name,
                            "version": int(mv.version),
                            "alias": settings.ALIAS_PRODUCTION,
                            "tags": mv.tags if hasattr(mv, 'tags') else {}
                        })
                except Exception:
                    continue

            return production_models
        except Exception as e:
            logger.error(f"Error listing production models: {e}")
            return []

    def get_model_version_by_alias(self, model_name: str, alias: str = None) -> Optional[Dict]:
        """Get model version by alias"""
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
            logger.error(f"Error getting model version for {model_name}: {e}")
            return None