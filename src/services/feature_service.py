import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.config.feature_config import (
    ALL_FEATURES, INT_COLS, FLOAT_COLS,
    KNOWN_CATEGORIES, FALLBACK_CATEGORY
)
from src.api.models.pricing_models import PricingRequest

logger = logging.getLogger(__name__)


class FeatureService:
    """Handles feature engineering and data preprocessing"""

    @staticmethod
    def build_features_for_request(request: PricingRequest) -> Tuple[pd.DataFrame, List[str]]:
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
                    val, warn = FeatureService._process_int_feature(feature, value)
                    data[feature] = val
                    if warn:
                        warnings.append(warn)

                elif feature in FLOAT_COLS:
                    val, warn = FeatureService._process_float_feature(feature, value)
                    data[feature] = val
                    if warn:
                        warnings.append(warn)

                else:  # Categorical
                    data[feature] = FeatureService._safe_categorical_value(value, feature)

            elif feature == "siren" and request.siren:
                data[feature] = FeatureService._safe_categorical_value(request.siren, feature)

            elif feature == "binding_type" and request.binding_type:
                data[feature] = FeatureService._safe_categorical_value(request.binding_type, feature)

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

        if warnings:
            logger.warning(f"Conversion warnings: {warnings}")

        return df, warnings

    @staticmethod
    def _process_int_feature(feature: str, value: Any) -> Tuple[int, Optional[str]]:
        """Process integer/boolean feature"""
        try:
            if isinstance(value, bool):
                return (1 if value else 0, None)
            elif isinstance(value, (int, float)):
                return (1 if float(value) != 0 else 0, None)
            elif isinstance(value, str):
                str_val = value.lower().strip()
                if str_val in ['true', '1', 'yes', 'oui', 'vrai']:
                    return (1, None)
                else:
                    return (0, None)
            else:
                return (0, f"Could not convert {feature}={value} to int, using 0")
        except (ValueError, TypeError):
            return (0, f"Could not convert {feature}={value} to int, using 0")

    @staticmethod
    def _process_float_feature(feature: str, value: Any) -> Tuple[float, Optional[str]]:
        """Process float feature"""
        try:
            return (float(value), None)
        except (ValueError, TypeError):
            return (0.0, f"Could not convert {feature}={value} to float, using 0.0")

    @staticmethod
    def _safe_categorical_value(value: Any, column: str) -> str:
        """Converts a categorical value to a value known by the model"""
        if pd.isna(value) or value is None:
            return FALLBACK_CATEGORY

        str_value = str(value).strip()

        if column in KNOWN_CATEGORIES:
            if str_value in KNOWN_CATEGORIES[column]:
                return str_value
            else:
                logger.warning(
                    f"Unknown category '{str_value}' for column {column}, using fallback '{FALLBACK_CATEGORY}'")
                return FALLBACK_CATEGORY

        return str_value

# Ajoutez cette méthode à votre FeatureService existant

    @staticmethod
    def transform_prediction(prediction: float) -> float:
        """Transform prediction if needed (log transformation)"""
        if prediction < 0 or prediction < 100:  # If it's small, it's probably in log
            return float(np.expm1(prediction))
        return float(prediction)