"""Global training public facade.

Readable API for orchestration and jury presentation.
Heavy implementation lives in `_global_core/global_training_impl.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from pricing__epac.src.machine_learning.training._global_core.global_training_impl import (
    DATA_DIR,
    MODELS_DIR,
    TrainingConfig,
    generate_prediction_examples,
    train_and_compare,
)


def train_global_models(
    file_path: Optional[Path] = None,
    register_to_mlflow: bool = True,
    mlflow_run: Optional[Any] = None,
) -> Tuple[str, List[Dict], Pipeline, pd.DataFrame, pd.Series, Pipeline, Dict[str, float], Optional[str]]:
    """Primary entrypoint used by pipelines for global model training."""
    return train_and_compare(
        file_path=file_path,
        register_to_mlflow=register_to_mlflow,
        mlflow_run=mlflow_run,
    )


__all__ = [
    "TrainingConfig",
    "train_global_models",
    "train_and_compare",
    "generate_prediction_examples",
    "MODELS_DIR",
    "DATA_DIR",
]
