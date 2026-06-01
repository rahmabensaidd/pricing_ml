"""BindingType training public facade.

This module is intentionally readable for navigation and jury presentation.
Heavy implementation details live in `_bindingtype_core/bindingtype_training_impl.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pricing__epac.src.machine_learning.training._bindingtype_core.bindingtype_training_impl import (
    display_family_results_regularized,
    run_family_analysis_regularized,
    train_by_bindingtype_regularized,
)


def train_bindingtype_models(
    file_path: str | Path | None = None,
    save_pipelines: bool = False,
    output_dir: Path | None = None,
    register_to_mlflow: bool = True,
    mlflow_run: Optional[Any] = None,
) -> Dict[str, Any]:
    """Primary entrypoint used by pipelines for binding_type model training."""
    return train_by_bindingtype_regularized(
        file_path=file_path,
        save_pipelines=save_pipelines,
        output_dir=output_dir,
        register_to_mlflow=register_to_mlflow,
        mlflow_run=mlflow_run,
    )


__all__ = [
    "train_bindingtype_models",
    "train_by_bindingtype_regularized",
    "run_family_analysis_regularized",
    "display_family_results_regularized",
]
