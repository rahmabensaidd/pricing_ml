"""BindingType + SIREN training public facade.

Readable public API for orchestration.
Heavy implementation details live in
`_bindingtype_siren_core/bindingtype_siren_training_impl.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pricing__epac.src.machine_learning.training._bindingtype_siren_core.bindingtype_siren_training_impl import (
    display_couple_results_regularized,
    run_couple_analysis_regularized,
    train_by_bindingtype_siren,
    train_by_bindingtype_siren_regularized,
)


def train_bindingtype_siren_models(
    file_path: str | Path | None = None,
    save_pipelines: bool = False,
    output_dir: Path | None = None,
    register_to_mlflow: bool = True,
    mlflow_run: Optional[Any] = None,
) -> Dict[str, Any]:
    """Primary entrypoint used by pipelines for binding_type+siren training."""
    return train_by_bindingtype_siren_regularized(
        file_path=file_path,
        save_pipelines=save_pipelines,
        output_dir=output_dir,
        register_to_mlflow=register_to_mlflow,
        mlflow_run=mlflow_run,
    )


__all__ = [
    "train_bindingtype_siren_models",
    "train_by_bindingtype_siren",
    "train_by_bindingtype_siren_regularized",
    "run_couple_analysis_regularized",
    "display_couple_results_regularized",
]
