from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pricing__epac.src.machine_learning.pipelines.pipeline_client_features import mlflow_log_client_features
from pricing__epac.src.machine_learning.ingestion.watcher import SQLFileHandler
from pricing__epac.src.machine_learning.training.client_features_training import aggregate_client_features


def test_aggregate_client_features_handles_mixed_reception_date_types():
    df = pd.DataFrame(
        {
            "siren": ["A", "A", "B"],
            "unit_price_ht": [10.0, 12.0, 20.0],
            "quantity": [100, 120, 90],
            "price_volatility": [1.0, 1.2, 0.5],
            "relative_price_index": [1.0, 1.1, 0.9],
            "reception_date": ["2026-01-02", np.nan, "2026-03-14"],
        }
    )

    aggregated = aggregate_client_features(df)

    assert not aggregated.empty
    assert "client_first_order" in aggregated.columns
    assert aggregated.loc[aggregated["siren"] == "A", "client_first_order"].iloc[0] == pd.Timestamp("2026-01-02")


def test_watcher_deduplicates_same_sql_file_path(tmp_path):
    sql_file = tmp_path / "sample.sql"
    sql_file.write_text("SELECT 1;", encoding="utf-8")

    with patch.object(SQLFileHandler, "start_processing_thread", return_value=None):
        handler = SQLFileHandler()
    handler._validate_sql_file = Mock(return_value=True)
    handler._get_file_hash = Mock(return_value="same-hash")

    handler._handle_new_sql_file(str(sql_file))
    handler._handle_new_sql_file(str(sql_file))

    assert handler.pending_queue.qsize() == 1


def test_watcher_normalizes_prefixed_timestamps():
    line = "2026-04-21 14:06:38 - HTTP Request: POST http://127.0.0.1:8016/api/logs/"

    normalized = SQLFileHandler._normalize_output_line(line)

    assert normalized == 'HTTP Request: POST http://127.0.0.1:8016/api/logs/'


def test_mlflow_log_client_features_uses_current_mlflow_api(tmp_path):
    cleaned_file = tmp_path / "cleaned.xlsx"
    cleaned_file.write_text("placeholder", encoding="utf-8")
    client_features = pd.DataFrame(
        {
            "siren": ["A", "B"],
            "client_price_elasticity": [0.2, 0.4],
            "client_avg_price_ht": [10.0, 12.0],
        }
    )

    class DummyRunContext:
        def __init__(self, run_obj):
            self.run_obj = run_obj

        def __enter__(self):
            return self.run_obj

        def __exit__(self, exc_type, exc, tb):
            return False

    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="run-123",
            experiment_id="exp-456",
            artifact_uri="s3://mlflow-artifacts/run-123",
        )
    )
    run_context = DummyRunContext(run)
    logged_model = SimpleNamespace(registered_model_version="7")

    with patch("pricing__epac.src.machine_learning.pipelines.pipeline_client_features.mlflow.set_experiment"), patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.mlflow.start_run",
        return_value=run_context,
    ), patch("pricing__epac.src.machine_learning.pipelines.pipeline_client_features.mlflow.set_tag"), patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.mlflow.log_param"
    ), patch("pricing__epac.src.machine_learning.pipelines.pipeline_client_features.mlflow.log_metric"), patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.mlflow.log_artifact"
    ), patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.mlflow.pyfunc.log_model",
        return_value=logged_model,
    ) as log_model, patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.save_dvc_hash_tracking",
        return_value=tmp_path / "tracking.json",
    ), patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.set_model_version_tags"
    ), patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.get_latest_model_version",
        return_value=7,
    ), patch(
        "pricing__epac.src.machine_learning.pipelines.pipeline_client_features.set_production_alias"
    ):
        result = mlflow_log_client_features.fn(
            client_features=client_features,
            cleaned_file=cleaned_file,
            model_name="ClientFeatures",
            dvc_tracking_dir=tmp_path,
            project_root=tmp_path,
        )

    assert result["model_version"] == 7
    assert log_model.call_args.kwargs["name"] == "client_features_model"
    assert "artifact_path" not in log_model.call_args.kwargs


