from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pricing__epac.src.api.schemas.pricing_models import PricingRequest
from pricing__epac.src.api.services.feature_service import FeatureService
from pricing__epac.src.api.services.mlflow_service import MLflowService
from pricing__epac.src.config.settings import Settings
from pricing__epac.src.machine_learning.preprocessing import full_preprocess
from pricing__epac.src.machine_learning.preprocessing.full_prepro import (
    cleanup_temp_files,
    get_project_root,
    save_processed,
)


def test_settings_ignore_extra_env_vars(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MLFLOW_TRACKING_URI=http://localhost:5000",
                "MYSQL_PASSWORD=root",
                "TRAIN_TEST_SPLIT=0.25",
                "LOG_LEVEL=INFO",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.MLFLOW_TRACKING_URI == "http://localhost:5000"


def test_get_project_root_points_to_repository():
    root = get_project_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "pricing__epac").exists()


def test_feature_service_preserves_integer_magnitudes_and_normalizes_categories():
    request = PricingRequest(
        siren="sav",
        binding_type="ss",
        features={
            "quantity": 500,
            "production_page": "252",
            "has_coil": 1,
            "text_color": "4/4",
            "cover_size": None,
        },
    )

    df, warnings = FeatureService.build_features_for_request(request)

    assert warnings == []
    assert int(df.loc[0, "quantity"]) == 500
    assert int(df.loc[0, "production_page"]) == 252
    assert int(df.loc[0, "has_coil"]) == 1
    assert df.loc[0, "siren"] == "SAV"
    assert df.loc[0, "binding_type"] == "SS"
    assert df.loc[0, "cover_size"] == "MISSING"


def test_transform_prediction_uses_log_inverse():
    log_prediction = np.log1p(42.5)

    assert FeatureService.transform_prediction(log_prediction) == np.expm1(log_prediction)


def test_save_processed_writes_under_package_data_root(tmp_path):
    df = pd.DataFrame({"quantity": [100], "unit_price": [12.5]})

    with patch(
        "pricing__epac.src.machine_learning.preprocessing.full_prepro.get_project_root",
        return_value=tmp_path,
    ):
        output_path = save_processed(df, "pricing_fully_cleaned.xlsx")

    expected = tmp_path / "pricing__epac" / "data" / "processed" / "pricing_fully_cleaned.xlsx"
    assert output_path == expected
    assert expected.exists()


def test_cleanup_temp_files_preserves_watch_folder(tmp_path):
    sql_folder = tmp_path / "pricing__epac" / "data" / "raw" / "dumps" / "sql"
    sql_folder.mkdir(parents=True)
    dump_file = sql_folder / "current_source.sql"
    dump_file.write_text("SELECT 1;", encoding="utf-8")

    with patch(
        "pricing__epac.src.machine_learning.preprocessing.full_prepro.get_project_root",
        return_value=tmp_path,
    ):
        cleanup_temp_files()

    assert sql_folder.exists()
    assert not dump_file.exists()


def test_legacy_full_preprocess_module_wraps_full_prepro():
    assert full_preprocess.full_preprocessing is not None
    assert callable(full_preprocess.uppercase_all_string_columns)
    assert callable(full_preprocess.replace_nat_nan_none)


def test_mlflow_service_uses_current_loader_api():
    with patch("pricing__epac.src.api.services.mlflow_service.ModelLoader") as loader_cls, patch(
        "pricing__epac.src.api.services.mlflow_service.ModelRegistry"
    ) as registry_cls, patch(
        "pricing__epac.src.api.services.mlflow_service.MlflowClient"
    ), patch("pricing__epac.src.api.services.mlflow_service.mlflow.set_tracking_uri"):
        loader = Mock()
        loader.get_production_model_info.return_value = {"name": "PricingModelGlobal", "metrics": {"rmse": 1.0}}
        loader.get_client_features_data.return_value = {"client_avg_price_ht": 12.0}
        loader_cls.return_value = loader

        registry = Mock()
        registry.list_production_models.return_value = [{"name": "PricingModelGlobal"}]
        registry_cls.return_value = registry

        service = MLflowService()

        assert service.get_client_features_data("SAV") == {"client_avg_price_ht": 12.0}
        assert service.get_model_by_name("PricingModelGlobal")["name"] == "PricingModelGlobal"
        assert service.get_model_metrics("PricingModelGlobal") == {"rmse": 1.0}
        assert service.list_available_models() == ["PricingModelGlobal"]

        loader.get_client_features_data.assert_called_once_with("SAV")
        loader.get_production_model_info.assert_any_call("PricingModelGlobal")
