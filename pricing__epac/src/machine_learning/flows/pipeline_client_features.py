import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from joblib import dump
from mlflow.pyfunc import PythonModel
from prefect import task

from pricing__epac.src.machine_learning.training.client_history_features import (
    add_client_features_to_orders,
    create_client_features,
    save_client_features,
)
from pricing__epac.src.machine_learning.flows.pipeline_support import (
    ALIAS_PRODUCTION,
    TAG_ARTIFACT_PATH,
    TAG_DATA_TYPE,
    TAG_DVC_TRACKED,
    TAG_GIT_COMMIT,
    TAG_LIFECYCLE_STATUS,
    TAG_N_CLIENTS,
    TAG_N_FEATURES,
    TAG_RUN_ID,
    TAG_RUN_NAME,
    TAG_TRAINING_DATE,
    compute_dvc_hash,
    get_git_commit_hash,
    get_latest_model_version,
    save_dvc_hash_tracking,
    set_model_version_tags,
    set_production_alias,
)


class ClientFeaturesWrapper(PythonModel):
    def load_context(self, context):
        self.features = pd.read_csv(context.artifacts["features_csv"])
        self.metadata = json.loads(Path(context.artifacts["summary_json"]).read_text(encoding="utf-8"))

    def predict(self, context, model_input):
        if self.features is None:
            return pd.DataFrame({"error": ["Features not loaded"]})

        if isinstance(model_input, (str, int)):
            if "siren" in self.features.columns:
                mask = self.features["siren"].astype(str) == str(model_input)
                if mask.any():
                    result = self.features[mask].to_dict("records")[0]
                    return pd.DataFrame([result])
                return pd.DataFrame({"siren": [model_input], "found": [False]})

        if isinstance(model_input, pd.DataFrame) and "siren" in model_input.columns:
            result = model_input[["siren"]].copy()
            return result.merge(self.features, on="siren", how="left")

        return self.features


@task(name="MLflow Logging Client Features")
def mlflow_log_client_features(
    *,
    client_features: pd.DataFrame,
    cleaned_file: Path,
    model_name: str,
    dvc_tracking_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("MLFLOW LOGGING CLIENT FEATURES")
    print("=" * 60)

    mlflow.set_experiment("Pricing_Client_Features")
    run_name = f"client-features-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dvc_tracking_file = None

    with mlflow.start_run(run_name=run_name) as run:
        input_dvc_hash = compute_dvc_hash(cleaned_file)
        mlflow.set_tag("data_type", "client_features")
        mlflow.set_tag("git_commit", get_git_commit_hash(project_root))
        mlflow.set_tag("run_name", run_name)
        mlflow.set_tag("source_file", str(cleaned_file))
        mlflow.set_tag("input_data_dvc_hash", input_dvc_hash)

        mlflow.log_param("n_clients", len(client_features))
        mlflow.log_param("n_features", len(client_features.columns) - 1)
        mlflow.log_param("min_samples_elasticity", 8)
        mlflow.log_param("min_price_cv", 0.05)
        mlflow.log_param("input_data_dvc_hash", input_dvc_hash)

        if "client_price_elasticity" in client_features.columns:
            mlflow.log_metric("elasticity_mean", float(client_features["client_price_elasticity"].mean()))
            mlflow.log_metric("elasticity_std", float(client_features["client_price_elasticity"].std()))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            features_csv = tmp_path / "client_features.csv"
            features_excel = tmp_path / "client_features.xlsx"
            features_joblib = tmp_path / "client_features.joblib"
            summary_json = tmp_path / "client_features_summary.json"

            client_features.to_csv(features_csv, index=False)
            client_features.to_excel(features_excel, index=False)
            dump(client_features, features_joblib)

            csv_hash = compute_dvc_hash(features_csv)
            excel_hash = compute_dvc_hash(features_excel)
            joblib_hash = compute_dvc_hash(features_joblib)

            summary = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run.info.run_id,
                "run_name": run_name,
                "experiment_id": run.info.experiment_id,
                "n_clients": len(client_features),
                "n_features": len(client_features.columns) - 1,
                "columns": list(client_features.columns),
                "source_file": str(cleaned_file),
                "source_file_dvc_hash": input_dvc_hash,
                "dvc_hashes": {
                    "csv": csv_hash,
                    "excel": excel_hash,
                    "joblib": joblib_hash,
                },
            }
            summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            mlflow.log_artifact(str(features_csv), "features")
            mlflow.log_artifact(str(features_excel), "features")
            mlflow.log_artifact(str(features_joblib), "features")
            mlflow.log_artifact(str(summary_json), "features")

            model_info = mlflow.pyfunc.log_model(
                name="client_features_model",
                python_model=ClientFeaturesWrapper(),
                artifacts={
                    "features_csv": str(features_csv),
                    "features_excel": str(features_excel),
                    "features_joblib": str(features_joblib),
                    "summary_json": str(summary_json),
                },
                registered_model_name=model_name,
                metadata=summary,
            )

            version = getattr(model_info, "registered_model_version", None)
            if version:
                dvc_tracking_file = save_dvc_hash_tracking(
                    model_name,
                    int(version),
                    [features_csv, features_excel, features_joblib, summary_json],
                    dvc_tracking_dir,
                    project_root,
                )
                tags = {
                    TAG_LIFECYCLE_STATUS: "new",
                    TAG_TRAINING_DATE: datetime.now().isoformat(),
                    TAG_GIT_COMMIT: get_git_commit_hash(project_root),
                    TAG_DATA_TYPE: "client_features",
                    TAG_N_CLIENTS: str(len(client_features)),
                    TAG_N_FEATURES: str(len(client_features.columns) - 1),
                    TAG_RUN_ID: run.info.run_id,
                    TAG_RUN_NAME: run_name,
                    TAG_ARTIFACT_PATH: f"{run.info.artifact_uri}/client_features_model",
                    TAG_DVC_TRACKED: "true",
                }
                set_model_version_tags(model_name, int(version), tags)

                latest_version = get_latest_model_version(model_name)
                if latest_version and int(version) == latest_version:
                    set_production_alias(model_name, int(version), {"n_clients": len(client_features)})

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run_name,
            "n_clients": len(client_features),
            "n_features": len(client_features.columns) - 1,
            "model_name": model_name,
            "model_version": int(version) if version else None,
            "artifact_uri": f"{run.info.artifact_uri}/client_features_model",
            "artifacts": {
                "csv": f"{run.info.artifact_uri}/features/client_features.csv",
                "excel": f"{run.info.artifact_uri}/features/client_features.xlsx",
                "joblib": f"{run.info.artifact_uri}/features/client_features.joblib",
                "summary": f"{run.info.artifact_uri}/features/client_features_summary.json",
            },
            "dvc_hashes": {
                "input_data": input_dvc_hash,
                "csv": csv_hash,
                "excel": excel_hash,
                "joblib": joblib_hash,
            },
            "dvc_tracking_file": str(dvc_tracking_file) if dvc_tracking_file else None,
        }


@task(name="Create Client Features", retries=1)
def create_client_features_task(
    *,
    cleaned_file: Path,
    client_features_file: Path,
    enriched_data_file: Path,
    mlflow_model_name: str,
    dvc_tracking_dir: Path,
    project_root: Path,
) -> Tuple[Path, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    print("\n" + "=" * 60)
    print("CREATING CLIENT FEATURES")
    print("=" * 60)

    df = pd.read_excel(cleaned_file)
    initial_shape = df.shape
    df = df[df["unit_price"] != 0].copy()
    print(f"Data after price=0 filter: {df.shape} (was {initial_shape})")

    if "siren" not in df.columns:
        return cleaned_file, pd.DataFrame(), df, {"error": "missing_siren"}

    client_features = create_client_features(df, min_samples_elasticity=8, min_price_cv=0.05)
    if client_features.empty:
        return cleaned_file, pd.DataFrame(), df, {"error": "empty_features"}

    mlflow_result = mlflow_log_client_features.fn(
        client_features=client_features,
        cleaned_file=cleaned_file,
        model_name=mlflow_model_name,
        dvc_tracking_dir=dvc_tracking_dir,
        project_root=project_root,
    )

    save_client_features(client_features, client_features_file)
    enriched_df = add_client_features_to_orders(df, client_features)
    enriched_data_file.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_excel(enriched_data_file, index=False)

    return enriched_data_file, client_features, enriched_df, mlflow_result


@task(name="Validate Client Features")
def validate_client_features(client_features: pd.DataFrame) -> Dict[str, Any]:
    if client_features.empty:
        return {"error": "empty_dataframe"}

    stats = {
        "n_clients": len(client_features),
        "n_features": len(client_features.columns) - 1,
        "missing_values": int(client_features.isna().sum().sum()),
    }

    if "client_price_elasticity" in client_features.columns:
        stats.update(
            {
                "clients_with_elasticity": int((client_features["client_price_elasticity"] != 0).sum()),
                "elasticity_mean": float(client_features["client_price_elasticity"].mean()),
                "elasticity_std": float(client_features["client_price_elasticity"].std()),
                "elasticity_min": float(client_features["client_price_elasticity"].min()),
                "elasticity_max": float(client_features["client_price_elasticity"].max()),
            }
        )

    if "client_seniority_years" in client_features.columns:
        stats["seniority_mean"] = float(client_features["client_seniority_years"].mean())

    if "client_recency_days" in client_features.columns:
        stats["recency_mean"] = float(client_features["client_recency_days"].mean())

    if "client_avg_price_ht" in client_features.columns:
        stats["avg_price_mean"] = float(client_features["client_avg_price_ht"].mean())

    return stats
