import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


ALIAS_PRODUCTION = "production"
ALIAS_ARCHIVED = "archived"
ALIAS_STAGING = "staging"

TAG_LIFECYCLE_STATUS = "lifecycle_status"
TAG_DEPLOYMENT_DATE = "deployment_date"
TAG_PERFORMANCE_RMSE = "performance_rmse"
TAG_PERFORMANCE_R2 = "performance_r2"
TAG_TRAINING_DATE = "training_date"
TAG_GIT_COMMIT = "git_commit"
TAG_MODEL_TYPE = "model_type"
TAG_FAMILY = "family"
TAG_BEST_MODEL = "best_model"
TAG_REASON = "archive_reason"
TAG_RUN_ID = "run_id"
TAG_RUN_NAME = "run_name"
TAG_DATA_TYPE = "data_type"
TAG_N_CLIENTS = "n_clients"
TAG_N_FEATURES = "n_features"
TAG_ARTIFACT_PATH = "artifact_path"
TAG_DVC_HASH = "dvc_hash"
TAG_DVC_TRACKED = "dvc_tracked"


def compute_dvc_hash(file_path: Path) -> str:
    if not file_path.exists():
        return "file_not_found"

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_directory_dvc_hash(directory: Path) -> Dict[str, str]:
    if not directory.exists() or not directory.is_dir():
        return {}

    hashes: Dict[str, str] = {}
    for file_path in directory.glob("**/*"):
        if file_path.is_file() and not file_path.name.startswith(".") and not file_path.name.endswith(".tmp"):
            hashes[str(file_path.relative_to(directory))] = compute_dvc_hash(file_path)
    return hashes


def get_git_commit_hash(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "unknown"


def run_dvc_pull(path: Path, project_root: Path):
    if path.exists():
        print(f"[DVC] {path.name} already present.")
        return

    print(f"[DVC] attempting pull -> {path}")
    try:
        subprocess.run(["dvc", "pull", str(path)], cwd=project_root, check=True, capture_output=True, text=True)
        print("[DVC] pull successful")
    except subprocess.CalledProcessError as exc:
        print(f"[DVC] Pull failed: {exc.stderr.strip() or 'unknown error'}")


def save_dvc_hash_tracking(model_name: str, version: int, artifact_paths: List[Path], output_dir: Path, project_root: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_file = output_dir / f"dvc_hashes_{model_name}_v{version}_{timestamp}.json"

    tracking_info = {
        "model_name": model_name,
        "version": version,
        "timestamp": timestamp,
        "git_commit": get_git_commit_hash(project_root),
        "artifacts": {},
    }

    for artifact_path in artifact_paths:
        if artifact_path and artifact_path.exists():
            if artifact_path.is_file():
                tracking_info["artifacts"][str(artifact_path)] = {
                    "hash": compute_dvc_hash(artifact_path),
                    "size": artifact_path.stat().st_size,
                    "type": "file",
                }
            elif artifact_path.is_dir():
                tracking_info["artifacts"][str(artifact_path)] = {
                    "files": compute_directory_dvc_hash(artifact_path),
                    "total_files": len(list(artifact_path.glob("**/*"))),
                    "type": "directory",
                }

    with open(tracking_file, "w", encoding="utf-8") as file_handle:
        json.dump(tracking_info, file_handle, indent=2)

    print(f"   OK DVC hash tracking saved: {tracking_file}")
    return tracking_file


def clean_artifacts(models_dir: Path):
    print("Cleaning artifacts...")
    deleted = 0
    patterns = ["best_*", "global_model_*", "family_models_*", "tmp_*", "couple_models_*", "client_features_*"]

    for pattern in patterns:
        for item in models_dir.glob(pattern):
            try:
                if item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                else:
                    item.unlink()
                deleted += 1
            except Exception as exc:
                print(f"Failed to delete {item} -> {exc}")

    print(f"-> {deleted} item(s) deleted" if deleted else "-> nothing to clean")


def create_comparison_dataframe(results: List[Dict], model_type: str = "global") -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    comparison_data = []
    for result in results:
        row = {
            "Model": result.get("model_name", "Unknown"),
            "RMSE": round(result.get("rmse", 0), 4),
            "R2": round(result.get("r2", 0), 4),
            "MAE": round(result.get("mae", 0), 4),
            "Time (s)": round(result.get("training_time", 0), 2) if "training_time" in result else None,
        }
        if model_type == "linear" and "cv_r2" in result:
            row["CV R2"] = round(result["cv_r2"], 4)
            row["Formula"] = (result.get("formula", "")[:50] + "...") if result.get("formula") else ""
        elif model_type == "nonlinear" and "shap_success" in result:
            row["SHAP"] = "OK" if result.get("shap_success") else "NO"
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def create_comparison_plot(df: pd.DataFrame, output_path: Path, title: str = "Model Comparison"):
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df_sorted_rmse = df.sort_values("RMSE", ascending=True)
    axes[0].barh(df_sorted_rmse["Model"], df_sorted_rmse["RMSE"])
    axes[0].set_title(f"{title} - RMSE")
    axes[0].set_xlabel("RMSE")

    df_sorted_r2 = df.sort_values("R2", ascending=False)
    axes[1].barh(df_sorted_r2["Model"], df_sorted_r2["R2"])
    axes[1].set_title(f"{title} - R2")
    axes[1].set_xlabel("R2")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def set_model_version_tags(model_name: str, version: int, tags: Dict[str, str], client: Optional[MlflowClient] = None):
    mlflow_client = client or MlflowClient()
    for key, value in tags.items():
        mlflow_client.set_model_version_tag(model_name, version, key, str(value))


def get_model_version_tags(model_name: str, version: int, client: Optional[MlflowClient] = None) -> Dict[str, str]:
    mlflow_client = client or MlflowClient()
    return mlflow_client.get_model_version(model_name, str(version)).tags


def update_lifecycle_status(model_name: str, version: int, new_status: str, reason: str = "", client: Optional[MlflowClient] = None):
    tags = {
        TAG_LIFECYCLE_STATUS: new_status,
        TAG_DEPLOYMENT_DATE: datetime.now().isoformat(),
    }
    if reason:
        tags[TAG_REASON] = reason
    set_model_version_tags(model_name, version, tags, client=client)


def archive_version_with_tags(model_name: str, version: int, reason: str = "Old production version", client: Optional[MlflowClient] = None):
    update_lifecycle_status(model_name, version, ALIAS_ARCHIVED, reason=reason, client=client)


def set_model_alias(model_name: str, alias: str, version: int, client: Optional[MlflowClient] = None) -> bool:
    try:
        mlflow_client = client or MlflowClient()
        mlflow_client.set_registered_model_alias(model_name, alias, str(version))
        return True
    except Exception as exc:
        print(f"Alias assignment failed for {model_name}@{version}: {exc}")
        return False


def get_model_version_by_alias(model_name: str, alias: str, client: Optional[MlflowClient] = None) -> Optional[int]:
    try:
        mlflow_client = client or MlflowClient()
        model_version = mlflow_client.get_model_version_by_alias(model_name, alias)
        return int(model_version.version)
    except Exception:
        return None


def model_version_exists(model_name: str, version: int, client: Optional[MlflowClient] = None) -> bool:
    try:
        mlflow_client = client or MlflowClient()
        mlflow_client.get_model_version(model_name, str(version))
        return True
    except Exception:
        return False


def get_all_model_versions(model_name: str, client: Optional[MlflowClient] = None) -> List[Dict[str, Any]]:
    mlflow_client = client or MlflowClient()
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    return [
        {
            "name": version.name,
            "version": int(version.version),
            "run_id": version.run_id,
            "tags": dict(version.tags),
            "aliases": list(getattr(version, "aliases", [])),
        }
        for version in versions
    ]


def get_latest_model_version(model_name: str, client: Optional[MlflowClient] = None) -> Optional[int]:
    versions = get_all_model_versions(model_name, client=client)
    if not versions:
        return None
    return max(version_info["version"] for version_info in versions)


def set_production_alias(model_name: str, new_version: int, metrics: Optional[Dict] = None, client: Optional[MlflowClient] = None) -> bool:
    mlflow_client = client or MlflowClient()
    previous_version = get_model_version_by_alias(model_name, ALIAS_PRODUCTION, client=mlflow_client)
    if previous_version and previous_version != new_version:
        archive_version_with_tags(model_name, previous_version, client=mlflow_client)

    success = set_model_alias(model_name, ALIAS_PRODUCTION, new_version, client=mlflow_client)
    if success:
        tags = {
            TAG_LIFECYCLE_STATUS: ALIAS_PRODUCTION,
            TAG_DEPLOYMENT_DATE: datetime.now().isoformat(),
        }
        if metrics:
            if "rmse" in metrics:
                tags[TAG_PERFORMANCE_RMSE] = str(metrics["rmse"])
            if "r2" in metrics:
                tags[TAG_PERFORMANCE_R2] = str(metrics["r2"])
        set_model_version_tags(model_name, new_version, tags, client=mlflow_client)
    return success
