import os
import sys

from pricing__epac.src.machine_learning.ingestion.watcher import SQL_FOLDER
from pricing__epac.src.machine_learning.training.client_features_training import save_client_features, \
    add_client_features_to_orders, create_client_features

# Add root path to find openssl_patch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from prefect import flow, task
import subprocess
import hashlib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PythonModel
import matplotlib
from mlflow.models import infer_signature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
import numpy as np
import tempfile
import json
from datetime import datetime
import sys
import warnings
import traceback
from typing import Optional, Dict, Any, List, Tuple

warnings.filterwarnings('ignore')
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump

from pricing__epac.src.machine_learning.ingestion.consolidate_data import run_consolidation
from pricing__epac.src.machine_learning.preprocessing.full_prepro import full_preprocessing, save_processed
from pricing__epac.src.machine_learning.training.global_training import train_and_compare
from pricing__epac.src.machine_learning.training.bindingtype_training import train_by_bindingtype_regularized as train_by_bindingtype
from pricing__epac.src.machine_learning.training.bindingtype_siren_training import train_by_bindingtype_siren
from pricing__epac.src.config.settings import settings
from pricing__epac.src.machine_learning.pipelines.pipeline_client_features import (
    create_client_features_task as pipeline_create_client_features_task,
    validate_client_features as pipeline_validate_client_features,
)
from pricing__epac.src.machine_learning.pipelines.pipeline_preprocessing import (
    consolidate_data_task as pipeline_consolidate_data_task,
    run_preprocessing as pipeline_run_preprocessing,
)
from pricing__epac.src.machine_learning.pipelines.pipeline_support import (
    ALIAS_ARCHIVED,
    ALIAS_PRODUCTION,
    ALIAS_STAGING,
    TAG_ARTIFACT_PATH,
    TAG_BEST_MODEL,
    TAG_DATA_TYPE,
    TAG_DEPLOYMENT_DATE,
    TAG_DVC_HASH,
    TAG_DVC_TRACKED,
    TAG_FAMILY,
    TAG_GIT_COMMIT,
    TAG_LIFECYCLE_STATUS,
    TAG_MODEL_TYPE,
    TAG_N_CLIENTS,
    TAG_N_FEATURES,
    TAG_PERFORMANCE_R2,
    TAG_PERFORMANCE_RMSE,
    TAG_REASON,
    TAG_RUN_ID,
    TAG_RUN_NAME,
    TAG_TRAINING_DATE,
    archive_version_with_tags,
    clean_artifacts,
    compute_dvc_hash,
    create_comparison_dataframe,
    create_comparison_plot,
    get_all_model_versions,
    get_git_commit_hash,
    get_latest_model_version,
    get_model_version_by_alias,
    get_model_version_tags,
    model_version_exists,
    run_dvc_pull,
    save_dvc_hash_tracking,
    set_model_alias,
    set_model_version_tags,
    set_production_alias,
    update_lifecycle_status,
)

# Configuration for MinIO (S3 compatible)
import warnings

# Disable pyOpenSSL in urllib3 (even if not installed)
os.environ['URLLIB3_USE_PYOPENSSL'] = '0'
warnings.filterwarnings('ignore', module='urllib3.contrib.pyopenssl')
os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MLFLOW_S3_ENDPOINT_URL
os.environ['AWS_DEFAULT_REGION'] = settings.AWS_DEFAULT_REGION
# ────────────────────────────────────────────────
# Configuration for client features
# ────────────────────────────────────────────────

# ────────────────────────────────────────────────
# DVC Hash utilities
# ────────────────────────────────────────────────
def compute_dvc_hash(file_path: Path) -> str:
    """
    Computes DVC-style MD5 hash for a file
    Used to track model versions with DVC
    """
    if not file_path.exists():
        return "file_not_found"

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def compute_directory_dvc_hash(directory: Path) -> Dict[str, str]:
    """
    Computes DVC-style MD5 hashes for all files in a directory
    Returns a dictionary of {file_path: hash}
    """
    if not directory.exists() or not directory.is_dir():
        return {}

    hashes = {}
    for file_path in directory.glob("**/*"):
        if file_path.is_file():
            # Skip hidden files and temporary files
            if not file_path.name.startswith('.') and not file_path.name.endswith('.tmp'):
                rel_path = file_path.relative_to(directory)
                hashes[str(rel_path)] = compute_dvc_hash(file_path)

    return hashes


def save_dvc_hash_tracking(model_name: str, version: int, artifact_paths: List[Path], output_dir: Path) -> Path:
    """
    Saves DVC hash tracking information for model artifacts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_file = output_dir / f"dvc_hashes_{model_name}_v{version}_{timestamp}.json"

    tracking_info = {
        "model_name": model_name,
        "version": version,
        "timestamp": timestamp,
        "git_commit": get_git_commit_hash(),
        "artifacts": {}
    }

    for artifact_path in artifact_paths:
        if artifact_path and artifact_path.exists():
            if artifact_path.is_file():
                tracking_info["artifacts"][str(artifact_path)] = {
                    "hash": compute_dvc_hash(artifact_path),
                    "size": artifact_path.stat().st_size,
                    "type": "file"
                }
            elif artifact_path.is_dir():
                tracking_info["artifacts"][str(artifact_path)] = {
                    "files": compute_directory_dvc_hash(artifact_path),
                    "total_files": len(list(artifact_path.glob("**/*"))),
                    "type": "directory"
                }

    with open(tracking_file, 'w') as f:
        json.dump(tracking_info, f, indent=2)

    print(f"   ✅ DVC hash tracking saved: {tracking_file}")
    return tracking_file


# ────────────────────────────────────────────────
# Configuration - MLflow Server Version
# ────────────────────────────────────────────────

_current_file = Path(__file__).resolve()
PACKAGE_ROOT = _current_file.parents[4]
PROJECT_ROOT = _current_file.parents[4]
PACKAGE_DIR=PACKAGE_ROOT / "pricing__epac"

# Définir tous les chemins
MODELS_DIR = settings.MODELS_ARTIFACT_ROOT
DVC_TRACKING_DIR = settings.DVC_TRACKING_ROOT
TARGET = "unit_price"
MLFLOW_MODEL_NAME_GLOBAL = "PricingModelGlobal"
MLFLOW_MODEL_NAME_CLIENT_FEATURES = "ClientFeatures"
CLIENT_FEATURES_FILE = settings.DATA_ROOT / "features" / "client_features.xlsx"
ENRICHED_DATA_FILE = settings.DATA_ROOT / "enriched" / "dataset_with_client_features.xlsx"
# Create necessary directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DVC_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
CLIENT_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

# MLflow configuration to use server
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"\n🔍 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   Artifacts folder: {MODELS_DIR.absolute()}")

# Check server connection
try:
    client = MlflowClient()
    experiments = client.search_experiments()
    print(f"   ✅ Connection to MLflow server established")
    print(f"   📊 {len(experiments)} experiment(s) found")
except Exception as e:
    print(f"   ⚠️ Cannot connect to MLflow server: {e}")
    print(f"   ⚠️ Make sure the server is running with:")
    print("      docker compose up -d postgres-mlflow minio create-bucket mlflow-server")

# Create/Verify experiments
print("\n📊 Creating/Verifying MLflow experiments...")
experiments = [
    ("Pricing_Global_Model", "Global pricing model"),
    ("Pricing_Family_Models", "Models by BindingType"),
    ("Pricing_Couple_Models", "Models by BindingType × SIREN pair"),
    ("Pricing_Client_Features", "Historical client features")
]

for exp_name, exp_desc in experiments:
    try:
        exp_id = mlflow.create_experiment(exp_name)
        print(f"   ✅ Experiment created: {exp_name} (ID: {exp_id})")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            exp = mlflow.get_experiment_by_name(exp_name)
            print(f"   ⚠️ Existing experiment: {exp_name} (ID: {exp.experiment_id})")
        else:
            print(f"   ❌ Error creating {exp_name}: {e}")

CONSOLIDATED_FILE = PROJECT_ROOT /"pricing__epac"/ "data" / "consolidated" / "dataset_complet.xlsx"

RAW_FEATURES = [
    "quantity", "production_page", "height", "thickness", "width",
    "text_paper_type", "text_color", "cover_finish_type", "cover_color",
    "cover_size", "cover_paper_type", "head_and_tail", "priority_level",
    "binding_type", "coil_type", "tab_color", "insert_paper_type",
    "case_finish_type", "spine_type", "label_type", "siren",
    "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
    "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill"
]

NUM_VARS = [
    "quantity",
    "production_page",
    "height",
    "thickness",
    "width",
    # Booleans treated as numeric (0/1)
    "security_label",
    "has_coil",
    "has_insert",
    "has_tab",
    "has_backcover",
    "perf",
    "double_sided_cover",
    "shrinkwrap",
    "three_hole_drill"
]

CAT_VARS = [
    "text_paper_type",
    "text_color",
    "cover_finish_type",
    "cover_color",
    "cover_size",
    "cover_paper_type",
    "head_and_tail",
    "priority_level",
    "binding_type",
    "coil_type",
    "tab_color",
    "insert_paper_type",
    "case_finish_type",
    "spine_type",
    "label_type",
    "siren"
]


# ────────────────────────────────────────────────
# FORCED input schema (without any boolean)
# ────────────────────────────────────────────────
def get_forced_input_schema(available_columns: list = None):
    """
    Generates an MLflow schema where ALL boolean-like columns are long
    If available_columns is provided, only keep the present columns
    """
    schema_list = []

    # Continuous numeric → double
    for col in ["height", "thickness", "width"]:
        if available_columns is None or col in available_columns:
            schema_list.append(ColSpec("double", col, required=True))

    # Integer numeric + booleans treated as integers → long
    long_cols = [
        "quantity", "production_page",
        "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
        "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill"
    ]
    for col in long_cols:
        if available_columns is None or col in available_columns:
            schema_list.append(ColSpec("long", col, required=True))

    # Categorical → string
    for col in CAT_VARS:
        if available_columns is None or col in available_columns:
            schema_list.append(ColSpec("string", col, required=True))

    return Schema(schema_list)


# ────────────────────────────────────────────────
# Constants for aliases AND tags
# ────────────────────────────────────────────────
ALIAS_PRODUCTION = "production"
ALIAS_ARCHIVED = "archived"
ALIAS_STAGING = "staging"  # NEW: Alias for pre-production

# Tags for lifecycle
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
TAG_ARTIFACT_PATH = "artifact_path"  # NEW: To store artifact path
TAG_DVC_HASH = "dvc_hash"  # NEW: DVC hash tag
TAG_DVC_TRACKED = "dvc_tracked"  # NEW: Whether model is tracked with DVC


# ────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────
def get_git_commit_hash() -> str:
    """Retrieves current git commit hash"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True).strip()
    except:
        return "unknown"


def run_dvc_pull(path: Path):
    """Attempts to retrieve a file via DVC if it doesn't exist"""
    if path.exists():
        print(f"[DVC] {path.name} already present.")
        return
    print(f"[DVC] attempting pull → {path}")
    try:
        subprocess.run(["dvc", "pull", str(path)], cwd=PROJECT_ROOT,
                       check=True, capture_output=True, text=True)
        print("[DVC] pull successful")
    except subprocess.CalledProcessError as e:
        print(f"[DVC] Pull failed: {e.stderr.strip() or 'unknown error'}")


def clean_artifacts():
    """Cleans temporary artifacts"""
    print("🧹 Cleaning artifacts...")
    deleted = 0
    patterns = ["best_*", "global_model_*", "family_models_*", "tmp_*", "couple_models_*", "client_features_*"]

    for pattern in patterns:
        for item in MODELS_DIR.glob(pattern):
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {item} → {e}")

    print(f"→ {deleted} item(s) deleted" if deleted else "→ nothing to clean")


# ────────────────────────────────────────────────
# Functions to create comparison tables
# ────────────────────────────────────────────────
def create_comparison_dataframe(results: List[Dict], model_type: str = "global") -> pd.DataFrame:
    """Creates a comparison DataFrame from training results"""
    if not results:
        return pd.DataFrame()

    comparison_data = []
    for r in results:
        row = {
            "Model": r.get("model_name", "Unknown"),
            "RMSE": round(r.get("rmse", 0), 4),
            "R²": round(r.get("r2", 0), 4),
            "MAE": round(r.get("mae", 0), 4),
            "Time (s)": round(r.get("training_time", 0), 2) if "training_time" in r else None
        }

        if model_type == "linear" and "cv_r2" in r:
            row["CV R²"] = round(r["cv_r2"], 4)
            row["Formula"] = r.get("formula", "")[:50] + "..." if r.get("formula") else ""
        elif model_type == "nonlinear" and "shap_success" in r:
            row["SHAP"] = "✅" if r.get("shap_success") else "❌"

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    if not df.empty and "RMSE" in df.columns:
        df = df.sort_values("RMSE").reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

    return df


def save_comparison_table(df: pd.DataFrame, output_path: Path, format: str = "csv") -> Path:
    """Saves a comparison table in different formats"""
    if df.empty:
        return None

    if format == "csv":
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
    elif format == "excel":
        excel_path = output_path.with_suffix(".xlsx")
        df.to_excel(excel_path, index=False)
        return excel_path
    elif format == "html":
        html_path = output_path.with_suffix(".html")
        df.to_html(html_path, index=False)
        return html_path
    elif format == "markdown":
        md_path = output_path.with_suffix(".md")
        df.to_markdown(md_path, index=False)
        return md_path

    return None


def create_comparison_plot(df: pd.DataFrame, output_path: Path, title: str = "Model Comparison"):
    """Creates a performance comparison plot"""
    if df.empty or len(df) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE plot
    ax1 = axes[0]
    models = df["Model"].tolist()
    rmse_values = df["RMSE"].tolist()
    colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(models))]
    bars1 = ax1.barh(range(len(models)), rmse_values, color=colors)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel("RMSE")
    ax1.set_title(f"RMSE - Best: {rmse_values[0]:.4f}")
    ax1.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
        ax1.text(val, bar.get_y() + bar.get_height() / 2, f' {val:.4f}',
                 va='center', ha='left', fontsize=9)

    # R² plot
    ax2 = axes[1]
    r2_values = df["R²"].tolist()
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(models))]
    bars2 = ax2.barh(range(len(models)), r2_values, color=colors)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("R²")
    ax2.set_title(f"R² - Best: {r2_values[0]:.4f}")
    ax2.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars2, r2_values)):
        ax2.text(val, bar.get_y() + bar.get_height() / 2, f' {val:.4f}',
                 va='center', ha='left', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path


# ────────────────────────────────────────────────
# MLflow tag management
# ────────────────────────────────────────────────
def set_model_version_tags(model_name: str, version: int, tags: Dict[str, str],
                           artifact_paths: List[Path] = None) -> bool:
    """
    Adds tags to a specific model version
    Now includes DVC hash if artifact paths are provided
    """
    try:
        client = MlflowClient()

        # Add DVC hash if artifacts are provided
        if artifact_paths:
            dvc_hashes = {}
            for path in artifact_paths:
                if path and path.exists():
                    if path.is_file():
                        dvc_hashes[str(path)] = compute_dvc_hash(path)
                    elif path.is_dir():
                        dvc_hashes[str(path)] = str(compute_directory_dvc_hash(path))

            if dvc_hashes:
                tags[TAG_DVC_HASH] = json.dumps(dvc_hashes)
                tags[TAG_DVC_TRACKED] = "true"

        for key, value in tags.items():
            client.set_model_version_tag(model_name, str(version), key, str(value))
        print(f"   → Tags added to {model_name} v{version}: {list(tags.keys())}")
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to add tags → {model_name} v{version} : {e}")
        return False


def get_model_version_tags(model_name: str, version: int) -> Dict[str, str]:
    """Retrieves all tags from a model version"""
    try:
        client = MlflowClient()
        mv = client.get_model_version(model_name, str(version))
        return mv.tags if hasattr(mv, 'tags') else {}
    except Exception as e:
        print(f"   ⚠️ Error retrieving tags {model_name} v{version}: {e}")
        return {}


def update_lifecycle_status(model_name: str, version: int, new_status: str, reason: str = ""):
    """Updates lifecycle status of a version with tags"""
    tags = {
        TAG_LIFECYCLE_STATUS: new_status,
        f"{TAG_LIFECYCLE_STATUS}_updated_at": datetime.now().isoformat()
    }
    if reason:
        tags[TAG_REASON] = reason

    return set_model_version_tags(model_name, version, tags)


def archive_version_with_tags(model_name: str, version: int, reason: str = "Old production version"):
    """Archives a version using tags"""
    print(f"   📦 Archiving (tags) {model_name} v{version}")
    return update_lifecycle_status(model_name, version, "archived", reason)


# ────────────────────────────────────────────────
# MLflow alias management
# ────────────────────────────────────────────────
def set_model_alias(model_name: str, alias: str, version: int) -> bool:
    """Assigns an alias to a specific model version"""
    try:
        client = MlflowClient()
        client.set_registered_model_alias(model_name, alias, str(version))
        print(f"   → Alias '{alias}' assigned to {model_name} v{version}")
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to assign alias '{alias}' → {model_name} v{version} : {e}")
        return False


def get_model_version_by_alias(model_name: str, alias: str) -> Optional[int]:
    """Retrieves the model version associated with an alias"""
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(model_name, alias)
        return int(mv.version)
    except:
        return None


def model_version_exists(model_name: str, version: int) -> bool:
    """Checks if a specific model version exists"""
    try:
        client = MlflowClient()
        client.get_model_version(model_name, str(version))
        return True
    except:
        return False


def get_all_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """Retrieves all versions of a model with their information"""
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        result = []
        for v in versions:
            version_info = {
                "version": int(v.version),
                "run_id": v.run_id,
                "status": v.status,
                "aliases": [],
                "tags": v.tags if hasattr(v, 'tags') else {}
            }

            # Get aliases if available
            try:
                model_version = client.get_model_version(model_name, v.version)
                if hasattr(model_version, 'aliases'):
                    version_info["aliases"] = model_version.aliases
            except:
                pass

            result.append(version_info)

        result.sort(key=lambda x: x['version'], reverse=True)
        return result
    except Exception as e:
        print(f"   ⚠️ Error retrieving versions for {model_name}: {e}")
        return []


def get_latest_model_version(model_name: str) -> Optional[int]:
    """Retrieves the latest (highest) version of a model"""
    versions = get_all_model_versions(model_name)
    if versions:
        return versions[0]['version']
    return None


def set_production_alias(model_name: str, new_version: int, metrics: Optional[Dict] = None) -> bool:
    """
    Assigns the 'production' alias to the new version
    - This is the only promotion function - single production alias
    - Automatically archives the old version with a tag
    """
    if not isinstance(new_version, int) or new_version < 1:
        print(f"   ⚠️ Invalid version ({new_version}) → skip")
        return False

    print(f"\n📌 Setting production alias for {model_name} v{new_version}")

    if not model_version_exists(model_name, new_version):
        print(f"   ❌ Version {new_version} not found in registry")
        return False

    print(f"   ✅ Version {new_version} found in registry")

    # Verify it's the latest version
    latest_version = get_latest_model_version(model_name)
    if latest_version and new_version != latest_version:
        print(f"   ⚠️ Version {new_version} is not the latest version (latest = v{latest_version})")
        print(f"   ⚠️ Production alias is automatically assigned to the latest version")
        return False

    # Get the old production version
    client = MlflowClient()
    old_version = get_model_version_by_alias(model_name, ALIAS_PRODUCTION)

    if old_version and old_version != new_version:
        print(f"   ↳ Archiving old production version: v{old_version} → tag 'archived'")

        # Add tag to old version
        archive_version_with_tags(
            model_name,
            old_version,
            reason=f"Archived on {datetime.now().strftime('%Y-%m-%d')} - Replaced by v{new_version}"
        )

        # Delete old alias
        try:
            client.delete_registered_model_alias(model_name, ALIAS_PRODUCTION)
            print(f"   → Old alias '{ALIAS_PRODUCTION}' deleted")
        except:
            pass

    # Add tags to new version
    new_version_tags = {
        TAG_LIFECYCLE_STATUS: "production",
        TAG_DEPLOYMENT_DATE: datetime.now().isoformat(),
        TAG_TRAINING_DATE: datetime.now().isoformat(),
    }

    if metrics:
        if "rmse" in metrics:
            new_version_tags[TAG_PERFORMANCE_RMSE] = str(metrics["rmse"])
        if "r2" in metrics:
            new_version_tags[TAG_PERFORMANCE_R2] = str(metrics["r2"])

    set_model_version_tags(model_name, new_version, new_version_tags)

    # Assign new production alias
    print(f"   ↳ Assigning '{ALIAS_PRODUCTION}' → v{new_version}")
    success = set_model_alias(model_name, ALIAS_PRODUCTION, new_version)

    if success:
        after = get_model_version_by_alias(model_name, ALIAS_PRODUCTION)
        if after == new_version:
            print(f"   ✅ Success: production points to v{new_version}")
            return True

    print(f"   ❌ Failed to set production alias")
    return False


# ────────────────────────────────────────────────
# PythonModel class for client features
# ────────────────────────────────────────────────
class ClientFeaturesWrapper(PythonModel):
    """
    Wrapper for client features that inherits from PythonModel
    Allows registration in MLflow as a model
    """

    def __init__(self):
        self.features = None
        self.columns = []
        self.n_clients = 0
        self.n_features = 0
        self.metadata = {}

    def load_context(self, context):
        """Loads features from artifacts"""
        import pandas as pd
        import json
        from pathlib import Path

        # Load features CSV
        features_path = context.artifacts.get("features_csv")
        if features_path and Path(features_path).exists():
            self.features = pd.read_csv(features_path)
            self.columns = self.features.columns.tolist()
            self.n_clients = len(self.features)
            self.n_features = len(self.columns) - 1 if 'siren' in self.columns else len(self.columns)
            print(f"   ✅ Client features loaded: {self.n_clients} clients, {self.n_features} features")

        # Load summary
        summary_path = context.artifacts.get("summary_json")
        if summary_path and Path(summary_path).exists():
            with open(summary_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"   ✅ Metadata loaded")

    def predict(self, context, model_input):
        """
        Prediction method required by PythonModel
        For client features, returns features for a given siren
        """
        if self.features is None:
            return pd.DataFrame({"error": ["Features not loaded"]})

        # If model_input is a single siren or list of sirens
        if isinstance(model_input, (str, int)):
            # Search by siren
            if 'siren' in self.features.columns:
                mask = self.features['siren'].astype(str) == str(model_input)
                if mask.any():
                    result = self.features[mask].to_dict('records')[0]
                    return pd.DataFrame([result])
                else:
                    return pd.DataFrame({"siren": [model_input], "found": [False]})

        # If model_input is a DataFrame with a siren column
        elif isinstance(model_input, pd.DataFrame) and 'siren' in model_input.columns:
            # Join features with provided sirens
            result = model_input[['siren']].copy()
            result = result.merge(self.features, on='siren', how='left')
            return result

        # By default, return all features
        return self.features

    def get_features(self):
        """Returns all features"""
        return self.features

    def get_client_features(self, siren):
        """Retrieves features for a specific client"""
        if self.features is not None and 'siren' in self.features.columns:
            mask = self.features['siren'].astype(str) == str(siren)
            if mask.any():
                return self.features[mask].iloc[0].to_dict()
        return None

    def get_metadata(self):
        """Returns metadata"""
        return self.metadata


# ────────────────────────────────────────────────
# TASK: MLflow Logging Client Features (WITH PRODUCTION ALIAS)
# ────────────────────────────────────────────────
@task(name="MLflow Logging Client Features")
def mlflow_log_client_features(client_features: pd.DataFrame, cleaned_file: Path) -> Dict[str, Any]:
    """
    Registers client features in MLflow as a versioned model
    Now with DVC hash tracking
    """
    print("\n" + "=" * 60)
    print("📊 MLFLOW LOGGING CLIENT FEATURES")
    print("=" * 60)

    # Create dedicated experiment for client features
    mlflow.set_experiment("Pricing_Client_Features")
    run_name = f"client-features-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"📝 MLflow Run ID: {run.info.run_id}")
        print(f"📝 MLflow Experiment ID: {run.info.experiment_id}")

        # Calculate DVC hash of input file
        input_dvc_hash = compute_dvc_hash(cleaned_file)
        print(f"   🔗 Input data DVC hash: {input_dvc_hash}")

        # Tags
        mlflow.set_tag("data_type", "client_features")
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("run_name", run_name)
        mlflow.set_tag("source_file", str(cleaned_file))
        mlflow.set_tag("input_data_dvc_hash", input_dvc_hash)

        # Parameters
        mlflow.log_param("n_clients", len(client_features))
        mlflow.log_param("n_features", len(client_features.columns) - 1)  # -1 for siren
        mlflow.log_param("min_samples_elasticity", 8)
        mlflow.log_param("min_price_cv", 0.05)
        mlflow.log_param("input_data_dvc_hash", input_dvc_hash)

        # Metrics
        if 'client_price_elasticity' in client_features.columns:
            mlflow.log_metric("elasticity_mean", float(client_features['client_price_elasticity'].mean()))
            mlflow.log_metric("elasticity_std", float(client_features['client_price_elasticity'].std()))
            mlflow.log_metric("elasticity_min", float(client_features['client_price_elasticity'].min()))
            mlflow.log_metric("elasticity_max", float(client_features['client_price_elasticity'].max()))

            clients_with_elasticity = (client_features['client_price_elasticity'] != 0).sum()
            mlflow.log_metric("clients_with_elasticity", clients_with_elasticity)
            mlflow.log_metric("pct_clients_with_elasticity",
                              float(clients_with_elasticity / len(client_features) * 100))

        if 'client_seniority_years' in client_features.columns:
            mlflow.log_metric("seniority_mean", float(client_features['client_seniority_years'].mean()))

        if 'client_recency_days' in client_features.columns:
            mlflow.log_metric("recency_mean", float(client_features['client_recency_days'].mean()))

        if 'client_avg_price_ht' in client_features.columns:
            mlflow.log_metric("avg_price_mean", float(client_features['client_avg_price_ht'].mean()))

        # Save features to temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save as CSV
            features_csv = tmp_path / "client_features.csv"
            client_features.to_csv(features_csv, index=False)

            # Save as Excel
            features_excel = tmp_path / "client_features.xlsx"
            client_features.to_excel(features_excel, index=False)

            # Save as joblib (for reuse)
            features_joblib = tmp_path / "client_features.joblib"
            dump(client_features, features_joblib)

            # Calculate DVC hashes for generated files
            csv_hash = compute_dvc_hash(features_csv)
            excel_hash = compute_dvc_hash(features_excel)
            joblib_hash = compute_dvc_hash(features_joblib)

            mlflow.log_params({
                "csv_dvc_hash": csv_hash,
                "excel_dvc_hash": excel_hash,
                "joblib_dvc_hash": joblib_hash
            })

            print(f"   🔗 Generated file hashes:")
            print(f"      CSV: {csv_hash}")
            print(f"      Excel: {excel_hash}")
            print(f"      Joblib: {joblib_hash}")

            # Create JSON summary with all information
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
                "parameters": {
                    "min_samples_elasticity": 8,
                    "min_price_cv": 0.05
                },
                "statistics": {},
                "artifacts": {
                    "csv": "client_features.csv",
                    "excel": "client_features.xlsx",
                    "joblib": "client_features.joblib",
                    "summary": "client_features_summary.json"
                },
                "dvc_hashes": {
                    "csv": csv_hash,
                    "excel": excel_hash,
                    "joblib": joblib_hash
                }
            }

            # Add statistics to summary
            for col in client_features.select_dtypes(include=[np.number]).columns:
                if col != 'siren':
                    summary["statistics"][col] = {
                        "mean": float(client_features[col].mean()),
                        "std": float(client_features[col].std()),
                        "min": float(client_features[col].min()),
                        "max": float(client_features[col].max()),
                        "q25": float(client_features[col].quantile(0.25)),
                        "q50": float(client_features[col].quantile(0.5)),
                        "q75": float(client_features[col].quantile(0.75))
                    }

            summary_json = tmp_path / "client_features_summary.json"
            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)

            # Log all artifacts
            mlflow.log_artifact(str(features_csv), "features")
            mlflow.log_artifact(str(features_excel), "features")
            mlflow.log_artifact(str(features_joblib), "features")
            mlflow.log_artifact(str(summary_json), "features")

            # Create and log visualizations
            if len(client_features) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Elasticity distribution
                if 'client_price_elasticity' in client_features.columns:
                    ax1 = axes[0, 0]
                    elasticity_nonzero = client_features[client_features['client_price_elasticity'] != 0][
                        'client_price_elasticity']
                    if len(elasticity_nonzero) > 0:
                        ax1.hist(elasticity_nonzero, bins=30, edgecolor='black', alpha=0.7)
                        ax1.set_xlabel("Price Elasticity")
                        ax1.set_ylabel("Frequency")
                        ax1.set_title(f"Elasticity Distribution (n={len(elasticity_nonzero)})")
                        ax1.axvline(x=elasticity_nonzero.mean(), color='red', linestyle='--',
                                    label=f'Mean: {elasticity_nonzero.mean():.2f}')
                        ax1.legend()

                # Seniority distribution
                if 'client_seniority_years' in client_features.columns:
                    ax2 = axes[0, 1]
                    ax2.hist(client_features['client_seniority_years'].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    ax2.set_xlabel("Seniority (years)")
                    ax2.set_ylabel("Frequency")
                    ax2.set_title("Client Seniority Distribution")

                # Recency distribution
                if 'client_recency_days' in client_features.columns:
                    ax3 = axes[1, 0]
                    ax3.hist(client_features['client_recency_days'].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    ax3.set_xlabel("Recency (days)")
                    ax3.set_ylabel("Frequency")
                    ax3.set_title("Recency Distribution")

                # Average price distribution
                if 'client_avg_price_ht' in client_features.columns:
                    ax4 = axes[1, 1]
                    ax4.hist(client_features['client_avg_price_ht'].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    ax4.set_xlabel("Average Price HT")
                    ax4.set_ylabel("Frequency")
                    ax4.set_title("Client Average Price Distribution")

                plt.tight_layout()
                plot_path = tmp_path / "client_features_distributions.png"
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()

                mlflow.log_artifact(str(plot_path), "features/plots")

                # Create correlation matrix
                numeric_cols = client_features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = client_features[numeric_cols].corr()

                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    ax.set_xticks(range(len(numeric_cols)))
                    ax.set_yticks(range(len(numeric_cols)))
                    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                    ax.set_yticklabels(numeric_cols)
                    plt.colorbar(im)
                    plt.title("Correlation Matrix - Client Features")

                    for i in range(len(numeric_cols)):
                        for j in range(len(numeric_cols)):
                            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                           ha='center', va='center',
                                           color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')

                    plt.tight_layout()
                    corr_path = tmp_path / "client_features_correlation.png"
                    plt.savefig(corr_path, dpi=100, bbox_inches='tight')
                    plt.close()

                    mlflow.log_artifact(str(corr_path), "features/plots")

            # CREATE AN MLflow MODEL FOR CLIENT FEATURES
            features_wrapper = ClientFeaturesWrapper()

            # Metadata for the model
            metadata = {
                "n_clients": len(client_features),
                "n_features": len(client_features.columns) - 1,
                "columns": list(client_features.columns),
                "timestamp": datetime.now().isoformat(),
                "run_id": run.info.run_id,
                "run_name": run_name,
                "has_elasticity": 'client_price_elasticity' in client_features.columns,
                "has_seniority": 'client_seniority_years' in client_features.columns,
                "has_recency": 'client_recency_days' in client_features.columns,
                "artifact_paths": {
                    "csv": "features/client_features.csv",
                    "excel": "features/client_features.xlsx",
                    "joblib": "features/client_features.joblib",
                    "summary": "features/client_features_summary.json",
                    "plots": "features/plots/"
                },
                "dvc_hashes": {
                    "csv": csv_hash,
                    "excel": excel_hash,
                    "joblib": joblib_hash,
                    "input_data": input_dvc_hash
                }
            }

            if 'client_price_elasticity' in client_features.columns:
                metadata["elasticity_mean"] = float(client_features['client_price_elasticity'].mean())
                metadata["clients_with_elasticity"] = int((client_features['client_price_elasticity'] != 0).sum())

            # Log the model in MLflow
            model_info = mlflow.pyfunc.log_model(
                name="client_features_model",
                python_model=features_wrapper,
                artifacts={
                    "features_csv": str(features_csv),
                    "features_excel": str(features_excel),
                    "features_joblib": str(features_joblib),
                    "summary_json": str(summary_json)
                },
                registered_model_name=MLFLOW_MODEL_NAME_CLIENT_FEATURES,
                metadata=metadata
            )

            version = getattr(model_info, 'registered_model_version', None)

            if version:
                # Save DVC hash tracking file
                dvc_tracking_file = save_dvc_hash_tracking(
                    MLFLOW_MODEL_NAME_CLIENT_FEATURES,
                    int(version),
                    [features_csv, features_excel, features_joblib, summary_json],
                    DVC_TRACKING_DIR
                )

                # Add tags to the version
                tags = {
                    TAG_LIFECYCLE_STATUS: "new",
                    TAG_TRAINING_DATE: datetime.now().isoformat(),
                    TAG_GIT_COMMIT: get_git_commit_hash(),
                    TAG_DATA_TYPE: "client_features",
                    TAG_N_CLIENTS: str(len(client_features)),
                    TAG_N_FEATURES: str(len(client_features.columns) - 1),
                    TAG_RUN_ID: run.info.run_id,
                    TAG_RUN_NAME: run_name,
                    TAG_ARTIFACT_PATH: f"{run.info.artifact_uri}/client_features_model",
                    TAG_DVC_TRACKED: "true",
                    "input_data_dvc_hash": input_dvc_hash,
                    "csv_dvc_hash": csv_hash,
                    "joblib_dvc_hash": joblib_hash,
                    "dvc_tracking_file": str(dvc_tracking_file)
                }

                if 'client_price_elasticity' in client_features.columns:
                    tags["elasticity_mean"] = str(client_features['client_price_elasticity'].mean())

                set_model_version_tags(MLFLOW_MODEL_NAME_CLIENT_FEATURES, int(version), tags)

                # IMPORTANT: Assign 'production' alias to the latest version
                # First check if it's the latest version
                latest_version = get_latest_model_version(MLFLOW_MODEL_NAME_CLIENT_FEATURES)
                if latest_version and int(version) == latest_version:
                    set_production_alias(MLFLOW_MODEL_NAME_CLIENT_FEATURES, int(version),
                                         {"n_clients": len(client_features)})
                    print(f"\n✅ 'production' alias assigned to {MLFLOW_MODEL_NAME_CLIENT_FEATURES} v{version}")
                else:
                    print(f"\n⚠️ Version {version} is not the latest version, alias not assigned")

                print(f"\n✅ Client features model registered: {MLFLOW_MODEL_NAME_CLIENT_FEATURES} v{version}")
                print(f"   📁 Artifacts available at: {run.info.artifact_uri}/client_features_model")
                print(f"   🔗 DVC tracking: {dvc_tracking_file}")
            else:
                print("\n⚠️ Client features model not registered in registry")

        print(f"\n✅ Client features logged in MLflow")
        print(
            f"📊 MLflow Run: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run_name,
            "n_clients": len(client_features),
            "n_features": len(client_features.columns) - 1,
            "model_name": MLFLOW_MODEL_NAME_CLIENT_FEATURES,
            "model_version": int(version) if version else None,
            "artifact_uri": f"{run.info.artifact_uri}/client_features_model",
            "artifacts": {
                "csv": f"{run.info.artifact_uri}/features/client_features.csv",
                "excel": f"{run.info.artifact_uri}/features/client_features.xlsx",
                "joblib": f"{run.info.artifact_uri}/features/client_features.joblib",
                "summary": f"{run.info.artifact_uri}/features/client_features_summary.json"
            },
            "dvc_hashes": {
                "input_data": input_dvc_hash,
                "csv": csv_hash,
                "excel": excel_hash,
                "joblib": joblib_hash
            },
            "dvc_tracking_file": str(dvc_tracking_file) if version else None
        }


# ────────────────────────────────────────────────
# Prefect Tasks (continued)
# ────────────────────────────────────────────────
@task(name="Consolidate data", retries=1)
def consolidate_data_task() -> Path:
    """Data consolidation task"""
    print("🔄 Consolidating...")

    # Check if a specific SQL file is requested via environment variable
    sql_file = os.environ.get('PRICING_SQL_FILE')
    if sql_file and os.path.exists(sql_file):
        print(f"📁 Using specific SQL file: {sql_file}")
        # If run_consolidation accepts a parameter
        try:
            path = run_consolidation(sql_file_path=sql_file)
        except TypeError:
            # If run_consolidation doesn't accept parameters, copy the file
            # to the default location
            default_sql_dir = PACKAGE_DIR / "data" / "raw" / "dumps" / "sql"
            default_sql_dir.mkdir(parents=True, exist_ok=True)
            target = default_sql_dir / "mysql_db_dump.sql"
            shutil.copy2(sql_file, target)
            print(f"📋 File copied to: {target}")
            path = run_consolidation()

    else:
        path = run_consolidation()

    print(f"✅ Consolidation completed → {path}")
    return path


@task(name="Preprocessing", retries=2)
def run_preprocessing() -> Tuple[Path, pd.DataFrame]:
    """Data preprocessing task"""
    input_path = CONSOLIDATED_FILE
    cleaned_path = PACKAGE_DIR / "data" / "processed" / "pricing_fully_cleaned.xlsx"

    if not input_path.exists():
        raise FileNotFoundError(f"Consolidated file not found: {input_path}")

    run_dvc_pull(input_path)

    print(f"🧹 Preprocessing → {input_path.name}")
    df = full_preprocessing(input_path)
    save_processed(df, cleaned_path)
    print(f"✅ Processed data → {cleaned_path}")

    available = [c for c in RAW_FEATURES if c in df.columns]
    if not available:
        raise ValueError("No features available after preprocessing")

    sample_X = df[available].sample(n=min(100, len(df)), random_state=42)
    return cleaned_path, sample_X


@task(name="Global Model Training")
def run_global_training(cleaned_file: Path, sample_X: pd.DataFrame) -> Dict[str, Any]:
    """Trains the global model and registers it in MLflow with DVC hash tracking"""
    print("\n" + "=" * 70)
    print("🌍 GLOBAL MODEL TRAINING")
    print("=" * 70)

    # Force end of any active run
    try:
        mlflow.end_run()
    except:
        pass

    mlflow.set_experiment("Pricing_Global_Model")
    run_name = f"global-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Calculate DVC hash of input file
    input_dvc_hash = compute_dvc_hash(cleaned_file)
    print(f"\n🔗 Input data DVC hash: {input_dvc_hash}")

    # SINGLE PARENT RUN - everything will be logged in this context
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Run ID       : {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")

        # === PARENT RUN TAGS ===
        mlflow.set_tag("model_type", "global")
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("run_name", run_name)
        mlflow.set_tag("input_data_dvc_hash", input_dvc_hash)

        # === PARAMETERS ===
        mlflow.log_params({
            "features_count": len(RAW_FEATURES),
            "num_features": len(NUM_VARS),
            "cat_features": len(CAT_VARS),
            "target": TARGET,
            "input_data_dvc_hash": input_dvc_hash
        })

        # Call train_and_compare WITH THE EXISTING RUN
        # This will prevent train_and_compare from creating a new run
        best_name, results, best_pipe, X_test, y_test, _, feature_importance, feature_importance_json = train_and_compare(
            str(cleaned_file),
            register_to_mlflow=True,
            mlflow_run=run  # ← Passing existing run to avoid duplication
        )

        best_metrics = {}
        if results:
            best = min(results, key=lambda x: x['rmse'])
            best_metrics = {
                "rmse": best['rmse'],
                "r2": best['r2'],
                "mae": best['mae'],
            }
            # Note: metrics are already logged in _log_training_to_mlflow
            # but we also log them at parent level for visibility
            mlflow.log_metrics(best_metrics)

        print("\n📊 Creating model comparison table...")
        comparison_df = create_comparison_dataframe(results, "global")

        if not comparison_df.empty:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                csv_path = tmp_path / "global_model_comparison.csv"
                comparison_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(str(csv_path), "comparisons")
                print(f"   ✅ CSV table saved")

                excel_path = tmp_path / "global_model_comparison.xlsx"
                comparison_df.to_excel(excel_path, index=False)
                mlflow.log_artifact(str(excel_path), "comparisons")
                print(f"   ✅ Excel table saved")

                html_path = tmp_path / "global_model_comparison.html"
                comparison_df.to_html(html_path, index=False)
                mlflow.log_artifact(str(html_path), "comparisons")
                print(f"   ✅ HTML table saved")

                plot_path = tmp_path / "global_model_comparison.png"
                create_comparison_plot(comparison_df, plot_path, "Global Models Comparison")
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path), "comparisons")
                    print(f"   ✅ Comparison plot saved")

                mlflow.log_table(data=comparison_df, artifact_file="comparisons/comparison_table.json")
                print(f"   ✅ Table logged as metric")

        # ===== Custom signature with long for booleans =====
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec

        # Define boolean columns that should be integers
        bool_cols_as_int = [
            "security_label", "has_coil", "has_insert", "has_tab", "has_backcover",
            "perf", "double_sided_cover", "shrinkwrap", "three_hole_drill"
        ]

        # Load a sample of data to build the schema
        df_sample = pd.read_excel(cleaned_file, nrows=5)

        schema_list = []

        # Continuous numeric columns (height, thickness, width) → double
        for col in ["height", "thickness", "width"]:
            if col in df_sample.columns:
                schema_list.append(ColSpec("double", col, required=True))

        # Integer columns (quantity, production_page) → long
        for col in ["quantity", "production_page"]:
            if col in df_sample.columns:
                schema_list.append(ColSpec("long", col, required=True))

        # Booleans as integers → long (NOT boolean)
        for col in bool_cols_as_int:
            if col in df_sample.columns:
                schema_list.append(ColSpec("long", col, required=True))

        # Categorical columns → string
        for col in CAT_VARS:
            if col in df_sample.columns:
                schema_list.append(ColSpec("string", col, required=True))

        # Create input schema
        input_schema = Schema(schema_list)

        # Infer output signature
        sample_input = df_sample[[c for c in df_sample.columns if c in RAW_FEATURES]]
        if sample_input.empty or len(sample_input) == 0:
            sample_input = sample_X.head(5)

        sample_output = best_pipe.predict(sample_input)
        output_schema = infer_signature(sample_output).inputs

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        print(f"\n✅ Custom signature created with {len(schema_list)} columns:")
        for col in signature.inputs.inputs:
            print(f"   - {col.name}: {col.type}")

        desc = f"Global pricing model – best: {best_name} – RMSE {best['rmse']:.4f} – {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # Enriched metadata with feature importance
        model_metadata = {
            "best_model": best_name,
            "model_type": "global",
            "timestamp": datetime.now().isoformat(),
            "run_id": run.info.run_id,
            "run_name": run_name,
            "comparison_table": "comparisons/global_model_comparison.csv",
            "bool_cols_as_int": bool_cols_as_int,
            "input_data_dvc_hash": input_dvc_hash
        }

        # Add feature importance to metadata if available
        if feature_importance:
            model_metadata["feature_importance"] = feature_importance

        # Note: the model is ALREADY registered in _log_training_to_mlflow
        # We retrieve the version from tags or let the internal function handle it

        # Retrieve model version from MLflow client
        version = None
        try:
            client = MlflowClient()
            # Search for the latest model version
            latest_versions = client.get_latest_versions(MLFLOW_MODEL_NAME_GLOBAL, stages=["None"])
            if latest_versions:
                # Take the most recent one (the one just created)
                version = latest_versions[-1].version
        except:
            pass

        # If no version found, use None
        if version:
            # Save model temporarily to compute its hash
            temp_model_path = MODELS_DIR / f"temp_global_model_v{version}.joblib"
            dump(best_pipe, temp_model_path)
            model_dvc_hash = compute_dvc_hash(temp_model_path)
            temp_model_path.unlink()

            # Save DVC hash tracking file
            dvc_tracking_file = save_dvc_hash_tracking(
                MLFLOW_MODEL_NAME_GLOBAL,
                int(version),
                [Path(cleaned_file)],
                DVC_TRACKING_DIR
            )

            try:
                client = MlflowClient()
                client.update_model_version(
                    name=MLFLOW_MODEL_NAME_GLOBAL,
                    version=version,
                    description=desc
                )
                print(f"   ✅ Description added to version {version}")

                # Additional tags for model version
                additional_tags = {
                    TAG_LIFECYCLE_STATUS: "new",
                    TAG_TRAINING_DATE: datetime.now().isoformat(),
                    TAG_GIT_COMMIT: get_git_commit_hash(),
                    TAG_MODEL_TYPE: "global",
                    TAG_BEST_MODEL: best_name,
                    TAG_RUN_ID: run.info.run_id,
                    TAG_RUN_NAME: run_name,
                    TAG_DVC_TRACKED: "true",
                    "bool_cols_as_int": json.dumps(bool_cols_as_int),
                    "input_data_dvc_hash": input_dvc_hash,
                    "model_dvc_hash": model_dvc_hash,
                    "dvc_tracking_file": str(dvc_tracking_file)
                }

                if feature_importance_json:
                    additional_tags["feature_importance"] = feature_importance_json

                if best_metrics:
                    additional_tags[TAG_PERFORMANCE_RMSE] = str(best_metrics["rmse"])
                    additional_tags[TAG_PERFORMANCE_R2] = str(best_metrics["r2"])

                # Update tags (without overwriting already defined ones)
                set_model_version_tags(MLFLOW_MODEL_NAME_GLOBAL, int(version), additional_tags)

            except Exception as e:
                print(f"   ⚠️ Unable to add description/tags: {e}")
        else:
            model_dvc_hash = None
            dvc_tracking_file = None

        print(f"\nBest model: {best_name}")
        print(f"MLflow version: {version or 'not registered'}")
        if feature_importance:
            print(f"📊 Feature importance: {len(feature_importance)} features extracted")
        print(f"🔗 Model DVC hash: {model_dvc_hash or 'N/A'}")

        return {
            "best_model_name": best_name,
            "results": results,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_name": MLFLOW_MODEL_NAME_GLOBAL,
            "model_version": int(version) if version else None,
            "metrics": best_metrics,
            "comparison_table": "comparisons/global_model_comparison.csv",
            "run_name": run_name,
            "bool_cols_as_int": bool_cols_as_int,
            "feature_importance": feature_importance,
            "feature_importance_json": feature_importance_json,
            "dvc_hashes": {
                "input_data": input_dvc_hash,
                "model": model_dvc_hash if version else None
            },
            "dvc_tracking_file": str(dvc_tracking_file) if version else None
        }


@task(name="Create Client Features", retries=1)
def create_client_features_task(cleaned_file: Path) -> Tuple[Path, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Prefect task to create client features with MLflow logging
    """
    print("\n" + "=" * 60)
    print("👥 CREATING CLIENT FEATURES")
    print("=" * 60)

    # Load cleaned data
    print(f"📂 Loading data: {cleaned_file}")
    df = pd.read_excel(cleaned_file)

    # Filter zero prices
    initial_shape = df.shape
    df = df[df["unit_price"] != 0].copy()
    print(f"📊 Data after price=0 filter: {df.shape} (was {initial_shape})")

    # Check for SIREN presence
    if 'siren' not in df.columns:
        print("❌ 'siren' column missing in data")
        return cleaned_file, pd.DataFrame(), df, {"error": "missing_siren"}

    # Create client features
    print("🔄 Computing client features...")
    client_features = create_client_features(
        df,
        min_samples_elasticity=8,
        min_price_cv=0.05
    )

    if client_features.empty:
        print("⚠️ No client features generated, using original data")
        return cleaned_file, pd.DataFrame(), df, {"error": "empty_features"}

    # LOG IN MLflow
    print("\n📤 Registering in MLflow...")
    mlflow_result = mlflow_log_client_features(client_features, cleaned_file)

    # Save locally
    print("💾 Locally saving client features...")
    save_client_features(client_features, CLIENT_FEATURES_FILE)

    # Enrich orders
    print("🔄 Enriching orders...")
    enriched_df = add_client_features_to_orders(df, client_features)

    # Save enriched data
    ENRICHED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_excel(ENRICHED_DATA_FILE, index=False)
    print(f"✅ Enriched data saved: {ENRICHED_DATA_FILE}")

    # Display some statistics
    print("\n📊 Client features statistics:")
    print(f"   - Number of clients: {len(client_features)}")
    print(f"   - MLflow Run ID: {mlflow_result['run_id']}")
    print(f"   - MLflow Experiment: {mlflow_result['experiment_id']}")
    print(f"   - MLflow Model: {mlflow_result.get('model_name', 'N/A')} v{mlflow_result.get('model_version', 'N/A')}")
    print(f"   - Artifacts: {mlflow_result.get('artifact_uri', 'N/A')}")
    if mlflow_result.get('dvc_hashes'):
        print(f"   - DVC hashes: {mlflow_result['dvc_hashes']}")

    if 'client_price_elasticity' in client_features.columns:
        non_zero = (client_features['client_price_elasticity'] != 0).sum()
        print(f"   - Average elasticity: {client_features['client_price_elasticity'].mean():.3f}")
        print(f"   - % clients with elasticity: {non_zero / len(client_features) * 100:.1f}%")

    if 'client_seniority_years' in client_features.columns:
        print(f"   - Average seniority: {client_features['client_seniority_years'].mean():.1f} years")

    if 'client_recency_days' in client_features.columns:
        print(f"   - Average recency: {client_features['client_recency_days'].mean():.0f} days")

    return ENRICHED_DATA_FILE, client_features, enriched_df, mlflow_result


@task(name="Validate Client Features")
def validate_client_features(client_features: pd.DataFrame) -> Dict[str, Any]:

    """
    Validates the quality of client features
    """
    print("\n🔍 CLIENT FEATURES VALIDATION")
    print("=" * 40)

    if client_features.empty:
        print("⚠️ Empty DataFrame - no validation possible")
        return {"error": "empty_dataframe"}

    stats = {
        "n_clients": len(client_features),
        "n_features": len(client_features.columns) - 1,  # -1 for siren
        "missing_values": int(client_features.isna().sum().sum()),
    }

    if 'client_price_elasticity' in client_features.columns:
        stats.update({
            "clients_with_elasticity": int((client_features["client_price_elasticity"] != 0).sum()),
            "elasticity_mean": float(client_features["client_price_elasticity"].mean()),
            "elasticity_std": float(client_features["client_price_elasticity"].std()),
            "elasticity_min": float(client_features["client_price_elasticity"].min()),
            "elasticity_max": float(client_features["client_price_elasticity"].max()),
        })

    if 'client_seniority_years' in client_features.columns:
        stats["seniority_mean"] = float(client_features["client_seniority_years"].mean())

    if 'client_recency_days' in client_features.columns:
        stats["recency_mean"] = float(client_features["client_recency_days"].mean())

    if 'client_avg_price_ht' in client_features.columns:
        stats["avg_price_mean"] = float(client_features["client_avg_price_ht"].mean())

    # Display stats
    print("\n📊 Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    # Quality checks
    issues = []
    if stats["missing_values"] > 0:
        issues.append(f"⚠️ {stats['missing_values']} missing values")

    if 'clients_with_elasticity' in stats and stats["clients_with_elasticity"] < stats["n_clients"] * 0.3:
        issues.append(f"⚠️ Few clients with elasticity: {stats['clients_with_elasticity']}/{stats['n_clients']}")

    if issues:
        print("\n⚠️ Issues detected:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ Validation OK - No major issues")

    return stats


@task(name="Training by BindingType")
def run_family_training(cleaned_file: Path) -> Dict[str, Any]:
    """Trains models by BindingType with MLflow versioning and DVC hash tracking"""
    print("\n" + "=" * 60)
    print("🏷️ STARTING MODELS BY BINDINGTYPE")
    print("=" * 60)

    # Calculate DVC hash of input file
    input_dvc_hash = compute_dvc_hash(cleaned_file)
    print(f"\n🔗 Input data DVC hash: {input_dvc_hash}")

    # DO NOT create a run here - train_by_bindingtype will do it
    # We'll just retrieve run info from training_results

    # Create a list to store models registered in MLflow
    mlflow_created_models = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("   🔄 Launching train_by_bindingtype...")
        # train_by_bindingtype will create its own MLflow run
        training_results = train_by_bindingtype(
            file_path=str(cleaned_file),
            run_linear=True,
            run_nonlinear=True,
            min_samples=50,
            save_pipelines=True,
            output_dir=tmp_path,
            register_to_mlflow=True  # ← This is where the run is created
        )

        # Retrieve run information from training_results
        run_id = training_results.get("run_id")
        experiment_id = training_results.get("experiment_id")
        run_name = training_results.get("run_name", "family-training")

        print(f"📝 MLflow Run ID: {run_id}")
        print(f"📝 MLflow Experiment ID: {experiment_id}")

        # Now we have access to training_results["created_models"]
        local_created_models = training_results.get("created_models", [])

        # Log metrics and artifacts to the EXISTING run
        if run_id:
            client = MlflowClient()

            # === KEEP ALL PARENT RUN TAGS ===
            # Tags already set in train_by_bindingtype
            client.set_tag(run_id, "model_type", "family")
            client.set_tag(run_id, "git_commit", get_git_commit_hash())
            client.set_tag(run_id, "run_name", run_name)
            client.set_tag(run_id, "input_data_dvc_hash", input_dvc_hash)

            # Parameters
            client.log_param(run_id, "input_data_dvc_hash", input_dvc_hash)

            # Global metrics
            client.log_metric(run_id, "n_families_linear", len(training_results["linear"]))
            client.log_metric(run_id, "n_families_nonlinear", len(training_results["nonlinear"]))
            client.log_metric(run_id, "total_families",
                              len(training_results["linear"]) + len(training_results["nonlinear"]))

            # Processing linear models
            print(f"   📊 Processing {len(training_results['linear'])} linear families...")

            linear_comparison_data = []
            for lin in training_results["linear"]:
                linear_comparison_data.append({
                    "Family": lin["family"],
                    "Best model": lin["best_model"],
                    "CV R²": round(lin["cv_r2"], 4),
                    "N samples": lin["n_samples"],
                    "Formula": lin.get("formula", "")[:100] + "..." if lin.get("formula") else ""
                })

            if linear_comparison_data:
                linear_df = pd.DataFrame(linear_comparison_data)
                linear_df = linear_df.sort_values("CV R²", ascending=False).reset_index(drop=True)
                linear_df.insert(0, "Rank", range(1, len(linear_df) + 1))

                linear_csv_path = tmp_path / "linear_families_comparison.csv"
                linear_df.to_csv(linear_csv_path, index=False)
                client.log_artifact(run_id, str(linear_csv_path), "comparisons/families")
                print(f"   ✅ Global comparison table for linear families saved")

            for lin in training_results["linear"]:
                family = lin["family"]
                try:
                    # Log metrics via client
                    client.log_metric(run_id, f"{family}_linear_cv_r2", float(lin["cv_r2"]))
                    client.log_metric(run_id, f"{family}_linear_n_samples", int(lin["n_samples"]))
                    client.set_tag(run_id, f"{family}_linear_model_type", str(lin["best_model"]))

                    # Save metrics JSON locally and log as artifact
                    model_metadata = {
                        "family": family,
                        "cv_r2": float(lin["cv_r2"]),
                        "n_samples": int(lin["n_samples"]),
                        "best_model": str(lin["best_model"]),
                        "model_type": "linear",
                        "run_id": run_id,
                        "run_name": run_name,
                        "input_data_dvc_hash": input_dvc_hash,
                        "formula": lin.get("formula")
                    }

                    metrics_path = tmp_path / f"{family}_linear_metrics.json"
                    with open(metrics_path, "w") as f:
                        json.dump(model_metadata, f, indent=2)
                    client.log_artifact(run_id, str(metrics_path), f"families/{family}/linear")

                    if lin.get("formula"):
                        formula_path = tmp_path / f"{family}_formula.txt"
                        with open(formula_path, "w") as f:
                            f.write(lin["formula"])
                        client.log_artifact(run_id, str(formula_path), f"families/{family}/linear")

                    linear_candidates = lin.get("all_results", [])
                    if linear_candidates:
                        linear_candidates_df = pd.DataFrame(
                            [
                                {
                                    "Model": str(item.get("model_name", "")),
                                    "CV_R2": float(item.get("cv_r2", 0.0)),
                                    "CV_STD": float(item.get("cv_std", 0.0)),
                                    "Selected": "yes" if str(item.get("model_name", "")) == str(lin["best_model"]) else "no",
                                }
                                for item in linear_candidates
                            ]
                        )
                        linear_candidates_df = linear_candidates_df.sort_values("CV_R2", ascending=False).reset_index(drop=True)
                        linear_candidates_df.insert(0, "Rank", range(1, len(linear_candidates_df) + 1))
                        linear_candidates_path = tmp_path / f"{family}_linear_models_comparison.csv"
                        linear_candidates_df.to_csv(linear_candidates_path, index=False)
                        client.log_artifact(run_id, str(linear_candidates_path), f"families/{family}/linear")

                    # Retrieve version info from local_created_models
                    version = None
                    dvc_tracking_file = None
                    for m in local_created_models:
                        if m.get("family") == family and m.get("type") == "linear":
                            version = m.get("version")
                            dvc_tracking_file = m.get("dvc_tracking_file")
                            break

                    if version:
                        mlflow_created_models.append({
                            "model_name": f"PricingModel_{family}_Linear",
                            "version": int(version),
                            "family": family,
                            "type": "linear",
                            "metrics": {"r2": float(lin["cv_r2"]), "n_samples": int(lin["n_samples"])},
                            "run_id": run_id,
                            "run_name": run_name,
                            "formula": lin.get("formula"),
                            "dvc_tracking_file": dvc_tracking_file
                        })
                        print(f"   ✅ Family {family} (linear) versioned: PricingModel_{family}_Linear v{version}")
                except Exception as e:
                    print(f"   ⚠️ Error versioning {family} linear: {e}")

            # Processing nonlinear models
            print(f"   📊 Processing {len(training_results['nonlinear'])} nonlinear families...")

            nonlinear_comparison_data = []
            for nl in training_results["nonlinear"]:
                nonlinear_comparison_data.append({
                    "Family": nl["family"],
                    "Best model": nl["best_model"],
                    "R²": round(nl.get("r2_test", nl.get("r2", 0)), 4),
                    "N samples": nl["n_samples"],
                    "SHAP": "✅" if nl.get("shap_success") else "❌"
                })

            if nonlinear_comparison_data:
                nonlinear_df = pd.DataFrame(nonlinear_comparison_data)
                nonlinear_df = nonlinear_df.sort_values("R²", ascending=False).reset_index(drop=True)
                nonlinear_df.insert(0, "Rank", range(1, len(nonlinear_df) + 1))

                nonlinear_csv_path = tmp_path / "nonlinear_families_comparison.csv"
                nonlinear_df.to_csv(nonlinear_csv_path, index=False)
                client.log_artifact(run_id, str(nonlinear_csv_path), "comparisons/families")
                print(f"   ✅ Global comparison table for nonlinear families saved")

            for nl in training_results["nonlinear"]:
                family = nl["family"]
                try:
                    client.log_metric(run_id, f"{family}_nonlinear_r2", float(nl["r2"]))
                    client.log_metric(run_id, f"{family}_nonlinear_n_samples", int(nl["n_samples"]))
                    client.log_metric(run_id, f"{family}_nonlinear_shap_success", 1 if nl.get("shap_success") else 0)
                    client.set_tag(run_id, f"{family}_nonlinear_model_type", str(nl["best_model"]))

                    model_metadata = {
                        "family": family,
                        "r2": float(nl["r2"]),
                        "n_samples": int(nl["n_samples"]),
                        "best_model": str(nl["best_model"]),
                        "shap_success": 1 if nl.get("shap_success") else 0,
                        "model_type": "nonlinear",
                        "run_id": run_id,
                        "run_name": run_name,
                        "input_data_dvc_hash": input_dvc_hash
                    }

                    metrics_path = tmp_path / f"{family}_nonlinear_metrics.json"
                    with open(metrics_path, "w") as f:
                        json.dump(model_metadata, f, indent=2)
                    client.log_artifact(run_id, str(metrics_path), f"families/{family}/nonlinear")

                    if nl.get("feature_importance"):
                        fi_path = tmp_path / f"{family}_feature_importance.json"
                        fi_serializable = {k: float(v) for k, v in nl["feature_importance"].items()}
                        with open(fi_path, "w") as f:
                            json.dump(fi_serializable, f, indent=2)
                        client.log_artifact(run_id, str(fi_path), f"families/{family}/nonlinear")

                    nonlinear_candidates = nl.get("all_results", [])
                    if nonlinear_candidates:
                        nonlinear_candidates_df = pd.DataFrame(
                            [
                                {
                                    "Model": str(item.get("model_name", "")),
                                    "R2_TEST": float(item.get("r2_test", 0.0)),
                                    "MAE": float(item.get("mae", 0.0)),
                                    "RMSE": float(item.get("rmse", 0.0)),
                                    "Selected": "yes" if str(item.get("model_name", "")) == str(nl["best_model"]) else "no",
                                }
                                for item in nonlinear_candidates
                            ]
                        )
                        nonlinear_candidates_df = nonlinear_candidates_df.sort_values("R2_TEST", ascending=False).reset_index(drop=True)
                        nonlinear_candidates_df.insert(0, "Rank", range(1, len(nonlinear_candidates_df) + 1))
                        nonlinear_candidates_path = tmp_path / f"{family}_nonlinear_models_comparison.csv"
                        nonlinear_candidates_df.to_csv(nonlinear_candidates_path, index=False)
                        client.log_artifact(run_id, str(nonlinear_candidates_path), f"families/{family}/nonlinear")

                    # Retrieve version info from local_created_models
                    version = None
                    dvc_tracking_file = None
                    for m in local_created_models:
                        if m.get("family") == family and m.get("type") == "nonlinear":
                            version = m.get("version")
                            dvc_tracking_file = m.get("dvc_tracking_file")
                            break

                    if version:
                        mlflow_created_models.append({
                            "model_name": f"PricingModel_{family}_NonLinear",
                            "version": int(version),
                            "family": family,
                            "type": "nonlinear",
                            "metrics": {"r2": float(nl["r2"]), "n_samples": int(nl["n_samples"])},
                            "run_id": run_id,
                            "run_name": run_name,
                            "shap_success": nl.get("shap_success", False),
                            "feature_importance": nl.get("feature_importance"),
                            "dvc_tracking_file": dvc_tracking_file
                        })
                        print(f"   ✅ Family {family} (nonlinear) versioned: PricingModel_{family}_NonLinear v{version}")
                except Exception as e:
                    print(f"   ⚠️ Error versioning {family} nonlinear: {e}")

            # Create and log summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "run_name": run_name,
                "input_data_dvc_hash": input_dvc_hash,
                "linear_families": [
                    {
                        "family": f["family"],
                        "best_model": f["best_model"],
                        "cv_r2": float(f["cv_r2"]),
                        "n_samples": int(f["n_samples"]),
                        "model_name": f"PricingModel_{f['family']}_Linear",
                        "version": next((m["version"] for m in mlflow_created_models if
                                         m["family"] == f["family"] and m["type"] == "linear"), None),
                        "formula": next((m.get("formula") for m in mlflow_created_models if
                                         m["family"] == f["family"] and m["type"] == "linear"), None)
                    }
                    for f in training_results["linear"]
                ],
                "nonlinear_families": [
                    {
                        "family": f["family"],
                        "best_model": f["best_model"],
                        "r2": float(nl["r2"]),
                        "n_samples": int(f["n_samples"]),
                        "model_name": f"PricingModel_{f['family']}_NonLinear",
                        "version": next((m["version"] for m in mlflow_created_models if
                                         m["family"] == f["family"] and m["type"] == "nonlinear"), None),
                        "shap_success": next((m.get("shap_success") for m in mlflow_created_models if
                                              m["family"] == f["family"] and m["type"] == "nonlinear"), False)
                    }
                    for f in training_results["nonlinear"]
                ]
            }

            summary_path = tmp_path / "family_models_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            client.log_artifact(run_id, str(summary_path), "summary")

        print(
            f"\n✅ Models by BindingType completed: {len(training_results['linear'])} linear, {len(training_results['nonlinear'])} nonlinear")
        if run_id:
            print(
                f"📊 MLflow Run: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run_id}")

        # Return mlflow_created_models containing models registered in MLflow
        return {
            "n_linear": len(training_results["linear"]),
            "n_nonlinear": len(training_results["nonlinear"]),
            "run_id": run_id,
            "experiment_id": experiment_id,
            "linear_families": [f["family"] for f in training_results["linear"]],
            "nonlinear_families": [f["family"] for f in training_results["nonlinear"]],
            "created_models": mlflow_created_models,
            "run_name": run_name,
            "summary": summary,
            "input_data_dvc_hash": input_dvc_hash
        }


@task(name="Training by Pair (BindingType × SIREN)")
def run_couple_training(cleaned_file: Path) -> Dict[str, Any]:
    """Trains models by pair (binding_type × siren) with MLflow versioning and DVC hash tracking"""
    print("\n" + "=" * 60)
    print("🔗 STARTING MODELS BY PAIR (BINDING_TYPE × SIREN)")
    print("=" * 60)

    # Calculate DVC hash of input file
    input_dvc_hash = compute_dvc_hash(cleaned_file)
    print(f"\n🔗 Input data DVC hash: {input_dvc_hash}")

    # DO NOT create a run here - train_by_bindingtype_siren will do it
    # We'll just retrieve run info from training_results

    created_models = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("   🔄 Launching train_by_bindingtype_siren...")

        # Execute training function - IT WILL CREATE ITS OWN RUN
        try:
            results = train_by_bindingtype_siren(
                file_path=str(cleaned_file),
                save_pipelines=True,
                output_dir=tmp_path
            )

            print(f"   📊 Results structure: {type(results)}")
            if isinstance(results, dict):
                print(f"   📊 Available keys: {list(results.keys())}")
        except Exception as e:
            print(f"   ❌ Error calling train_by_bindingtype_siren: {e}")
            traceback.print_exc()
            return {
                "n_linear": 0,
                "n_nonlinear": 0,
                "run_id": None,
                "experiment_id": None,
                "linear_pairs": [],
                "nonlinear_pairs": [],
                "created_models": [],
                "run_name": None,
                "summary": {"linear_pairs": [], "nonlinear_pairs": []},
                "input_data_dvc_hash": input_dvc_hash
            }

        # Retrieve run information from results
        run_id = results.get("run_id")
        experiment_id = results.get("experiment_id")
        run_name = results.get("run_name", "couple-training")

        print(f"📝 MLflow Run ID: {run_id}")
        print(f"📝 MLflow Experiment ID: {experiment_id}")

        # Adapt results structure
        linear_results = []
        nonlinear_results = []
        created_models_from_training = []

        if isinstance(results, dict):
            linear_results = results.get('linear', [])
            nonlinear_results = results.get('nonlinear', [])
            created_models_from_training = results.get('created_models', [])

        print(f"\n   📊 Linear results count: {len(linear_results)}")
        print(f"   📊 Nonlinear results count: {len(nonlinear_results)}")
        print(f"   📊 Created models from training: {len(created_models_from_training)}")

        if run_id:
            client = MlflowClient()

            # === KEEP ALL PARENT RUN TAGS ===
            client.set_tag(run_id, "model_type", "couple")
            client.set_tag(run_id, "git_commit", get_git_commit_hash())
            client.set_tag(run_id, "run_name", run_name)
            client.set_tag(run_id, "input_data_dvc_hash", input_dvc_hash)

            # Parameters
            client.log_param(run_id, "input_data_dvc_hash", input_dvc_hash)

            # Global metrics
            client.log_metric(run_id, "n_pairs_linear", len(linear_results))
            client.log_metric(run_id, "n_pairs_nonlinear", len(nonlinear_results))
            client.log_metric(run_id, "total_pairs", len(linear_results) + len(nonlinear_results))

            # Create comparison tables for pairs
            print(f"\n   📊 Creating comparison tables...")

            # Comparison table for linear models
            linear_comparison_data = []
            for lin in linear_results:
                if isinstance(lin, dict):
                    linear_comparison_data.append({
                        "Pair": lin.get("group", "Unknown"),
                        "Best model": lin.get("best_model", "Unknown"),
                        "CV R²": round(lin.get("cv_r2", 0), 4),
                        "N samples": lin.get("n_samples", 0),
                        "Formula": lin.get("formula", "")[:100] + "..." if lin.get("formula") else ""
                    })

            if linear_comparison_data:
                linear_df = pd.DataFrame(linear_comparison_data)
                linear_df = linear_df.sort_values("CV R²", ascending=False).reset_index(drop=True)
                linear_df.insert(0, "Rank", range(1, len(linear_df) + 1))

                linear_csv_path = tmp_path / "linear_pairs_comparison.csv"
                linear_df.to_csv(linear_csv_path, index=False)
                client.log_artifact(run_id, str(linear_csv_path), "comparisons/pairs")
                print(f"   ✅ Global comparison table for linear pairs saved")

            # Comparison table for nonlinear models
            nonlinear_comparison_data = []
            for nl in nonlinear_results:
                if isinstance(nl, dict):
                    nonlinear_comparison_data.append({
                        "Pair": nl.get("group", "Unknown"),
                        "Best model": nl.get("best_model", "Unknown"),
                        "R²": round(nl.get("r2_test", nl.get("r2", 0)), 4),
                        "N samples": nl.get("n_samples", 0),
                        "SHAP": "✅" if nl.get("shap_success") else "❌"
                    })

            if nonlinear_comparison_data:
                nonlinear_df = pd.DataFrame(nonlinear_comparison_data)
                nonlinear_df = nonlinear_df.sort_values("R²", ascending=False).reset_index(drop=True)
                nonlinear_df.insert(0, "Rank", range(1, len(nonlinear_df) + 1))

                nonlinear_csv_path = tmp_path / "nonlinear_pairs_comparison.csv"
                nonlinear_df.to_csv(nonlinear_csv_path, index=False)
                client.log_artifact(run_id, str(nonlinear_csv_path), "comparisons/pairs")
                print(f"   ✅ Global comparison table for nonlinear pairs saved")

            # ============================================
            # PROCESS LINEAR MODELS
            # ============================================
            print(f"\n   📊 Processing {len(linear_results)} linear pairs...")

            for idx, lin in enumerate(linear_results):
                try:
                    if not isinstance(lin, dict):
                        continue

                    # Get group name (format: "binding_type__siren")
                    group = lin.get("group", "")
                    if not group:
                        print(f"   ⚠️ No group found for linear model {idx}")
                        continue

                    # Extract binding_type and siren from group
                    if '__' in group:
                        parts = group.split('__')
                        binding_type = parts[0] if len(parts) > 0 else ""
                        siren = parts[1] if len(parts) > 1 else ""
                    else:
                        binding_type = group
                        siren = ""

                    # Create name in desired format
                    if siren:
                        model_name = f"PricingModel_{binding_type}__{siren}_Linear"
                    else:
                        model_name = f"PricingModel_{binding_type}_Linear"

                    # Retrieve pipeline and metrics
                    pipeline = lin.get("pipeline")
                    cv_r2 = float(lin.get("cv_r2", 0))
                    n_samples = int(lin.get("n_samples", 0))
                    best_model = str(lin.get("best_model", "linear_model"))
                    formula = lin.get("formula")

                    # Log metrics via client
                    client.log_metric(run_id, f"{group}_linear_cv_r2", cv_r2)
                    client.log_metric(run_id, f"{group}_linear_n_samples", n_samples)
                    client.set_tag(run_id, f"{group}_linear_model_type", best_model)

                    # Save metrics JSON locally and log as artifact
                    model_metadata = {
                        "group": group,
                        "binding_type": binding_type,
                        "siren": siren,
                        "cv_r2": cv_r2,
                        "n_samples": n_samples,
                        "best_model": best_model,
                        "model_type": "linear",
                        "run_id": run_id,
                        "run_name": run_name,
                        "input_data_dvc_hash": input_dvc_hash,
                        "formula": formula
                    }

                    metrics_path = tmp_path / f"{group}_linear_metrics.json"
                    with open(metrics_path, "w") as f:
                        json.dump(model_metadata, f, indent=2)
                    client.log_artifact(run_id, str(metrics_path), f"pairs/{group}/linear")

                    if formula:
                        formula_path = tmp_path / f"{group}_formula.txt"
                        with open(formula_path, "w") as f:
                            f.write(formula)
                        client.log_artifact(run_id, str(formula_path), f"pairs/{group}/linear")

                    linear_candidates = lin.get("all_results", [])
                    if linear_candidates:
                        linear_candidates_df = pd.DataFrame(
                            [
                                {
                                    "Model": str(item.get("model_name", "")),
                                    "CV_R2": float(item.get("cv_r2", 0.0)),
                                    "CV_STD": float(item.get("cv_std", 0.0)),
                                    "Selected": "yes" if str(item.get("model_name", "")) == best_model else "no",
                                }
                                for item in linear_candidates
                            ]
                        )
                        linear_candidates_df = linear_candidates_df.sort_values("CV_R2", ascending=False).reset_index(drop=True)
                        linear_candidates_df.insert(0, "Rank", range(1, len(linear_candidates_df) + 1))
                        linear_candidates_path = tmp_path / f"{group}_linear_models_comparison.csv"
                        linear_candidates_df.to_csv(linear_candidates_path, index=False)
                        client.log_artifact(run_id, str(linear_candidates_path), f"pairs/{group}/linear")

                    # Retrieve version info from created_models_from_training
                    version = None
                    dvc_tracking_file = None
                    for m in created_models_from_training:
                        if m.get("group") == group and m.get("type") == "linear":
                            version = m.get("version")
                            dvc_tracking_file = m.get("dvc_tracking_file")
                            break

                    if version:
                        created_models.append({
                            "model_name": model_name,
                            "version": int(version),
                            "group": group,
                            "binding_type": binding_type,
                            "siren": siren,
                            "type": "linear",
                            "metrics": {
                                "r2": cv_r2,
                                "n_samples": n_samples
                            },
                            "run_id": run_id,
                            "run_name": run_name,
                            "formula": formula,
                            "dvc_tracking_file": dvc_tracking_file
                        })
                        print(f"   ✅ Pair {group} (linear) versioned: {model_name} v{version}")

                except Exception as e:
                    print(f"   ⚠️ Error processing linear model {idx}: {e}")
                    traceback.print_exc()

            # ============================================
            # PROCESS NONLINEAR MODELS
            # ============================================
            print(f"\n   📊 Processing {len(nonlinear_results)} nonlinear pairs...")

            for idx, nl in enumerate(nonlinear_results):
                try:
                    if not isinstance(nl, dict):
                        continue

                    # Get group name (format: "binding_type__siren")
                    group = nl.get("group", "")
                    if not group:
                        print(f"   ⚠️ No group found for nonlinear model {idx}")
                        continue

                    # Extract binding_type and siren from group
                    if '__' in group:
                        parts = group.split('__')
                        binding_type = parts[0] if len(parts) > 0 else ""
                        siren = parts[1] if len(parts) > 1 else ""
                    else:
                        binding_type = group
                        siren = ""

                    # Create name in desired format
                    if siren:
                        model_name = f"PricingModel_{binding_type}__{siren}_NonLinear"
                    else:
                        model_name = f"PricingModel_{binding_type}_NonLinear"

                    # Retrieve pipeline and metrics
                    pipeline = nl.get("pipeline")
                    r2 = float(nl.get("r2_test", nl.get("r2", 0)))
                    n_samples = int(nl.get("n_samples", 0))
                    best_model = str(nl.get("best_model", "nonlinear_model"))
                    shap_success = nl.get("shap_success", False)
                    feature_importance = nl.get("feature_importance")

                    # Log metrics via client
                    client.log_metric(run_id, f"{group}_nonlinear_r2", r2)
                    client.log_metric(run_id, f"{group}_nonlinear_n_samples", n_samples)
                    client.log_metric(run_id, f"{group}_nonlinear_shap_success", 1 if shap_success else 0)
                    client.set_tag(run_id, f"{group}_nonlinear_model_type", best_model)

                    # Model metadata
                    model_metadata = {
                        "group": group,
                        "binding_type": binding_type,
                        "siren": siren,
                        "r2_test": r2,
                        "n_samples": n_samples,
                        "best_model": best_model,
                        "shap_success": shap_success,
                        "model_type": "nonlinear",
                        "run_id": run_id,
                        "run_name": run_name,
                        "input_data_dvc_hash": input_dvc_hash
                    }

                    if feature_importance:
                        model_metadata["feature_importance"] = feature_importance

                    # Save metrics JSON
                    metrics_path = tmp_path / f"{group}_nonlinear_metrics.json"
                    with open(metrics_path, "w") as f:
                        json.dump(model_metadata, f, indent=2)
                    client.log_artifact(run_id, str(metrics_path), f"pairs/{group}/nonlinear")

                    if feature_importance:
                        fi_path = tmp_path / f"{group}_feature_importance.json"
                        fi_serializable = {k: float(v) for k, v in feature_importance.items()}
                        with open(fi_path, "w") as f:
                            json.dump(fi_serializable, f, indent=2)
                        client.log_artifact(run_id, str(fi_path), f"pairs/{group}/nonlinear")

                    nonlinear_candidates = nl.get("all_results", [])
                    if nonlinear_candidates:
                        nonlinear_candidates_df = pd.DataFrame(
                            [
                                {
                                    "Model": str(item.get("model_name", "")),
                                    "R2_TEST": float(item.get("r2_test", 0.0)),
                                    "MAE": float(item.get("mae", 0.0)),
                                    "RMSE": float(item.get("rmse", 0.0)),
                                    "Selected": "yes" if str(item.get("model_name", "")) == best_model else "no",
                                }
                                for item in nonlinear_candidates
                            ]
                        )
                        nonlinear_candidates_df = nonlinear_candidates_df.sort_values("R2_TEST", ascending=False).reset_index(drop=True)
                        nonlinear_candidates_df.insert(0, "Rank", range(1, len(nonlinear_candidates_df) + 1))
                        nonlinear_candidates_path = tmp_path / f"{group}_nonlinear_models_comparison.csv"
                        nonlinear_candidates_df.to_csv(nonlinear_candidates_path, index=False)
                        client.log_artifact(run_id, str(nonlinear_candidates_path), f"pairs/{group}/nonlinear")

                    # Retrieve version info from created_models_from_training
                    version = None
                    dvc_tracking_file = None
                    for m in created_models_from_training:
                        if m.get("group") == group and m.get("type") == "nonlinear":
                            version = m.get("version")
                            dvc_tracking_file = m.get("dvc_tracking_file")
                            break

                    if version:
                        created_models.append({
                            "model_name": model_name,
                            "version": int(version),
                            "group": group,
                            "binding_type": binding_type,
                            "siren": siren,
                            "type": "nonlinear",
                            "metrics": {
                                "r2": r2,
                                "n_samples": n_samples
                            },
                            "run_id": run_id,
                            "run_name": run_name,
                            "shap_success": shap_success,
                            "feature_importance": feature_importance,
                            "dvc_tracking_file": dvc_tracking_file
                        })
                        print(f"   ✅ Pair {group} (nonlinear) versioned: {model_name} v{version}")

                except Exception as e:
                    print(f"   ⚠️ Error processing nonlinear model {idx}: {e}")
                    traceback.print_exc()

            # ============================================
            # SUMMARY
            # ============================================
            summary = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "run_name": run_name,
                "input_data_dvc_hash": input_dvc_hash,
                "linear_pairs": [
                    {
                        "group": lin.get("group", f"pair_{i}"),
                        "binding_type": lin.get("binding_type", ""),
                        "siren": lin.get("siren", ""),
                        "best_model": lin.get("best_model", "unknown"),
                        "cv_r2": float(lin.get("cv_r2", 0)),
                        "n_samples": int(lin.get("n_samples", 0)),
                        "model_name": f"PricingModel_{lin.get('binding_type', '')}__{lin.get('siren', '')}_Linear" if lin.get(
                            'binding_type') and lin.get(
                            'siren') else f"PricingModel_{lin.get('group', f'pair_{i}')}_Linear",
                        "version": next((m["version"] for m in created_models if
                                         m.get("group") == lin.get("group") and
                                         m["type"] == "linear"), None),
                        "formula": next((m.get("formula") for m in created_models if
                                         m.get("group") == lin.get("group") and
                                         m["type"] == "linear"), None)
                    }
                    for i, lin in enumerate(linear_results) if isinstance(lin, dict)
                ],
                "nonlinear_pairs": [
                    {
                        "group": nl.get("group", f"pair_{i}"),
                        "binding_type": nl.get("binding_type", ""),
                        "siren": nl.get("siren", ""),
                        "best_model": nl.get("best_model", "unknown"),
                        "r2": float(nl.get("r2_test", nl.get("r2", 0))),
                        "n_samples": int(nl.get("n_samples", 0)),
                        "model_name": f"PricingModel_{nl.get('binding_type', '')}__{nl.get('siren', '')}_NonLinear" if nl.get(
                            'binding_type') and nl.get(
                            'siren') else f"PricingModel_{nl.get('group', f'pair_{i}')}_NonLinear",
                        "version": next((m["version"] for m in created_models if
                                         m.get("group") == nl.get("group") and
                                         m["type"] == "nonlinear"), None),
                        "shap_success": next((m.get("shap_success") for m in created_models if
                                              m.get("group") == nl.get("group") and
                                              m["type"] == "nonlinear"), False)
                    }
                    for i, nl in enumerate(nonlinear_results) if isinstance(nl, dict)
                ]
            }

            summary_path = tmp_path / "pair_models_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            client.log_artifact(run_id, str(summary_path), "summary")

        print(
            f"\n✅ Models by pair completed: {len(linear_results)} linear, {len(nonlinear_results)} nonlinear")
        print(f"📦 Models registered in MLflow: {len(created_models)}")
        if run_id:
            print(
                f"📊 MLflow Run: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run_id}")

        return {
            "n_linear": len(linear_results),
            "n_nonlinear": len(nonlinear_results),
            "run_id": run_id,
            "experiment_id": experiment_id,
            "linear_pairs": [
                lin.get("group", f"pair_{i}") if isinstance(lin, dict) else f"pair_{i}" for
                i, lin in enumerate(linear_results)],
            "nonlinear_pairs": [
                nl.get("group", f"pair_{i}") if isinstance(nl, dict) else f"pair_{i}" for
                i, nl in enumerate(nonlinear_results)],
            "created_models": created_models,
            "run_name": run_name,
            "summary": summary,
            "input_data_dvc_hash": input_dvc_hash
        }


# ────────────────────────────────────────────────
# Model promotion (WITH CLIENT FEATURES IN PRODUCTION)
# ────────────────────────────────────────────────
def promote_models(global_result: Optional[Dict] = None,
                   family_result: Optional[Dict] = None,
                   couple_result: Optional[Dict] = None,
                   client_features_result: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Promotes models to production - Single 'production' alias for all
    Client features are also promoted to production with the same alias
    """
    print("\n" + "=" * 60)
    print("🏆 CONFIGURING PRODUCTION ALIAS")
    print("=" * 60)

    results = {
        "global": {"success": False, "message": ""},
        "family": {"total": 0, "success": 0, "failed": []},
        "couple": {"total": 0, "success": 0, "failed": []},
        "client_features": {"success": False, "message": ""},
        "verified": []
    }

    # Promote global model
    if global_result and global_result.get("model_version"):
        model_name = global_result["model_name"]
        version = global_result["model_version"]
        metrics = global_result.get("metrics")
        dvc_info = global_result.get("dvc_hashes", {})

        print(f"\n🌍 Configuring global model: {model_name}")

        latest_version = get_latest_model_version(model_name)
        if latest_version and version == latest_version:
            print(f"   ✅ Version {version} is the latest version")
            success = set_production_alias(model_name, version, metrics)
            results["global"]["success"] = success
            results["global"]["message"] = f"v{version} {'✅' if success else '❌'}"
            if success:
                results["verified"].append({
                    "model_name": model_name,
                    "version": version,
                    "alias": ALIAS_PRODUCTION,
                    "dvc_hashes": dvc_info
                })
        else:
            print(f"   ⚠️ Version {version} is not the latest version (latest = v{latest_version})")
            print(f"   ⚠️ Production alias not assigned")
            results["global"]["success"] = False
            results["global"]["message"] = f"v{version} (not latest version)"

    # Promote family models
    if family_result and family_result.get("created_models"):
        print(f"\n🏷️ Configuring family models:")
        results["family"]["total"] = len(family_result["created_models"])

        for model_info in family_result["created_models"]:
            model_name = model_info["model_name"]
            version = model_info["version"]
            metrics = model_info.get("metrics")

            print(f"\n   → {model_name}")

            latest_version = get_latest_model_version(model_name)
            if latest_version and version == latest_version:
                print(f"      ✅ Version {version} is the latest version")
                success = set_production_alias(model_name, version, metrics)
                if success:
                    results["family"]["success"] += 1
                    results["verified"].append({
                        "model_name": model_name,
                        "version": version,
                        "alias": ALIAS_PRODUCTION,
                        "dvc_tracking_file": model_info.get("dvc_tracking_file")
                    })
                else:
                    results["family"]["failed"].append(model_name)
            else:
                print(f"      ⚠️ Version {version} is not the latest version (latest = v{latest_version})")
                results["family"]["failed"].append(model_name)

    # Promote pair models
    if couple_result and couple_result.get("created_models"):
        print(f"\n🔗 Configuring pair models:")
        results["couple"]["total"] = len(couple_result["created_models"])

        for model_info in couple_result["created_models"]:
            model_name = model_info["model_name"]
            version = model_info["version"]
            metrics = model_info.get("metrics")

            print(f"\n   → {model_name}")

            latest_version = get_latest_model_version(model_name)
            if latest_version and version == latest_version:
                print(f"      ✅ Version {version} is the latest version")
                success = set_production_alias(model_name, version, metrics)
                if success:
                    results["couple"]["success"] += 1
                    results["verified"].append({
                        "model_name": model_name,
                        "version": version,
                        "alias": ALIAS_PRODUCTION,
                        "dvc_tracking_file": model_info.get("dvc_tracking_file")
                    })
                else:
                    results["couple"]["failed"].append(model_name)
            else:
                print(f"      ⚠️ Version {version} is not the latest version (latest = v{latest_version})")
                results["couple"]["failed"].append(model_name)

    # Promote client features (NOW WITH PRODUCTION ALIAS)
    if client_features_result and client_features_result.get("model_version"):
        model_name = client_features_result["model_name"]
        version = client_features_result["model_version"]

        print(f"\n👥 Configuring client features: {model_name}")

        latest_version = get_latest_model_version(model_name)
        if latest_version and version == latest_version:
            print(f"   ✅ Version {version} is the latest version")

            # Use the same 'production' alias as other models
            metrics = {"n_clients": client_features_result.get("n_clients", 0)}
            success = set_production_alias(model_name, version, metrics)

            results["client_features"]["success"] = success
            results["client_features"]["message"] = f"v{version} {'✅' if success else '❌'}"

            if success:
                results["verified"].append({
                    "model_name": model_name,
                    "version": version,
                    "alias": ALIAS_PRODUCTION,
                    "dvc_hashes": client_features_result.get("dvc_hashes", {}),
                    "dvc_tracking_file": client_features_result.get("dvc_tracking_file")
                })

                # Display artifact paths
                if "artifacts" in client_features_result:
                    print(f"\n   📁 Artifacts available:")
                    for art_name, art_path in client_features_result["artifacts"].items():
                        print(f"      - {art_name}: {art_path}")
        else:
            print(f"   ⚠️ Version {version} is not the latest version (latest = v{latest_version})")
            print(f"   ⚠️ Production alias not assigned")
            results["client_features"]["success"] = False
            results["client_features"]["message"] = f"v{version} (not latest version)"

    # Summary
    print("\n" + "=" * 60)
    print("📊 CONFIGURATION SUMMARY")
    print("=" * 60)

    if results["global"]["success"]:
        print(f"✅ Global model: {results['global']['message']}")
    else:
        print(f"❌ Global model: {results['global']['message']}")

    if results["client_features"]["success"]:
        print(f"✅ Client features: {results['client_features']['message']}")
    else:
        print(f"❌ Client features: {results['client_features']['message']}")

    if results["family"]["total"] > 0:
        print(f"✅ Family models: {results['family']['success']}/{results['family']['total']} configured")
        if results["family"]["failed"]:
            print(f"   ❌ Failures: {', '.join(results['family']['failed'][:3])}")
            if len(results['family']['failed']) > 3:
                print(f"      and {len(results['family']['failed']) - 3} others")

    if results["couple"]["total"] > 0:
        print(f"✅ Pair models: {results['couple']['success']}/{results['couple']['total']} configured")
        if results["couple"]["failed"]:
            print(f"   ❌ Failures: {', '.join(results['couple']['failed'][:3])}")
            if len(results['couple']['failed']) > 3:
                print(f"      and {len(results['couple']['failed']) - 3} others")

    print(f"\n🔍 {len(results['verified'])} model(s) with 'production' alias configured")

    return results




