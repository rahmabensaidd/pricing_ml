import os
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from prefect import task

from pricing__epac.src.machine_learning.ingestion.consolidate_data import run_consolidation
from pricing__epac.src.machine_learning.preprocessing.full_prepro import full_preprocessing, save_processed

from pricing__epac.src.machine_learning.pipelines.pipeline_support import run_dvc_pull


@task(name="Consolidate data", retries=1)
def consolidate_data_task(sql_file_env_var: str, package_dir: Path) -> Path:
    print("Consolidating...")

    sql_file = os.environ.get(sql_file_env_var)
    if sql_file and os.path.exists(sql_file):
        print(f"Using specific SQL file: {sql_file}")
        try:
            path = run_consolidation(sql_file_path=sql_file)
        except TypeError:
            default_sql_dir = package_dir / "data" / "raw" / "dumps" / "sql"
            default_sql_dir.mkdir(parents=True, exist_ok=True)
            target = default_sql_dir / "mysql_db_dump.sql"
            shutil.copy2(sql_file, target)
            print(f"File copied to: {target}")
            path = run_consolidation()
    else:
        path = run_consolidation()

    print(f"Consolidation completed -> {path}")
    return path


@task(name="Preprocessing", retries=2)
def run_preprocessing(
    *,
    consolidated_file: Path,
    cleaned_path: Path,
    raw_features: List[str],
    project_root: Path,
) -> Tuple[Path, pd.DataFrame]:
    if not consolidated_file.exists():
        raise FileNotFoundError(f"Consolidated file not found: {consolidated_file}")

    run_dvc_pull(consolidated_file, project_root)

    print(f"Preprocessing -> {consolidated_file.name}")
    df = full_preprocessing(consolidated_file)
    save_processed(df, cleaned_path)
    print(f"Processed data -> {cleaned_path}")

    available = [column for column in raw_features if column in df.columns]
    if not available:
        raise ValueError("No features available after preprocessing")

    sample_x = df[available].sample(n=min(100, len(df)), random_state=42)
    return cleaned_path, sample_x

