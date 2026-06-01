"""Readable top-level orchestrator for the Pricing MLOps pipeline.

This file intentionally shows only pipeline steps and imports the concrete
task implementations from dedicated modules.
"""

from datetime import datetime
import traceback
from typing import Dict, Optional

import mlflow
from prefect import flow

from pricing__epac.src.machine_learning.pipelines.pipeline_client_features import (
    create_client_features_task as pipeline_create_client_features_task,
    validate_client_features as pipeline_validate_client_features,
)
from pricing__epac.src.machine_learning.pipelines.pipeline_preprocessing import (
    consolidate_data_task as pipeline_consolidate_data_task,
    run_preprocessing as pipeline_run_preprocessing,
)
from pricing__epac.src.machine_learning.pipelines.pricing_full_pipeline_impl import (
    CLIENT_FEATURES_FILE,
    CONSOLIDATED_FILE,
    DVC_TRACKING_DIR,
    ENRICHED_DATA_FILE,
    MLFLOW_MODEL_NAME_CLIENT_FEATURES,
    MLFLOW_MODEL_NAME_GLOBAL,
    PACKAGE_DIR,
    PROJECT_ROOT,
    RAW_FEATURES,
    clean_artifacts,
    promote_models,
    run_couple_training,
    run_family_training,
    run_global_training,
)


@flow(name="Pricing MLOps Pipeline", log_prints=True)
def pricing_mlops_pipeline(
    run_consolidation: bool = True,
    clean: bool = False,
    run_client_features: bool = True,
    run_global: bool = True,
    run_family: bool = True,
    run_couple: bool = False,
    once: bool = False,
    promote: bool = True,
):
    """Orchestrate the full pricing pipeline using imported step tasks."""
    print("\n" + "=" * 80)
    print("PRICING MLOPS PIPELINE".center(80))
    print("=" * 80)
    print(f"\nMLflow Tracking URI: {mlflow.get_tracking_uri()}")

    start_time = datetime.now()
    results: Dict = {}

    if clean:
        clean_artifacts()

    # STEP 1
    if run_consolidation:
        print("\nSTEP 1: DATA CONSOLIDATION")
        pipeline_consolidate_data_task.fn("PRICING_SQL_FILE", PACKAGE_DIR)

    # STEP 2
    print("\nSTEP 2: PREPROCESSING")
    cleaned_path, sample_X = pipeline_run_preprocessing.fn(
        consolidated_file=CONSOLIDATED_FILE,
        cleaned_path=PACKAGE_DIR / "data" / "processed" / "pricing_fully_cleaned.xlsx",
        raw_features=RAW_FEATURES,
        project_root=PROJECT_ROOT,
    )

    # STEP 3
    training_data_path = cleaned_path
    client_features_result: Optional[Dict] = None
    if run_client_features:
        print("\nSTEP 3: CLIENT FEATURES")
        try:
            enriched_path, client_features, enriched_df, mlflow_result = pipeline_create_client_features_task.fn(
                cleaned_file=cleaned_path,
                client_features_file=CLIENT_FEATURES_FILE,
                enriched_data_file=ENRICHED_DATA_FILE,
                mlflow_model_name=MLFLOW_MODEL_NAME_CLIENT_FEATURES,
                dvc_tracking_dir=DVC_TRACKING_DIR,
                project_root=PROJECT_ROOT,
            )
            client_stats = pipeline_validate_client_features.fn(client_features)
            results["client_features"] = {
                "path": str(enriched_path),
                "stats": client_stats,
                "n_clients": len(client_features) if not client_features.empty else 0,
                "mlflow": mlflow_result,
                "dvc_hashes": mlflow_result.get("dvc_hashes", {}),
            }
            client_features_result = mlflow_result
            if enriched_path != cleaned_path and not enriched_df.empty:
                training_data_path = enriched_path
                available_features = [c for c in RAW_FEATURES if c in enriched_df.columns]
                if available_features:
                    sample_X = enriched_df[available_features].sample(n=min(100, len(enriched_df)), random_state=42)
        except Exception as exc:
            print(f"Client features failed: {exc}")
            traceback.print_exc()
            results["client_features"] = {"error": str(exc)}

    # STEP 4
    if run_global:
        print("\nSTEP 4: GLOBAL TRAINING")
        try:
            results["global"] = run_global_training(training_data_path, sample_X)
        except Exception as exc:
            print(f"Global training failed: {exc}")
            traceback.print_exc()
            results["global"] = {"error": str(exc), "model_name": MLFLOW_MODEL_NAME_GLOBAL}

    # STEP 5
    if run_family:
        print("\nSTEP 5: FAMILY TRAINING")
        try:
            results["family"] = run_family_training(training_data_path)
        except Exception as exc:
            print(f"Family training failed: {exc}")
            traceback.print_exc()
            results["family"] = {"error": str(exc)}

    # STEP 6
    if run_couple:
        print("\nSTEP 6: COUPLE TRAINING")
        try:
            results["couple"] = run_couple_training(training_data_path)
        except Exception as exc:
            print(f"Couple training failed: {exc}")
            traceback.print_exc()
            results["couple"] = {"error": str(exc)}

    # STEP 7
    if promote and not once:
        print("\nSTEP 7: PRODUCTION ALIAS PROMOTION")
        global_to_promote = results.get("global") if "global" in results and "error" not in results.get("global", {}) else None
        family_to_promote = results.get("family") if "family" in results and "error" not in results.get("family", {}) else None
        couple_to_promote = results.get("couple") if "couple" in results and "error" not in results.get("couple", {}) else None
        client_features_to_promote = client_features_result if run_client_features and client_features_result else None
        if global_to_promote or family_to_promote or couple_to_promote or client_features_to_promote:
            results["alias_config"] = promote_models(
                global_to_promote,
                family_to_promote,
                couple_to_promote,
                client_features_to_promote,
            )

    duration = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nPipeline completed in {duration:.1f} min")
    return results


if __name__ == "__main__":
    import argparse
    import os
    import shutil
    import sys
    from pathlib import Path

    from pricing__epac.src.machine_learning.ingestion.watcher import SQL_FOLDER

    parser = argparse.ArgumentParser(description="Pricing MLOPS Pipeline")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "global", "family", "couple", "features"])
    parser.add_argument("--input", type=str, help="Specific SQL file to process (optional)")
    parser.add_argument("--no-consolidation", action="store_true", help="Do not run consolidation")
    parser.add_argument("--no-client-features", action="store_true", help="Do not create client features")
    parser.add_argument("--clean", action="store_true", help="Clean artifacts before execution")
    parser.add_argument("--once", action="store_true", help="One-shot mode (no promotion)")
    parser.add_argument("--no-promote", action="store_true", help="Do not configure aliases automatically")
    args = parser.parse_args()

    if args.input:
        input_file = Path(args.input)
        if not input_file.exists():
            print(f"SQL file not found: {input_file}")
            sys.exit(1)

        SQL_FOLDER.mkdir(parents=True, exist_ok=True)
        target_file = SQL_FOLDER / "current_source.sql"
        shutil.copy2(input_file, target_file)
        os.environ["PRICING_SQL_FILE"] = str(input_file)

    should_promote = not args.no_promote and not args.once
    run_client_features = not args.no_client_features

    if args.mode == "global":
        pricing_mlops_pipeline(
            run_consolidation=not args.no_consolidation,
            clean=args.clean,
            run_client_features=run_client_features,
            run_global=True,
            run_family=False,
            run_couple=False,
            once=args.once,
            promote=should_promote,
        )
    elif args.mode == "family":
        pricing_mlops_pipeline(
            run_consolidation=not args.no_consolidation,
            clean=args.clean,
            run_client_features=run_client_features,
            run_global=False,
            run_family=True,
            run_couple=False,
            once=args.once,
            promote=should_promote,
        )
    elif args.mode == "couple":
        pricing_mlops_pipeline(
            run_consolidation=not args.no_consolidation,
            clean=args.clean,
            run_client_features=run_client_features,
            run_global=False,
            run_family=False,
            run_couple=True,
            once=args.once,
            promote=should_promote,
        )
    elif args.mode == "features":
        pricing_mlops_pipeline(
            run_consolidation=not args.no_consolidation,
            clean=args.clean,
            run_client_features=True,
            run_global=False,
            run_family=False,
            run_couple=False,
            once=args.once,
            promote=False,
        )
    else:
        pricing_mlops_pipeline(
            run_consolidation=not args.no_consolidation,
            clean=args.clean,
            run_client_features=run_client_features,
            run_global=True,
            run_family=True,
            run_couple=True,
            once=args.once,
            promote=should_promote,
        )
