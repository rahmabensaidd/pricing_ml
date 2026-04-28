# Architecture

## Overview

The project is organized into four main zones:

- `pricing__epac/src`: application source code
- `pricing__epac/data`: business datasets
- `pricing__epac/artifacts`: exported models and tracking artifacts
- `pricing__epac/runtime`: local execution state (logs, PID, metrics, run outputs)

## End-to-End Flow

```text
SQL dump
  -> consolidation
  -> preprocessing (full_prepro)
  -> client feature engineering
  -> model training
  -> MLflow registration/versioning
  -> API prediction serving
```

## Application Layers

### API Layer

Path: `pricing__epac/src/api`

- `schemas`: Pydantic request/response schemas
- `services`: feature preparation, prediction logic, MLflow service access
- `ml`: model loading and registry access
- `pricing_controller.py`: FastAPI endpoints
- `main.py`: API entry point

### Configuration Layer

Path: `pricing__epac/src/config`

- `settings.py`: centralized environment settings and paths
- `feature_config.py`: business feature definitions and categories

### Shared Utilities

Path: `pricing__epac/src/shared`

- `logging.py`: shared logging strategy used across API, watcher, preprocessing, and training

### Preprocessing Layer

Path: `pricing__epac/src/machine_learning/preprocessing`

- `full_prepro.py`: source of truth for preprocessing
- `full_preprocess.py`: compatibility wrapper

### Training Layer

Path: `pricing__epac/src/machine_learning/training`

- `train.py`: global model training
- `train_and_compare.py`: global model comparison pipeline
- `train_by_family_bindingtype.py`: family-level training
- `train_by_family_bindingtypeandsiren.py`: pair-level training (`binding_type + siren`)
- `client_history_features.py`: historical client feature engineering

### Orchestration Layer

Path: `pricing__epac/src/machine_learning/orchestration`

- `watcher.py`: SQL watcher daemon and pipeline trigger
- `watcher_paths.py`: watcher path utilities
- `watcher_runtime.py`: watcher runtime helpers (PID, metrics, checks, logging helpers)
- `consolidate_data.py`: SQL consolidation into a single dataset

### Flow Layer

Path: `pricing__epac/src/machine_learning/flows`

- `pricing_full_pipeline.py`: top-level ML pipeline orchestration
- `pipeline_preprocessing.py`: extracted preprocessing and consolidation stage tasks
- `pipeline_client_features.py`: extracted client-features stage and MLflow logging
- `pipeline_support.py`: extracted shared pipeline utilities (DVC hash, model aliases, tags, helpers)

## Data, Artifacts, and Runtime

### Business Data

Path: `pricing__epac/data`

- `raw`: SQL dumps and static raw sources
- `consolidated`: consolidated dataset output
- `processed`: cleaned dataset output
- `enriched`: datasets enriched with client features
- `features`: exported client feature files

### Artifacts

Path: `pricing__epac/artifacts`

- `models`: local exported model files
- `dvc_tracking`: local hash tracking files

### Runtime State

Path: `pricing__epac/runtime`

- `logs`: watcher and pipeline logs
- `watcher`: watcher PID, metrics, and dump-tracking files
- `pipeline_results`: run output artifacts

## MLflow Deployment Target

Target deployment uses:

- MLflow backend store: PostgreSQL
- MLflow artifact store: MinIO (S3-compatible)

Normal operation should not rely on local `mlruns` storage.

## Backward Compatibility

Compatibility wrappers still exist to avoid breaking old imports during migration:

- `pricing__epac/src/api/models`
- `pricing__epac/src/machine_learning/models`
- `pricing__epac/src/machine_learning/scripts`

These wrappers should be removed progressively once all imports are migrated.
