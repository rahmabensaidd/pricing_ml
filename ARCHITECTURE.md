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

### Training Layer

Path: `pricing__epac/src/machine_learning/training`

- `global_training.py`: public facade for global model training
- `bindingtype_training.py`: public facade for family-level training
- `bindingtype_siren_training.py`: public facade for pair-level training (`binding_type + siren`)
- `client_features_training.py`: public facade for client historical features
- `_global_core/*`: internal implementation of global training
- `_bindingtype_core/*`: internal implementation of family training
- `_bindingtype_siren_core/*`: internal implementation of pair training
- `_client_features_core/*`: internal implementation of client-feature engineering

### Orchestration Layer

Path: `pricing__epac/src/machine_learning/ingestion`

- `watcher.py`: SQL watcher daemon and pipeline trigger
- `watcher_paths.py`: watcher path utilities
- `watcher_runtime.py`: watcher runtime helpers (PID, metrics, checks, logging helpers)
- `consolidate_data.py`: SQL consolidation into a single dataset

### Flow Layer

Path: `pricing__epac/src/machine_learning/pipelines`

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

Legacy wrappers under `machine_learning/models` and `machine_learning/scripts` have been removed.
Imports now target the canonical packages directly:

- `pricing__epac/src/machine_learning/training`
- `pricing__epac/src/machine_learning/ingestion`
- `pricing__epac/src/machine_learning/pipelines`
