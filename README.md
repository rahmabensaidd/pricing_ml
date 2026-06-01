# Pricing EPAC

Pricingg EPAC is an end-to-end pricing platform for print products.  
It ingests SQL dumps, builds cleaned datasets, trains multiple ML models, tracks them in MLflow, and serves predictions through a FastAPI API.

## What This Project Does

Main business flow:

1. The watcher monitors `pricing__epac/data/raw/dumps/sql`.
2. SQL dumps are consolidated into a single dataset.
3. `full_prepro` cleans and normalizes data.
4. Client history features are computed.
5. Global, family, and pair models are trained.
6. Models and metrics are logged to MLflow.
7. The API serves predictions from production models.

## Current Layout

- Code: `pricing__epac/src`
- Business data: `pricing__epac/data`
- Artifacts: `pricing__epac/artifacts`
- Runtime state (logs, PID, watcher metrics): `pricing__epac/runtime`

Core packages:

- `pricing__epac/src/api`
- `pricing__epac/src/config`
- `pricing__epac/src/machine_learning/preprocessing`
- `pricing__epac/src/machine_learning/training`
- `pricing__epac/src/machine_learning/ingestion`
- `pricing__epac/src/machine_learning/pipelines`
- `pricing__epac/src/shared`

## Requirements

- Python 3.11+
- Poetry
- Docker + Docker Compose
- A valid `.env` file at the repository root

You can bootstrap env vars from:

```bash
cp .env.example .env
```

## Key Environment Variables

Minimum required variables:

- `MYSQL_PASSWORD`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_POSTGRES_USER`
- `MLFLOW_POSTGRES_PASSWORD`
- `MLFLOW_POSTGRES_DB`
- `MLFLOW_S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Quick Start

### Run in Local (after git pull)

Use these exact steps when a developer clones or pulls the repository and wants to run the project locally.

1. Clone and enter the repository

```bash
git clone https://github.com/rahmabensaidd/pricing_ml.git
cd pricing_ml
```

2. Prepare environment variables

```bash
cp .env.example .env
```

Then adjust values in `.env` if needed for your machine.

3. Start the full local stack with Docker

```bash
docker compose up -d --build
```

4. Verify services

- Pricing API docs: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`
- MinIO console: `http://localhost:9001`
- Prefect UI: `http://localhost:4200`

5. Check logs when debugging

```bash
docker compose logs -f pricing-api
docker compose logs -f mlflow-server
docker compose logs -f mysql
```

6. Stop services

```bash
docker compose down
```

7. Full cleanup (including Docker volumes/data)

```bash
docker compose down -v
```

### Optional: run API in local Python dev mode

If you prefer running API directly from Python instead of Docker container:

```bash
poetry install
poetry run python -m pricing__epac.src.api.main --reload
```

Optional ML jobs:

```bash
poetry run python -m pricing__epac.src.machine_learning.ingestion.watcher
poetry run python -m pricing__epac.src.machine_learning.pipelines.pricing_full_pipeline
```

### 1. Install dependencies

```bash
poetry install
```

### 2. Start the MLflow stack

The target stack uses:

- PostgreSQL for MLflow backend store
- MinIO for artifact storage

```bash
docker compose up -d
```

This will start:

- `mysql` on `localhost:3307`
- `postgres-mlflow` on `localhost:5433`
- `minio` on `localhost:9000` (console `:9001`)
- `mlflow-server` on `localhost:5000`
- `prefect-server` on `localhost:4200`
- `pricing-api` on `localhost:8000`

Useful services:

- MLflow: `http://localhost:5000`
- MinIO console: `http://localhost:9001`
- Prefect UI: `http://localhost:4200`
- Pricing API docs: `http://localhost:8000/docs`

### 3. Start the API

```bash
poetry run python -m pricing__epac.src.api.main --reload
```

Useful endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### 4. Start the watcher

```bash
poetry run python -m pricing__epac.src.machine_learning.ingestion.watcher
```

Watcher outputs:

- Logs: `pricing__epac/runtime/logs`
- PID and metrics: `pricing__epac/runtime/watcher`

### 5. Run the pipeline manually

```bash
poetry run python -m pricing__epac.src.machine_learning.pipelines.pricing_full_pipeline
```

## Useful Tree

```text
pricing__epac/
  src/
    api/
      schemas/
      services/
      ml/
    config/
    machine_learning/
      preprocessing/
      training/
      ingestion/
      pipelines/
    shared/
    tests/
  data/
    raw/
    consolidated/
    processed/
    enriched/
    features/
  artifacts/
    models/
    dvc_tracking/
  runtime/
    logs/
    watcher/
    pipeline_results/
```

## Local Validation

Quick checks:

```bash
poetry run python -c "import pricing__epac.src.api.pricing_controller"
poetry run python -c "from pricing__epac.src.machine_learning.ingestion.watcher import SQLFileHandler"
poetry run python -m pytest pricing__epac/src/tests -q
```

## Docker Automation on Main

This repository includes GitHub Actions workflow:

- `.github/workflows/docker-pricing.yml`

Trigger:

- On push to `main`
- Manual run (`workflow_dispatch`)

It builds and pushes:

- `${DOCKERHUB_USERNAME}/pricing-epac-api`
- `${DOCKERHUB_USERNAME}/pricing-epac-mlflow`
- `${DOCKERHUB_USERNAME}/pricing-epac-prefect`

Required GitHub secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## Migration Notes

- `pricing__epac/mlruns` is no longer the normal runtime path when MLflow runs on PostgreSQL + MinIO.
- Runtime outputs were moved from data folders to `pricing__epac/runtime`.
- Artifacts are now under `pricing__epac/artifacts`.
- `full_preprocess.py` wrapper was removed; use `full_prepro.py` as the only preprocessing module.
- Legacy wrapper folders `pricing__epac/src/machine_learning/models` and
  `pricing__epac/src/machine_learning/scripts` were removed to keep a single canonical import path.
- Training public modules are now:
  - `global_training.py`
  - `bindingtype_training.py`
  - `bindingtype_siren_training.py`
  - `client_features_training.py`

## More Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md)
