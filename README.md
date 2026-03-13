# Pricing MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)
![Prefect](https://img.shields.io/badge/Prefect-2.0+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)
![DVC](https://img.shields.io/badge/DVC-3.0+-red.svg)
![Docker](https://img.shields.io/badge/Docker-%E2%9C%93-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Complete MLOps pipeline** for pricing models with experiment tracking (MLflow), data & model versioning (DVC), pipeline orchestration (Prefect) and production-ready prediction API (FastAPI).

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage & Pipeline Execution](#usage--pipeline-execution)
- [Project Structure](#project-structure)
- [MLflow & Model Registry](#mlflow--model-registry)
- [API & Deployment](#api--deployment)
- [Monitoring & Interpretability](#monitoring--interpretability)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **modern, production-grade MLOps pipeline** dedicated to **pricing models** (B2B printing & binding services).

It covers the full lifecycle:

- Data consolidation from SQL/CSV sources
- Advanced feature engineering (client behavior: elasticity, seniority, recency…)
- Multi-level model training (global → family → client-specific)
- Experiment tracking & model registry
- Data & artifact versioning
- REST API for real-time predictions

---

## ✨ Key Features

| Feature                        | Description                                                                 |
|:-------------------------------|:----------------------------------------------------------------------------|
| ✅ SQL → Consolidated dataset  | Automatic consolidation of multiple SQL dumps / CSV files                   |
| ✅ Advanced client features    | Price elasticity, seniority (years), recency, historical avg price         |
| ✅ Multi-level modeling        | Global model + models per `BindingType` + models per `(BindingType × SIREN)`|
| ✅ 7 regression algorithms     | OLS, Ridge, Lasso, RandomForest, XGBoost, LightGBM, CatBoost               |
| ✅ MLflow Tracking & Registry  | Full experiment tracking, model versioning, aliases (`production`, `staging`) |
| ✅ DVC data/model versioning   | Hash-based tracking of datasets, features & trained models                 |
| ✅ FastAPI prediction service  | Production-ready REST API with OpenAPI documentation                        |
| ✅ Model interpretability      | SHAP values, feature importance, regression formulas (when applicable)     |
| ✅ Automated comparison tables | RMSE / R² / MAE tables + bar charts for model ranking                      |

---

## 🏗️ Architecture
┌─────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│  SQL / CSV      │────▶│ Consolidation      │────▶│ Preprocessing      │
│  (multiple)     │     │ scripts/           │     │ pricing_epac/      │
└─────────────────┘     └────────────────────┘     └────────────────────┘
│
▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│ Client Features    │◀───▶│ Feature Engineering│     │ Training           │
│ (elasticity, etc.) │     │ client_history...  │     │ 3 levels           │
└────────────────────┘     └────────────────────┘     └───────────┬────────┘
│
┌──────────────────────────────────────────────┘
│
┌────────────────────┐       ┌────────────────────┐
│ MLflow Tracking    │◀─────▶│ MLflow Model       │
│ & Experiments      │       │ Registry           │
└──────────┬─────────┘       └──────────┬─────────┘
│                             │
▼                             ▼
┌────────────────────┐       ┌────────────────────┐
│ DVC Versioning     │       │ FastAPI Prediction │
│ (data + models)    │       │ API                │
└────────────────────┘       └────────────────────┘

**Training levels**:

1. **Global model** — one model for all data  
2. **Family models** — one model per `BindingType`  
3. **Couple models** — one model per `(BindingType × SIREN)` pair

---

## 🛠️ Technologies

### Core MLOps stack

| Technology   | Version     | Role                              |
|--------------|-------------|-----------------------------------|
| Python       | 3.9+        | Language                          |
| Prefect 2    | 2.10+       | Pipeline orchestration            |
| MLflow       | 2.9+        | Experiment tracking & registry    |
| DVC          | 3.0+        | Data & model versioning           |
| FastAPI      | 0.100+      | REST API                          |
| Poetry       | latest      | Dependency management             |
| Docker       | —           | Containerization (MinIO, API…)    |

### Data Science stack

- pandas, numpy
- scikit-learn, xgboost, lightgbm, catboost
- SHAP, matplotlib/seaborn
- joblib, pickle

### Storage / Backend

- SQLite (default MLflow backend)
- MinIO (S3-compatible artifact store – optional)

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- Poetry
- Git
- Docker (optional – MinIO / API container)

### Quick setup

```bash
# 1. Clone repository
git clone https://github.com/rahmabensaidd/pricing_ml.git
cd pricing_ml

# 2. Install dependencies with Poetry
poetry install

# 3. Activate virtual environment
poetry shell

# 4. (Recommended) Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# 5. (Optional) Start MinIO (S3-compatible storage)
docker run -d -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  --name minio \
  minio/minio server /data --console-address ":9001"
```
## Usage & Pipeline Execution
## Project Structure (main folders)
pricing_ml/
├── data/
│   ├── raw/            ← SQL dumps, CSVs
│   ├── consolidated/
│   ├── processed/
│   ├── features/
│   └── enriched/
├── models/             ← trained pipelines (.joblib)
├── mlruns/             ← MLflow artifacts
├── pricing_epac/
│   ├── preprocessing/
│   ├── models/
│   │   ├── client_history_features.py
│   │   ├── train_and_compare.py
│   │   ├── train_by_family_bindingtype.py
│   │   └── train_by_family_bindingtypeandsiren.py
│   └── pipeline.py     ← main Prefect flow
├── scripts/
│   └── consolidate_data.py
└── README.md

## MLflow & Model Registry
Models are registered under these names:

PricingModelGlobal
PricingModel_<BindingType>_Linear / ..._NonLinear
PricingModel_<BindingType>__<SIREN>_Linear / ..._NonLinear
ClientFeatures (historical client features)

Aliases in use:

production → currently deployed version
staging    → pre-production / validation (optional)
archived   → old versions (tagged, not aliased)
## API & Deployment
## Monitoring & Interpretability
RMSE, R², MAE per model
SHAP values (tree-based models)
Feature importance plots
Regression formulas (linear models)
Client elasticity distribution
Correlation matrices

All artifacts are logged in MLflow.
## Contributions are welcome!

Fork the repository
Create a feature branch (git checkout -b feature/amazing-thing)
Commit your changes (git commit -m 'Add some amazing thing')
Push to the branch (git push origin feature/amazing-thing)
Open a Pull Request