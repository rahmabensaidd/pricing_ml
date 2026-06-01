# Feature -> Fichiers (Guide Rapide)

## Prediction API
- Entree API: `pricing__epac/src/api/main.py`
- Endpoints: `pricing__epac/src/api/pricing_controller.py`
- Service prediction: `pricing__epac/src/api/services/pricing_service.py`
- Chargement modeles: `pricing__epac/src/api/ml/model_loader.py`

## Watcher SQL (auto pipeline)
- Daemon watcher: `pricing__epac/src/machine_learning/orchestration/watcher.py`
- Consolidation SQL: `pricing__epac/src/machine_learning/orchestration/consolidate_data.py`

## Pipeline global
- Orchestrateur pipeline: `pricing__epac/src/machine_learning/flows/pricing_full_pipeline.py`
- Etape preprocess: `pricing__epac/src/machine_learning/flows/pipeline_preprocessing.py`
- Etape features client: `pricing__epac/src/machine_learning/flows/pipeline_client_features.py`
- Utilitaires pipeline: `pricing__epac/src/machine_learning/flows/pipeline_support.py`

## Preprocessing
- Source principale: `pricing__epac/src/machine_learning/preprocessing/full_prepro.py`

## Training global
- Orchestrateur: `pricing__epac/src/machine_learning/training/train.py`
- Data I/O: `pricing__epac/src/machine_learning/training/data_io.py`
- Feature prep: `pricing__epac/src/machine_learning/training/feature_prep.py`
- Modeles: `pricing__epac/src/machine_learning/training/model_registry.py`
- Evaluation + MLflow: `pricing__epac/src/machine_learning/training/evaluation.py`

## Training specialise
- Par famille: `pricing__epac/src/machine_learning/training/train_by_family_bindingtype.py`
- Par couple (binding_type + siren): `pricing__epac/src/machine_learning/training/train_by_family_bindingtypeandsiren.py`
- Features historiques client: `pricing__epac/src/machine_learning/training/client_history_features.py`

## Config et commun
- Settings: `pricing__epac/src/config/settings.py`
- Feature config: `pricing__epac/src/config/feature_config.py`
- Logging commun: `pricing__epac/src/shared/logging.py`

