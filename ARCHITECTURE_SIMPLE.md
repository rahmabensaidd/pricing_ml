# Architecture Simple (Meme Features)

Objectif: garder les memes fonctionnalites, avec une lecture tres claire.

## 1) Structure a retenir

```text
pricing__epac/
  src/
    api/                       # Exposer les predictions
    config/                    # Parametres et chemins
    machine_learning/
      orchestration/           # Watcher + consolidation
      preprocessing/           # Nettoyage/normalisation
      training/                # Entrainement modeles
      flows/                   # Pipeline global (ordre des etapes)
    shared/                    # Logging et utilitaires communs
    tests/                     # Tests
  data/                        # Donnees (raw -> consolidated -> processed -> enriched)
  artifacts/                   # Modeles exportes
  runtime/                     # Logs et etats d'execution
```

## 2) Regle de lisibilite

- `flows/` = orchestration metier (quoi executer et dans quel ordre)
- `training/` = logique modele (comment entrainer/mesurer/sauvegarder)
- `orchestration/` = execution continue (watcher, triggers)
- `preprocessing/` = transformations de donnees
- `api/` = service en ligne pour inferer

## 3) Training (deja simplifie)

```text
training/
  train.py            # orchestrateur court
  data_io.py          # chargement, validation, nettoyage, sauvegarde artifacts
  feature_prep.py     # preprocessor sklearn
  model_registry.py   # catalogue des modeles + hyperparametres
  evaluation.py       # metriques, feature importance, MLflow
  train_and_compare.py# point d'entree compatible
```

## 4) Flux fonctionnel (inchangé)

```text
SQL dump
 -> orchestration/watcher.py
 -> orchestration/consolidate_data.py
 -> preprocessing/full_prepro.py
 -> training/*
 -> MLflow + artifacts
 -> api/*
```

## 5) Convention simple pour les nouveaux fichiers

- 1 fichier = 1 responsabilite principale
- pas de fichier > 400-500 lignes sans raison forte
- noms explicites (`*_service.py`, `*_registry.py`, `*_pipeline.py`)
- pas de wrappers legacy

