# Architecture Simplifiee (Version Jury)

Objectif: expliquer le projet clairement sans changer les fonctionnalites.

## 1) Vision simple en 4 blocs

1. Entree des donnees
- Dossier: `pricing__epac/data/raw/dumps/sql`
- Role: recevoir les nouveaux fichiers SQL.

2. Traitement ML
- Dossier principal: `pricing__epac/src/machine_learning`
- Role: consolider, nettoyer, creer des features, entrainer les modeles.

3. Exposition API
- Dossier: `pricing__epac/src/api`
- Role: charger les modeles en production et retourner les predictions.

4. Observabilite et stockage
- Dossiers: `pricing__epac/artifacts`, `pricing__epac/runtime`
- Outils: MLflow + logs
- Role: garder les versions de modeles, metriques et traces d'execution.

## 2) Arborescence minimale a montrer au jury

```text
Pricing_epac/
  pricing__epac/
    src/
      api/                # endpoints et services de prediction
      config/             # variables, chemins, mappings
      machine_learning/   # pipeline data + training
      shared/             # utilitaires communs (logging, etc.)
      tests/              # tests
    data/
      raw/                # entrees brutes (SQL, CSV)
      consolidated/       # donnees consolidees
      processed/          # donnees nettoyees
      enriched/           # donnees enrichies
      features/           # features exportees
    artifacts/            # modeles exportes, tracking local
    runtime/              # logs, metriques watcher, resultats pipeline
```

## 3) Enchainement fonctionnel (de bout en bout)

```text
Nouveau SQL
 -> watcher detecte
 -> consolidation des dumps
 -> preprocessing
 -> feature engineering client
 -> entrainement (global/famille/couple)
 -> enregistrement MLflow + artifacts
 -> API charge le modele "production"
 -> prediction
```

## 4) Pourquoi le projet parait "trop long"

Ca venait principalement de:
- quelques gros fichiers de pipeline/training (logique metier dense),
- des couches legacy de compatibilite d'import.

Important: ces couches legacy ont ete retirees pour garder une seule structure lisible.

## 5) Architecture officielle a retenir (sans modifier le code)

Pour ton explication jury, considere que le "coeur officiel" est:

- `pricing__epac/src/api`
- `pricing__epac/src/config`
- `pricing__epac/src/machine_learning/preprocessing`
- `pricing__epac/src/machine_learning/training`
- `pricing__epac/src/machine_learning/orchestration`
- `pricing__epac/src/machine_learning/flows`
- `pricing__epac/src/shared`

Et que le package `training/` est maintenant organise en sous-modules clairs:
- `data_io.py`
- `feature_prep.py`
- `model_registry.py`
- `evaluation.py`
- `train.py` (orchestrateur court)

## 6) Plan de simplification progressive (sans perte de fonctionnalites)

Phase 1 (documentation uniquement):
1. Valider cette architecture simplifiee comme reference.
2. Presenter le flux en 7 etapes (ci-dessus).
3. Garder le code tel quel.

Phase 2 (amelioration continue):
1. Continuer le decoupage des gros fichiers restants (flow + training famille/couple).
2. Ajouter un "entrypoints map" (fichier court des commandes officielles).
3. Conserver les memes interfaces API et memes sorties pipeline.

## 7) Commandes officielles (a retenir)

API:
```bash
poetry run python -m pricing__epac.src.api.main --reload
```

Watcher:
```bash
poetry run python -m pricing__epac.src.machine_learning.orchestration.watcher
```

Pipeline manuel:
```bash
poetry run python -m pricing__epac.src.machine_learning.flows.pricing_full_pipeline
```

---

Si tu presentes cette version, tu montres une architecture claire, modulaire et evolutive, tout en gardant exactement le meme comportement du systeme actuel.
