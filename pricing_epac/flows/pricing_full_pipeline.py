# flows/pricing_full_pipeline.py

from pathlib import Path
from prefect import flow, task

# Imports du package pricing_epac
from pricing_epac.preprocessing.full_preprocess import full_preprocessing, save_processed
from pricing_epac.models.train_and_compare import train_and_compare  # Assure-toi que cette fonction existe


# Racine du projet pour fichiers output (utile si tu veux construire un chemin absolu)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@task(name="Nettoyage complet")
def run_preprocessing(input_file: str, output_file: str) -> Path:
    """
    Exécute le prétraitement complet et sauvegarde le fichier nettoyé.
    """
    print(f"Exécution du prétraitement sur {input_file}")

    # Appel pipeline complet de preprocessing
    df = full_preprocessing(input_file)
    save_processed(df, output_file)

    cleaned_path = PROJECT_ROOT / "data" / "processed" / output_file
    print(f"Fichier nettoyé généré : {cleaned_path}")

    return cleaned_path


@task(name="Entraînement & comparaison modèles")
def run_training(cleaned_file: Path):
    """
    Entraîne et compare plusieurs modèles sur le fichier nettoyé.
    """
    print(f"Entraînement sur {cleaned_file.name}")

    # Appel direct de la fonction train_and_compare
    best_model, results_df = train_and_compare(file_path=str(cleaned_file))

    print("Entraînement terminé.")
    print(f"Meilleur modèle : {best_model}")
    return best_model, results_df


@flow(name="Pipeline Pricing Complet - Nettoyage + Modèles")
def pricing_mlops_pipeline(
    input_file: str = "concateneRAHMA4.xlsx",
    output_cleaned: str = "pricing_fully_cleaned.xlsx"
):
    """
    Pipeline complet : nettoyage + entraînement et comparaison de modèles.
    """
    cleaned_path = run_preprocessing(input_file, output_cleaned)
    run_training(cleaned_path)

    print("Pipeline terminé avec succès !")


if __name__ == "__main__":
    pricing_mlops_pipeline()
