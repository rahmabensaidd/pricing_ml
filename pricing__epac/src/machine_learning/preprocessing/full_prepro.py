# src/preprocessing/full_preprocess.py
"""
Pipeline complet de prétraitement des données pricing en une seule passe.
Combine :
- Nettoyage initial (sélection colonnes, flags, remplissage basique)
- Nettoyage avancé (doublons, mappings, dates sentinelle, imputation ML)
- Rapport qualité + valeurs uniques
- Sauvegarde finale
"""

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import joblib
import pandas as pd
import yaml
from dateutil import parser

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppression des warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constantes globales
MISSING_VALUE = "MISSING"  # Changé de "NONE" pour éviter la confusion
SENTINEL_DATE = pd.Timestamp("9999-12-31")
MAX_UNIQUE_VALUES_TO_DISPLAY = 10
DATE_COLUMNS = ["expected_date", "delivery_date", "reception_date"]
EMPTY_VALUES = ["NAN", "NONE", "MISSING", "N/A", "", " ", "NaT", "<NA>", "NULL"]

# Colonnes à préserver même si constantes
CONSTANT_COLUMNS_TO_KEEP = [
    'has_coil', 'has_insert', 'has_tab', 'has_backcover',
    'insert_paper_type', 'unit_price', 'tva'
]

# Colonnes de dates pour la normalisation finale
FINAL_DATE_COLUMNS = ["expected_date", "delivery_date", "reception_date"]

# Colonnes requises pour le pipeline
REQUIRED_COLUMNS = ['binding_type', 'unit_price']

# Configuration via variable d'environnement
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parents[5]))


def get_project_root() -> Path:
    """Retourne la racine du projet de manière robuste."""
    return PROJECT_ROOT


def load_mappings() -> Dict[str, Dict[str, str]]:
    """
    Charge les mappings depuis un fichier de configuration.
    Si le fichier n'existe pas, retourne un dictionnaire vide.
    """
    mappings_path = get_project_root() / "pricing__epac" / "src" /"config" / "mappings.yaml"

    if mappings_path.exists():
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = yaml.safe_load(f)
            # Nettoyer les mappings : convertir toutes les clés en string
            cleaned_mappings = {}
            for col, mapping in mappings.items():
                cleaned_mappings[col] = {str(k): v for k, v in mapping.items()}
            logger.info(f"Mappings chargés depuis {mappings_path}")
            return cleaned_mappings
        except Exception as e:
            logger.error(f"Erreur lors du chargement des mappings: {e}")
            return {}
    else:
        logger.warning(f"Fichier de mappings non trouvé: {mappings_path}")
        return {}


def safe_create_flag(
        df: pd.DataFrame,
        source_cols: List[str],
        flag_name: str
) -> pd.DataFrame:
    """
    Crée un flag has_* de manière sécurisée.

    Args:
        df: DataFrame source
        source_cols: Liste des colonnes sources
        flag_name: Nom du flag à créer

    Returns:
        DataFrame avec le flag ajouté
    """
    existing_cols = [col for col in source_cols if col in df.columns]

    if existing_cols:
        df[flag_name] = df[existing_cols].notna().any(axis=1).astype(int)
        logger.debug(f"Flag {flag_name} créé à partir de {existing_cols}")
    else:
        df[flag_name] = 0
        logger.debug(f"Colonnes {source_cols} manquantes, {flag_name} mis à 0")

        for col in source_cols:
            if col not in df.columns:
                df[col] = MISSING_VALUE
                logger.debug(f"Création de la colonne {col} avec valeur par défaut")

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Valide que toutes les colonnes requises sont présentes."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes requises manquantes: {missing_cols}")
    logger.debug("Validation des colonnes requises: OK")


# ──────────────────────────────────────────────────────────────
# PARTIE 1 : NETTOYAGE INITIAL (sélection, flags, remplissage simple)
# ──────────────────────────────────────────────────────────────

def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage initial : sélection colonnes, création flags, remplissage basique."""
    logger.info("=== Étape 1 : Nettoyage initial démarré ===")

    # Sélection des colonnes pertinentes
    cols_to_keep = [
        'binding_type', 'height',
        'isbn10', 'isbn13', 'perf', 'production_page', 'security_label',
        'self_cover', 'shrinkwrap', 'status', 'thickness',
        'three_hole_drill', 'unit_price', 'version', 'weight',
        'width', 'label_location', 'label_type',
        'cover_finish_type', 'text_color', 'text_paper_type',
        'priority_level', 'quantity', 'quantity_min', 'quantity_max',
        'siren', 'coil_type', 'cover_paper_type',
        'double_sided_cover', 'cover_color', 'cover_size',
        'insert_lamination', 'insert_paper_type', 'insert_color',
        'insert_size', 'tab_page_number', 'trim_size', 'tab_color',
        'tab_lamination', 'tab_size', 'tab_paper_type',
        'case_finish_type', 'case_paper_type', 'cover_case_color',
        'back_cover_flat_size', 'spine_type', 'head_and_tail', 'tva', 'reception_date'
    ]

    # Garder seulement les colonnes présentes
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].copy()
    logger.info(f"Colonnes conservées : {len(existing_cols)} / {len(cols_to_keep)}")

    # ISBN → 1 si présent et non vide, 0 sinon
    isbn_cols = ['isbn10', 'isbn13']
    for col in isbn_cols:
        if col in df.columns:
            df[col] = (
                    df[col].notna() &
                    (df[col].astype(str).str.strip() != "")
            ).astype(int)

    # TVA → 0 si manquant (garder NaN pour l'imputation)
    if 'tva' in df.columns:
        df['tva'] = df['tva'].fillna(0)

    # Flags has_* (version sécurisée)
    df = safe_create_flag(df, ['insert_lamination', 'insert_paper_type', 'insert_color', 'insert_size'], 'has_insert')
    df = safe_create_flag(df, ['tab_page_number', 'tab_color', 'tab_lamination', 'tab_size', 'tab_paper_type'],
                          'has_tab')
    df = safe_create_flag(df, ['case_finish_type', 'case_paper_type', 'cover_case_color', 'back_cover_flat_size',
                               'spine_type'], 'has_backcover')
    df = safe_create_flag(df, ['coil_type'], 'has_coil')

    # Tailles numériques → -1 (pour distinguer des vraies valeurs)
    sizenum_cols = ['insert_size', 'tab_size', 'back_cover_flat_size', 'trim_size']
    for col in sizenum_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # Tailles catégorielles → MISSING_VALUE
    sizecat_cols = ['cover_size']
    for col in sizecat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(MISSING_VALUE)

    # Catégoriel → MISSING_VALUE (au lieu de DEFAULT_VALUE)
    categorical_defaults = [
        'cover_finish_type', 'text_color', 'text_paper_type', 'case_paper_type',
        'spine_type', 'coil_type', 'author', 'label_location', 'label_type',
        'cover_paper_type', 'cover_color', 'insert_color', 'case_finish_type',
        'case_paper_type', 'tab_paper_type', 'tab_lamination', 'tab_color',
        'tab_page_number', 'head_and_tail', 'cover_case_color',
        'insert_lamination', 'insert_paper_type', 'delivery_date'
    ]
    for col in categorical_defaults:
        if col in df.columns:
            df[col] = df[col].fillna(MISSING_VALUE)

    # Numérique → 0 (garder NaN pour l'imputation)
    numeric_defaults_zero = [
        'three_hole_drill', 'perf', 'double_sided_cover',
        'security_label', 'shrinkwrap'
    ]
    for col in numeric_defaults_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Version → 1
    if 'version' in df.columns:
        df['version'] = df['version'].fillna(1)

    # Validation des colonnes requises
    validate_required_columns(df)

    logger.info(f"Étape 1 terminée. Forme : {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────
# PARTIE 2 : NETTOYAGE AVANCÉ
# ──────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes dupliquées."""
    initial = len(df)
    df = df.drop_duplicates()
    removed = initial - len(df)
    if removed > 0:
        logger.info(f"Doublons supprimés : {removed:,} lignes")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes constantes sauf celles spécifiées."""
    all_constants = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    constants_to_drop = [col for col in all_constants if col not in CONSTANT_COLUMNS_TO_KEEP]

    if constants_to_drop:
        logger.info(f"Colonnes constantes supprimées : {constants_to_drop}")
        constants_preserved = [col for col in all_constants if col in CONSTANT_COLUMNS_TO_KEEP]
        if constants_preserved:
            logger.info(f"Colonnes constantes préservées : {constants_preserved}")
        df = df.drop(columns=constants_to_drop)

    return df


def uppercase_string_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convertit les colonnes string en majuscules.
    Si columns est None, convertit toutes les colonnes object/string.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

    for col in columns:
        if col in df.columns and (df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])):
            df[col] = df[col].astype(str).str.strip().str.upper()

    logger.debug(f"Colonnes mises en majuscules: {len(columns)} colonnes")
    return df


def normalize_column(
        df: pd.DataFrame,
        col: str,
        mapping: Optional[Dict[str, str]] = None,
        default: str = MISSING_VALUE,
        replace_empty: bool = True
) -> pd.DataFrame:
    """
    Normalise une colonne avec mapping optionnel.

    Args:
        df: DataFrame source
        col: Nom de la colonne à normaliser
        mapping: Dictionnaire de mapping des valeurs
        default: Valeur par défaut pour les valeurs vides
        replace_empty: Remplacer les valeurs vides par default
    """
    if col not in df.columns:
        return df

    s = df[col].astype(str)
    if replace_empty:
        s = s.replace(EMPTY_VALUES, default)
    if mapping:
        s = s.replace(mapping)

    df[col] = s
    logger.debug(f"{col:24} → {df[col].value_counts(dropna=False).head(10).to_dict()}")
    return df


def fix_corrupted_dates(series: pd.Series, col_name: Optional[str] = None) -> pd.Series:
    """
    Corrige les dates corrompues en utilisant dateutil.parser.

    Args:
        series: Série pandas contenant les dates
        col_name: Nom de la colonne pour le logging

    Returns:
        Série avec les dates converties en datetime
    """

    def safe_parse(date_str):
        if pd.isna(date_str) or date_str in ['', ' ', 'NONE', 'MISSING']:
            return pd.NaT
        try:
            # Essayer avec dateutil d'abord (plus robuste)
            return pd.to_datetime(parser.parse(str(date_str), fuzzy=False))
        except (ValueError, TypeError, OverflowError):
            try:
                # Fallback sur pandas
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return pd.NaT

    result = series.apply(safe_parse)
    invalid_count = result.isna().sum() - series.isna().sum()
    if invalid_count > 0 and col_name:
        logger.warning(f"{col_name}: {invalid_count} dates n'ont pas pu être converties")

    return result


def clean_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Nettoie et standardise les colonnes de dates."""
    for col in date_columns:
        if col not in df.columns:
            continue

        df[col] = fix_corrupted_dates(df[col], col_name=col)
        # Convertir en datetime et formater
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].dt.strftime("%d-%m-%Y").where(df[col].notna(), df[col])

    return df


def replace_nat_with_sentinel_date(
        df: pd.DataFrame,
        date_columns: List[str],
        sentinel_date: pd.Timestamp = SENTINEL_DATE
) -> pd.DataFrame:
    """Remplace les dates NaT par une date sentinelle."""
    sentinel_str = sentinel_date.strftime("%d-%m-%Y")
    for col in date_columns:
        if col in df.columns:
            nat_count = df[col].isna().sum()
            if nat_count > 0:
                df[col] = df[col].fillna(sentinel_str)
                logger.info(f"{col}: {nat_count} NaT remplacés par {sentinel_str}")
    return df


def impute_cover_size_saddle_stitch(df: pd.DataFrame) -> pd.DataFrame:
    """Impute cover_size pour les Saddle Stitch avec un modèle ML."""
    model_path = get_project_root() / "pricing__epac" / "src" / "machine_learning" / "models" / "cover_size_pipeline.pkl"

    if not model_path.exists():
        logger.error(f"Modèle non trouvé → {model_path}, imputation ignorée")
        # Fallback: imputer avec la valeur la plus fréquente
        if 'cover_size' in df.columns:
            mode_value = df[df['binding_type'] == 'SS']['cover_size'].mode()
            if not mode_value.empty:
                mask_ss = (df["binding_type"] == "SS") & (df["cover_size"] == MISSING_VALUE)
                df.loc[mask_ss, "cover_size"] = mode_value.iloc[0]
                logger.info(f"Imputation fallback avec mode: {mode_value.iloc[0]}")
        return df

    try:
        pipeline = joblib.load(model_path)
        logger.info("Pipeline cover_size chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur chargement modèle : {e}")
        return df

    mask_ss_to_impute = (
            (df["binding_type"] == "SS") &
            (df["cover_size"].isin([MISSING_VALUE, "SDL", "", pd.NA, None]))
    )

    if not mask_ss_to_impute.any():
        logger.info("Aucun SaddleStitch à imputer pour cover_size")
        return df

    df_to_impute = df.loc[mask_ss_to_impute].copy()
    num_features = ["width", "height", "weight", "production_page", "thickness"]
    cat_features = ["cover_color", "cover_paper_type", "text_color", "priority_level"]
    available_num = [f for f in num_features if f in df.columns]
    available_cat = [f for f in cat_features if f in df.columns]

    if not available_num and not available_cat:
        logger.warning("Aucune feature disponible pour l'imputation")
        return df

    X = df_to_impute[available_num + available_cat]

    try:
        preds = pipeline.predict(X)
        df.loc[mask_ss_to_impute, "cover_size"] = preds
        logger.info(f"Imputation cover_size terminée: {len(preds)} valeurs imputées")
    except Exception as e:
        logger.error(f"Erreur pendant l'imputation : {e}")

    return df


def final_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisation finale : remplace NaN/None par MISSING_VALUE."""
    for col in df.columns:
        if col in FINAL_DATE_COLUMNS:
            continue

        # Remplacer les NaN par MISSING_VALUE
        df[col] = df[col].fillna(MISSING_VALUE)

        # Pour les colonnes string, mettre en majuscules
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.upper()

    logger.info("Normalisation finale : NaN/None remplacés par 'MISSING' (sauf dates)")
    return df


def print_unique_values_summary(df: pd.DataFrame, max_values_per_col: int = MAX_UNIQUE_VALUES_TO_DISPLAY):
    """Affiche un résumé des valeurs uniques par colonne."""
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES VALEURS UNIQUES PAR COLONNE (top 10 + comptage total)")
    print("=" * 80)

    for col in df.columns:
        unique_vals = df[col].value_counts(dropna=False)
        total_unique = len(unique_vals)

        print(f"\nColonne : {col} ({df[col].dtype})")
        print(f" - Nombre total de valeurs uniques : {total_unique}")

        if total_unique == 0:
            print(" → Colonne vide")
            continue

        if total_unique == 1:
            print(f" → Valeur unique : {unique_vals.index[0]} ({unique_vals.iloc[0]} fois)")
            continue

        top_n = unique_vals.head(max_values_per_col)
        for val, count in top_n.items():
            print(f" - {val!r:30} : {count:,} ({count / len(df):.1%})")

        if total_unique > max_values_per_col:
            print(f" ... et {total_unique - max_values_per_col} autres valeurs")

    print("=" * 80)


def quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Affiche un rapport de qualité final et retourne les métriques."""
    print("\n=== RAPPORT QUALITÉ FINAL ===")
    print(f"Forme : {df.shape}")
    print("\nTypes des colonnes :")
    print(df.dtypes)

    missing = df.isna().sum()
    if missing.sum() > 0:
        print("\nManquants restants :")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("Aucun manquant → OK")

    print("\nStats numériques :")
    print(df.describe().round(2))

    return {
        'shape': df.shape,
        'missing_count': missing.sum(),
        'missing_by_column': missing[missing > 0].to_dict()
    }


def load_data(input_path: Path) -> pd.DataFrame:
    """Charge les données selon le format du fichier."""
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    if input_path.stat().st_size == 0:
        raise ValueError(f"Fichier vide : {input_path}")

    logger.info(f"Chargement : {input_path}")

    try:
        if input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path, engine="openpyxl")
        elif input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {input_path.suffix}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise

    logger.info(f"Forme brute : {df.shape}")
    return df


def apply_all_mappings(df: pd.DataFrame, mappings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Applique tous les mappings au DataFrame."""
    for col, mapping in mappings.items():
        df = normalize_column(df, col, mapping=mapping)

    # Mapping spécifique pour text_paper_type
    if "text_paper_type" in df.columns:
        text_paper_mapping = {
            "NONE": MISSING_VALUE,
            "80_GLOSSTEXT": "80_GLOSS_TEXT",
            "80_GLOSS_TEXT": "80_GLOSS_TEXT",
            "80_GLOSSCOVER": "80_GLOSS_COVER",
            "10PT_C2S": "10PT_C2S",
            "12PT_C2S": "12PT_C2S",
            "PAP1SW_70": "PAP1_70",
            "PAP1_75": "PAP1_75",
            "LETSGO MATTE 115GSM": "LETSGO_MATTE_115",
            "LETSGO MATTE 90GSM": "LETSGO_MATTE_90",
            "FSC_MC_CVG_SILKHO_1.0_70": "FSC_MC_CVG_SILKHO_1.0_70",
            "FSC_MC_CVG_SILKHO_1.061": "FSC_MC_CVG_SILKHO_1.061",
            "FSC_MC_CVG_SILKHO_1.061_CB": "FSC_MC_CVG_SILKHO_1.061",
            "FSC_MC_CVG_SILKHO_1.0_70_CB": "FSC_MC_CVG_SILKHO_1.0_70",
            "FSC_MC_CVG_SILKHO_1.0_70_BW": "FSC_MC_CVG_SILKHO_1.0_70",
            "FSC_MC_CVG_SILKHO_1.061_CB_BW": "FSC_MC_CVG_SILKHO_1.061",
            "FSC_MC_DOM_VJT_1.21_75": "FSC_MC_DOM_VJT_1.21_75",
            "FSC_MC_DOM_VJT_1.21_75_BW": "FSC_MC_DOM_VJT_1.21_75",
            "FSC_MC_DOM_VJT_1.29_90": "FSC_MC_DOM_VJT_1.29_90",
            "FSC_MC_DOM_VJT_1.29_90_BW": "FSC_MC_DOM_VJT_1.29_90",
            "BIRCH_W40_TB": "BIRCH_W40_TB",
            "SFI_CVG_UCR_1.8_66": "SFI_CVG_66",
        }
        df = normalize_column(df, "text_paper_type", mapping=text_paper_mapping)

    return df


def full_preprocessing(input_path: Path, verbose: bool = False) -> pd.DataFrame:
    """
    Pipeline complet de prétraitement.

    Args:
        input_path: Chemin complet vers le fichier d'entrée (Excel ou CSV)
        verbose: Si True, affiche les rapports détaillés

    Returns:
        DataFrame prétraité
    """
    logger.info("=== PIPELINE COMPLET DE PRÉTRAITEMENT DÉMARRÉ ===")

    # Chargement des données
    df = load_data(input_path)

    # Étape 1 : nettoyage initial
    df = initial_cleaning(df)

    # Étape 2 : nettoyage avancé
    df = remove_duplicates(df)
    df = drop_constant_columns(df)

    # Application des mappings (avant uppercase pour préserver la casse des mappings)
    mappings = load_mappings()
    df = apply_all_mappings(df, mappings)

    # Mettre en majuscules après les mappings
    df = uppercase_string_columns(df)

    # Traitement des dates
    df = clean_dates(df, DATE_COLUMNS)
    df = replace_nat_with_sentinel_date(df, DATE_COLUMNS)

    # Imputation ML
    df = impute_cover_size_saddle_stitch(df)

    # Normalisation finale
    df = final_normalization(df)

    # Rapports (uniquement si verbose)
    if verbose:
        print("\n=== 20 premières lignes du DataFrame final ===")
        print(df.head(20).to_string(index=False))
        print_unique_values_summary(df)
        quality_check(df)

    logger.info("=== PIPELINE COMPLET TERMINÉ ===")
    logger.info(f"Forme finale : {df.shape}")

    return df


def save_processed(df: pd.DataFrame, filename: Optional[str] = None):
    """Sauvegarde le DataFrame traité avec un nom basé sur la date si non spécifié."""
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pricing_fully_cleaned_{timestamp}.xlsx"

    root = get_project_root()
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    df.to_excel(out_path, index=False, engine="openpyxl")
    logger.info(f"Fichier final sauvegardé → {out_path}")
    return out_path


def cleanup_temp_files():
    """Nettoie les fichiers temporaires."""
    root = get_project_root()
    sql_dumps_path = root / "pricing__epac" /"data" / "raw" / "dumps" / "sql"

    if sql_dumps_path.exists():
        shutil.rmtree(sql_dumps_path, ignore_errors=True)
        logger.info(f"Dossier temporaire supprimé : {sql_dumps_path}")


if __name__ == "__main__":
    try:
        root = get_project_root()
        input_path = root / "pricing__epac" / "data" / "consolidated" / "dataset_complet.xlsx"

        # Ajout d'un flag verbose pour les rapports
        import sys

        verbose = "--verbose" in sys.argv or "-v" in sys.argv

        df_final = full_preprocessing(input_path, verbose=verbose)
        save_processed(df_final)
        cleanup_temp_files()

        logger.info("Traitement complet terminé avec succès !")

    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé : {e}")
        raise
    except ValueError as e:
        logger.error(f"Erreur de valeur : {e}")
        raise
    except Exception as e:
        logger.exception(f"Erreur inattendue lors du pipeline : {e}")
        raise