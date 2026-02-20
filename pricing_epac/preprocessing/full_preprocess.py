# src/preprocessing/full_preprocess.py
"""
Pipeline complet de prétraitement des données pricing en une seule passe.
Combine :
- Nettoyage initial (sélection colonnes, flags, remplissage basique)
- Nettoyage avancé (doublons, mappings, dates sentinelle, imputation ML)
- Rapport qualité + valeurs uniques
- Sauvegarde finale
"""

import pandas as pd
import unicodedata
import re
from pathlib import Path
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)


def get_project_root() -> Path:
    """Retourne la racine du projet de manière robuste."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "requirements.txt").exists() or (current / ".git").exists():
            return current
        current = current.parent
    # Fallback rare
    return Path(__file__).resolve().parents[3]


# ──────────────────────────────────────────────────────────────
# PARTIE 1 : NETTOYAGE INITIAL (sélection, flags, remplissage simple)
# ──────────────────────────────────────────────────────────────

def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage initial : sélection colonnes, création flags, remplissage basique."""
    print("=== Étape 1 : Nettoyage initial démarré ===")

    # Sélection des colonnes pertinentes
    cols_to_keep = [
        'author', 'binding_type', 'delivery_date', 'height',
        'isbn10', 'isbn13', 'perf', 'production_page', 'security_label',
        'self_cover', 'shrinkwrap', 'status', 'thickness',
        'three_hole_drill', 'title', 'unit_price', 'version', 'weight',
        'width', 'label_location', 'label_type',
        'cover_finish_type', 'text_color', 'text_paper_type',
        'expected_date', 'reception_date',
        'priority_level', 'quantity', 'quantity_min', 'quantity_max',
        'siren', 'coil_type', 'cover_paper_type',
        'double_sided_cover', 'cover_color', 'cover_size',
        'insert_lamination', 'insert_paper_type', 'insert_color',
        'insert_size', 'tab_page_number', 'trim_size', 'tab_color',
        'tab_lamination', 'tab_size', 'tab_paper_type',
        'case_finish_type', 'case_paper_type', 'cover_case_color',
        'back_cover_flat_size', 'spine_type', 'tva', 'head_and_tail'
    ]

    # Garder seulement les colonnes présentes
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].copy()
    print(f"Colonnes conservées : {len(existing_cols)} / {len(cols_to_keep)}")

    # ISBN → 1 si présent et non vide, 0 sinon
    isbn_cols = ['isbn10', 'isbn13']
    for col in isbn_cols:
        if col in df.columns:
            df[col] = (
                df[col].notna() &
                (df[col].astype(str).str.strip() != "")
            ).astype(int)

    # TVA → 0 si manquant
    if 'tva' in df.columns:
        df['tva'] = df['tva'].fillna(0)

    # Flags has_*
    insert_cols = ['insert_lamination', 'insert_paper_type', 'insert_color', 'insert_size']
    df['has_insert'] = df[insert_cols].notna().any(axis=1).astype(int)

    tab_cols = ['tab_page_number', 'tab_color', 'tab_lamination', 'tab_size', 'tab_paper_type']
    df['has_tab'] = df[tab_cols].notna().any(axis=1).astype(int)

    backcover_cols = ['case_finish_type', 'case_paper_type', 'cover_case_color', 'back_cover_flat_size', 'spine_type']
    df['has_backcover'] = df[backcover_cols].notna().any(axis=1).astype(int)

    coil_cols = ['coil_type']
    df['has_coil'] = df[coil_cols].notna().any(axis=1).astype(int)

    # Tailles numériques → -1
    sizenum_cols = ['insert_size', 'tab_size', 'back_cover_flat_size', 'trim_size']
    for col in sizenum_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # Tailles catégorielles → 'NONE'
    sizecat_cols = ['cover_size']
    for col in sizecat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('NONE')

    # Catégoriel → "NONE"
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
            df[col] = df[col].fillna("NONE")

    # Numérique → 0
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

    # Réordonnancement logique
    ordered_cols = [
        'expected_date', 'reception_date', 'delivery_date',
        'siren', 'author', 'title', 'binding_type', 'text_paper_type', 'text_color', 'double_sided_cover',
        'cover_finish_type', 'cover_color', 'cover_size', 'cover_paper_type', 'head_and_tail',
        'priority_level', 'status', 'version', 'quantity', 'quantity_min', 'quantity_max',
        'production_page', 'height', 'thickness', 'weight', 'width',
        'isbn10', 'isbn13', 'perf',
        'self_cover', 'shrinkwrap', 'three_hole_drill', 'tva', 'unit_price', 'security_label',
        'label_location', 'label_type',
        'has_coil', 'coil_type',
        'has_insert', 'insert_lamination', 'insert_paper_type', 'insert_color', 'insert_size',
        'has_tab', 'tab_page_number', 'trim_size', 'tab_color',
        'tab_lamination', 'tab_size', 'tab_paper_type',
        'has_backcover', 'case_finish_type', 'case_paper_type', 'cover_case_color',
        'back_cover_flat_size', 'spine_type'
    ]

    ordered_cols_existing = [col for col in ordered_cols if col in df.columns]
    df = df[ordered_cols_existing]

    print("Étape 1 terminée. Forme : ", df.shape)
    return df


# ──────────────────────────────────────────────────────────────
# PARTIE 2 : NETTOYAGE AVANCÉ
# ──────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    df = df.drop_duplicates()
    removed = initial - len(df)
    if removed > 0:
        print(f"Doublons supprimés (avancé) : {removed:,} lignes")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constants = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constants:
        print(f"Colonnes constantes supprimées (avancé) : {constants}")
        df = df.drop(columns=constants)
    return df


def uppercase_all_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
    print(f"→ Colonnes chaînes mises en MAJUSCULES : {list(str_cols)}")
    return df


def normalize_column(
    df: pd.DataFrame,
    col: str,
    mapping: dict | None = None,
    default: str = "NONE",
    replace_empty: bool = True
) -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = df[col].astype(str)
    if replace_empty:
        s = s.replace(["NAN", "NONE", "N/A", "", " ", "NaT", "<NA>"], default)
    if mapping:
        s = s.replace(mapping)
    df[col] = s
    print(f"{col:24} → {df[col].value_counts(dropna=False).head(10).to_dict()}")
    return df


def fix_corrupted_dates(series: pd.Series, col_name: str = None) -> pd.Series:
    def _fix_single_date(value):
        if pd.isnull(value):
            return pd.NaT
        val_str = str(value).strip()
        if val_str.startswith('00') and len(val_str) >= 4 and val_str[2:4].isdigit():
            year_part = val_str[:4]
            corrected_year = '20' + year_part[2:]
            val_str = corrected_year + val_str[4:]
        try:
            dt = pd.to_datetime(val_str, errors='coerce')
            if pd.notnull(dt):
                year = dt.year
                if year < 1900 or year > 2100:
                    if abs(year) < 100:
                        corrected_year = 2000 + abs(year) if year < 0 else 2000 + year
                        val_str = str(corrected_year) + val_str[4:]
                    elif str(year).startswith('00'):
                        corrected_year = '20' + str(year)[2:]
                        val_str = corrected_year + val_str[len(str(year)):]
            return pd.to_datetime(val_str, errors='coerce')
        except:
            return pd.NaT

    fixed = series.apply(_fix_single_date)
    invalid_count = fixed.isna().sum() - series.isna().sum()
    if invalid_count > 0 and col_name:
        print(f"→ {col_name}: {invalid_count} dates corrigées ou rendues NaT")
    return fixed


def clean_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    for col in date_columns:
        if col not in df.columns:
            continue
        df[col] = fix_corrupted_dates(df[col], col_name=col)
        dt = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed", dayfirst=False)
        failed = dt.isna()
        if failed.any():
            dt2 = pd.to_datetime(df.loc[failed, col], dayfirst=True, errors="coerce", utc=True, format="mixed")
            dt.update(dt2)
        dt = dt.dt.tz_localize(None)
        df[col] = dt.dt.strftime("%d-%m-%Y").where(dt.notna(), df[col].astype(str).str.strip())
    return df


def replace_nat_with_sentinel_date(
    df: pd.DataFrame,
    date_columns: list[str],
    sentinel_date: pd.Timestamp = pd.Timestamp("9999-12-31")
) -> pd.DataFrame:
    sentinel_str = sentinel_date.strftime("%d-%m-%Y")
    for col in date_columns:
        if col in df.columns:
            nat_count = df[col].isna().sum()
            if nat_count > 0:
                df[col] = df[col].fillna(sentinel_date)
                print(f"→ {col}: {nat_count} NaT remplacés par {sentinel_str}")
    return df


def impute_cover_size_saddle_stitch(df: pd.DataFrame) -> pd.DataFrame:
    model_path = get_project_root() / "src" / "models" / "cover_size_pipeline.pkl"
    if not model_path.exists():
        print(f"ERREUR : Modèle non trouvé → {model_path}")
        return df
    try:
        pipeline = joblib.load(model_path)
        print("Pipeline cover_size chargé avec succès")
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        return df
    mask_ss_to_impute = (
        (df["binding_type"] == "SS") &
        (df["cover_size"].isin(["NONE", "SDL", "", pd.NA, None]))
    )
    if not mask_ss_to_impute.any():
        print("Aucun SaddleStitch à imputer pour cover_size")
        return df
    df_to_impute = df.loc[mask_ss_to_impute].copy()
    num_features = ["width", "height", "weight", "production_page", "thickness"]
    cat_features = ["cover_color", "cover_paper_type", "text_color", "priority_level"]
    available_num = [f for f in num_features if f in df.columns]
    available_cat = [f for f in cat_features if f in df.columns]
    X = df_to_impute[available_num + available_cat]
    try:
        preds = pipeline.predict(X)
        df.loc[mask_ss_to_impute, "cover_size"] = preds
        print("Imputation cover_size terminée")
        print(df["cover_size"].value_counts(dropna=False))
    except Exception as e:
        print(f"Erreur pendant l'imputation : {e}")
    return df


def replace_nat_nan_none(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = ["expected_date", "delivery_date", "reception_date"]
    for col in df.columns:
        if col in date_cols:
            continue
        df[col] = df[col].fillna("NONE")
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.upper()
    print("→ Normalisation finale : NaN/None remplacés par 'NONE' (sauf dates)")
    return df


def print_unique_values_summary(df: pd.DataFrame, max_values_per_col: int = 10):
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


def quality_check(df: pd.DataFrame):
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


def full_preprocessing(input_file: str = "concateneRAHMA4.xlsx") -> pd.DataFrame:
    print("=== PIPELINE COMPLET DE PRÉTRAITEMENT DÉMARRÉ ===")

    # Chargement brut
    root = get_project_root()
    file_path = root / "data" / "raw" / input_file
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    print(f"Chargement : {file_path}")
    df = pd.read_excel(file_path, engine="openpyxl")
    print(f"Forme brute : {df.shape}")

    # Étape 1 : nettoyage initial
    df = initial_cleaning(df)

    # Étape 2 : nettoyage avancé
    df = remove_duplicates(df)
    df = drop_constant_columns(df)
    df = uppercase_all_string_columns(df)

    # Mappings
    mappings = {
        "siren": {"TLG": "CENGAGE", "NGL": "CENGAGE", "CLM": "CENGAGE"},
        "text_color": {"1": "1/1", "1/C": "1/1", "1C": "1/1", "4": "4/4", "4/C": "4/4", "4C": "4/4"},
        "cover_finish_type": {
            "LGLOSS": "LAYFLAT-GLOSS", "LAYFLAT GLOSS": "LAYFLAT-GLOSS",
            "LMATT_SF": "LAYFLAT MATTE SCUFF-FREE", "LAYFLAT MATTE SCUFF-FREE": "LAYFLAT MATTE SCUFF-FREE",
            "LMATTE": "LAYFLAT-MATTE", "MATT": "MATT",
        },
        "cover_color": {"4/C": "4/4", "4C": "4/4", "40": "4/0", "41": "4/1", "0": "0/0", "00": "0/0"},
        "binding_type": {
            "LOOSELEAF_NO_COVER": "LOOSELEAF-NC", "CASEBOUND": "CASEBIND",
            "CASEBINDENDSHEET": "CASEBIND-ES", "CASEBOUND-ES": "CASEBIND-ES",
            "COILHARDTAB": "COILHARD-TAB", "CARD": "CARD", "PERFECT-NC": "PERFECT-NC",
            "SADDLESTITCH": "SS", "SADDLE STITCH": "SS",
        },
        "cover_paper_type": {
            "100_GLOSSTEXT": "100_GLOSS_TEXT", "100_GLOSS_TEXT": "100_GLOSS_TEXT",
            "100_GLOSSTEXT\t": "100_GLOSS_TEXT",
            "80_GLOSS_TEXT": "80_GLOSS_TEXT", "80_GLOSSTEXT": "80_GLOSS_TEXT",
            "80_GLOSSCOVER": "80_GLOSS_COVER",
        },
        "head_and_tail": {
            "BLACK & WHITE": "BLACK & WHITE", "BLACK/WHITE": "BLACK & WHITE",
            "WHITE/WHITE": "WHITE", "WHITE": "WHITE", "NONE": "NONE"
        },
        "priority_level": {"HIGH": "HIGH1", "HIGH*": "HIGH1", "HIGH**": "HIGH2"},
        "status": {
            "NEW": "NEW", "ACCEPTED": "ACCEPTED", "PENDING": "PENDING", "PROOF OUT": "PROOF_OUT",
            "ONPROD": "ON_PROD", "READY": "READY",
            "DELIVERED": "DELIVERED", "PARTIAL DELIVERED": "PARTIAL DELIVERED",
            "INVOICED": "INVOICED", "COMPLETE": "COMPLETE",
            "CANCELLED": "CANCELLED", "CANCELED": "CANCELLED", "ONHOLD": "ON_HOLD"
        },
        "label_location": {
            "ES PAGE 2/BOTTOM/RIGHT": "OTHER", "NONE": "NONE",
            "SEE SAMPLE": "OTHER", "LABEL LOCATION MXTST": "OTHER"
        },
        "label_type": {
            "NONE": "NONE", "STANDARD": "STANDARD", "YES": "STANDARD",
            "ISBN": "ISBN", "ISBN-ST": "ISBN", "978-0-357-37403-0": "ISBN",
            "BAR CODE": "ISBN", "NO (IF BAR CODE LABEL)": "ISBN",
            "ADVANCE COPY (SILVER)": "OTHER",
            "FLORIDA": "OTHER", "RELX": "OTHER", "2": "OTHER",
            "GRADE 1": "OTHER", "GRADE 2": "OTHER", "GRADE 3": "OTHER",
            "GRADE 4": "OTHER", "GRADE 5": "OTHER", "GRADE K": "OTHER",
            "LABEL TYPE MXTST": "OTHER"
        },
        "case_finish_type": {
            "NONE": "NONE", "LAYFLAT-GLOSS": "LAYFLAT-GLOSS",
            "LAYFLAT MATTE": "LAYFLAT-MATTE", "LAYFLAT MATTE SCUFF-FREE": "LAYFLAT MATTE SCUFF-FREE",
            "GLOSS FILM": "GLOSS-FILM", "GLOSS-FILM": "GLOSS-FILM",
            "LAYFLAT GLOSS": "LAYFLAT-GLOSS"
        },
        "case_paper_type": {
            "NONE": "NONE", "100_GLOSSTEXT": "100_GLOSS_TEXT",
            "10PT_C1S": "10PT_C1S", "16PT_C1S": "16PT_C1S"
        },
        "coil_type": {
            "NONE": "NONE", "METAL": "METAL", "METAL WHITE": "METAL",
            "PLASTIC": "PLASTIC", "PLASTIC BLACK": "PLASTIC",
            "PLASTIC WHITE": "PLASTIC", "BLACK PLASTIC": "PLASTIC",
            "1/C": "PLASTIC", "4/C": "PLASTIC"
        },
        "tab_lamination": {
            "NONE": "NONE", "MYLAR": "MYLAR", "YES": "LAYFLAT-GLOSS", "NO": "NONE",
            "LAYFLAT-GLOSS": "LAYFLAT-GLOSS", "LAYFLAT GLOSS": "LAYFLAT-GLOSS",
            "GLOSS-FILM": "GLOSS-FILM"
        },
        "tab_paper_type": {
            "NONE": "NONE", "10PT_C2S": "10PT_C2S", "12PT_C2S": "12PT_C2S",
            "10PT_C1S": "10PT_C1S", "100_GLOSSTEXT": "100_GLOSS_TEXT"
        },
    }

    for col, mapping in mappings.items():
        df = normalize_column(df, col, mapping=mapping)

    if "text_paper_type" in df.columns:
        text_paper_mapping = {
            "NONE": "NONE",
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

    # Dates + sentinelle
    date_cols = ["expected_date", "delivery_date", "reception_date"]
    df = clean_dates(df, date_cols)
    df = replace_nat_with_sentinel_date(df, date_cols)

    # Imputation ML
    df = impute_cover_size_saddle_stitch(df)

    # Normalisation finale
    df = replace_nat_nan_none(df)

    # Rapports
    print("\n=== 20 premières lignes du DataFrame final ===")
    print(df.head(20).to_string(index=False))

    print_unique_values_summary(df)
    quality_check(df)

    print("=== PIPELINE COMPLET TERMINÉ ===")
    print(f"Forme finale : {df.shape}")
    return df


def save_processed(df: pd.DataFrame, filename: str = "pricing_fully_cleaned.xlsx"):
    root = get_project_root()
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"Fichier final sauvegardé → {out_path}")


if __name__ == "__main__":
    try:
        df_final = full_preprocessing("concateneRAHMA4.xlsx")
        save_processed(df_final, "pricing_fully_cleaned.xlsx")
        print("\nTraitement complet terminé avec succès !")
    except Exception as e:
        print("Erreur lors du pipeline :", str(e))