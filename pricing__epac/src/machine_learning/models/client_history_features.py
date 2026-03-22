"""
Feature engineering for client historical analysis
- Relative prices
- Price volatility
- Price elasticity
- Seniority and recency
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
# Ajouter le chemin racine pour trouver openssl_patch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pricing__epac import openssl_patch
import logging
from typing import Dict, List
import warnings
import joblib  # ADDED for joblib saving

warnings.filterwarnings('ignore')
# Désactiver pyOpenSSL dans urllib3 (même s'il n'est pas installé)
os.environ['URLLIB3_USE_PYOPENSSL'] = '0'
warnings.filterwarnings('ignore', module='urllib3.contrib.pyopenssl')
# Configuration pour MinIO (S3 compatible)
os.environ['AWS_ACCESS_KEY_ID'] = 'minio_admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio_password'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes price and VAT columns if they exist

    Args:
        df: DataFrame with 'unit_price' and possibly 'tva' columns

    Returns:
        DataFrame with normalized columns and HT prices
    """
    df = df.copy()

    # Check that unit_price column exists
    if 'unit_price' not in df.columns:
        logger.warning("Column 'unit_price' missing")
        df["unit_price_ht"] = 0
        return df

    # Normalize unit price
    df["unit_price"] = (
        df["unit_price"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)

    # Handle VAT if it exists
    if 'tva' in df.columns:
        df["tva"] = (
            df["tva"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["tva"] = pd.to_numeric(df["tva"], errors="coerce").fillna(0)
        df.loc[df["tva"] > 1, "tva"] = df["tva"] / 100

        # HT price with VAT
        df["unit_price_ht"] = df["unit_price"] / (1 + df["tva"].clip(lower=0))
    else:
        # No VAT, HT price = TTC price
        logger.info("Column 'tva' not found, using price as HT")
        df["unit_price_ht"] = df["unit_price"]

    logger.info(f"✅ Prices normalized - {len(df)} rows")

    return df


def compute_client_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes client indicators:
    - Relative price index
    - Price volatility

    Args:
        df: DataFrame with 'siren' and 'unit_price_ht' columns

    Returns:
        DataFrame with added indicators
    """
    df = df.copy()

    # Check required columns
    if 'siren' not in df.columns:
        logger.warning("siren column missing")
        df["relative_price_index"] = 1.0
        df["price_volatility"] = 0.0
        return df

    if 'unit_price_ht' not in df.columns:
        logger.warning("unit_price_ht column missing - creating with default value")
        df["unit_price_ht"] = 0
        df["relative_price_index"] = 1.0
        df["price_volatility"] = 0.0
        return df

    # Relative price index (client price / average client price)
    client_means = df.groupby("siren")["unit_price_ht"].transform("mean")
    df["relative_price_index"] = df["unit_price_ht"] / client_means.replace(0, np.nan)
    df["relative_price_index"] = df["relative_price_index"].fillna(1.0)

    # Price volatility by client
    df["price_volatility"] = df.groupby("siren")["unit_price_ht"].transform("std").fillna(0)

    logger.info("✅ Client indicators computed")

    return df


def aggregate_client_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates features by client

    Args:
        df: DataFrame with enriched data

    Returns:
        DataFrame aggregated by client
    """
    if 'siren' not in df.columns:
        logger.warning("siren column missing")
        return pd.DataFrame(columns=['siren'])

    # Check available columns
    available_cols = df.columns.tolist()
    logger.info(f"Columns available for aggregation: {available_cols}")

    # Build aggregation dictionary dynamically
    agg_dict = {}

    # unit_price_ht
    if 'unit_price_ht' in available_cols:
        agg_dict['unit_price_ht'] = ['count', 'mean', 'std']
        logger.info("✅ Added unit_price_ht to aggregation")

    # quantity
    if 'quantity' in available_cols:
        agg_dict['quantity'] = ['mean']
        logger.info("✅ Added quantity to aggregation")

    # price_volatility
    if 'price_volatility' in available_cols:
        agg_dict['price_volatility'] = 'mean'
        logger.info("✅ Added price_volatility to aggregation")

    # relative_price_index
    if 'relative_price_index' in available_cols:
        agg_dict['relative_price_index'] = 'mean'
        logger.info("✅ Added relative_price_index to aggregation")

    # reception_date
    if 'reception_date' in available_cols:
        agg_dict['reception_date'] = ['min', 'max']
        logger.info("✅ Added reception_date to aggregation")

    if not agg_dict:
        logger.warning("No columns available for aggregation")
        return pd.DataFrame(columns=['siren'])

    try:
        # Aggregation by client
        client_agg = df.groupby("siren").agg(agg_dict)

        # Flatten multi-index columns
        client_agg.columns = ['_'.join(col).strip() for col in client_agg.columns.values]
        client_agg = client_agg.reset_index()

        # Rename columns
        rename_map = {}
        if 'unit_price_ht_count' in client_agg.columns:
            rename_map['unit_price_ht_count'] = 'client_nb_orders'
        if 'unit_price_ht_mean' in client_agg.columns:
            rename_map['unit_price_ht_mean'] = 'client_avg_price_ht'
        if 'unit_price_ht_std' in client_agg.columns:
            rename_map['unit_price_ht_std'] = 'client_price_std_ht'
        if 'quantity_mean' in client_agg.columns:
            rename_map['quantity_mean'] = 'client_avg_quantity'
        if 'price_volatility_mean' in client_agg.columns:
            rename_map['price_volatility_mean'] = 'client_price_volatility'
        if 'relative_price_index_mean' in client_agg.columns:
            rename_map['relative_price_index_mean'] = 'client_relative_price'
        if 'reception_date_min' in client_agg.columns:
            rename_map['reception_date_min'] = 'client_first_order'
        if 'reception_date_max' in client_agg.columns:
            rename_map['reception_date_max'] = 'client_last_order'

        client_agg = client_agg.rename(columns=rename_map)

        logger.info(f"✅ Aggregation by client successful - {len(client_agg)} clients")
        logger.info(f"   Columns after aggregation: {list(client_agg.columns)}")

        return client_agg

    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['siren'])


def process_client_dates(client_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Processes client dates to compute seniority and recency

    Args:
        client_agg: DataFrame with 'client_first_order', 'client_last_order' columns

    Returns:
        DataFrame with temporal features
    """
    df = client_agg.copy()

    # Initialize default columns
    df['client_seniority_years'] = 0
    df['client_recency_days'] = 0

    # Check date columns
    date_cols_present = ['client_first_order' in df.columns, 'client_last_order' in df.columns]

    if not all(date_cols_present):
        logger.warning("Date columns missing - using default values")
        return df

    # Convert dates
    df["client_first_order"] = pd.to_datetime(
        df["client_first_order"], errors="coerce"
    )
    df["client_last_order"] = pd.to_datetime(
        df["client_last_order"], errors="coerce"
    )

    # Reference date (today or max of dates)
    if not df["client_last_order"].isna().all():
        today = df["client_last_order"].max()
    else:
        today = pd.Timestamp.now()

    # Seniority (years)
    seniority = (today - df["client_first_order"]).dt.days / 365
    df["client_seniority_years"] = seniority.round(2).fillna(0)

    # Recency (days)
    recency = (today - df["client_last_order"]).dt.days
    df["client_recency_days"] = recency.fillna(0).astype(int)

    logger.info(f"✅ Temporal features computed - Reference date: {today}")

    return df


def compute_price_elasticity(df: pd.DataFrame, min_samples: int = 8, min_cv: float = 0.05) -> pd.DataFrame:
    """
    Computes price elasticity by client via log-log regression

    Args:
        df: DataFrame with order data
        min_samples: Minimum number of observations per client
        min_cv: Minimum price coefficient of variation

    Returns:
        DataFrame with elasticities by client
    """
    # Check required columns
    required_cols = ['siren', 'unit_price_ht', 'quantity']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        logger.warning(f"Columns missing for elasticity calculation: {missing}")
        # Return DataFrame with default values
        unique_sirens = df['siren'].unique() if 'siren' in df.columns else []
        return pd.DataFrame({
            'siren': unique_sirens,
            'client_price_elasticity': 0.0,
            'elasticity_status': 'missing_columns'
        })

    # Filter valid data
    df_valid = df[(df['quantity'] > 0) & (df['unit_price_ht'] > 0)].copy()

    if df_valid.empty:
        logger.warning("No valid data for elasticity calculation")
        unique_sirens = df['siren'].unique() if 'siren' in df.columns else []
        return pd.DataFrame({
            'siren': unique_sirens,
            'client_price_elasticity': 0.0,
            'elasticity_status': 'no_valid_data'
        })

    elasticities = []

    for siren, g in df_valid.groupby("siren"):
        # Check minimum number of observations
        if len(g) < min_samples:
            elasticities.append({
                "siren": siren,
                "client_price_elasticity": 0.0,
                "elasticity_status": f"insufficient_samples_{len(g)}"
            })
            continue

        # Check sufficient price variation
        price_std = g["unit_price_ht"].std()
        price_mean = g["unit_price_ht"].mean()

        if price_mean == 0 or price_std == 0:
            elasticities.append({
                "siren": siren,
                "client_price_elasticity": 0.0,
                "elasticity_status": "zero_variance"
            })
            continue

        price_cv = price_std / price_mean if price_mean > 0 else 0
        if price_cv < min_cv:
            elasticities.append({
                "siren": siren,
                "client_price_elasticity": 0.0,
                "elasticity_status": f"low_price_variation_{price_cv:.3f}"
            })
            continue

        try:
            # Log-log regression: ln(quantity) ~ ln(price)
            X = np.log(g["unit_price_ht"].values.reshape(-1, 1))
            y = np.log(g["quantity"].values)

            # Check for infinite values
            if np.isinf(X).any() or np.isinf(y).any():
                raise ValueError("Infinite values in logs")

            model = LinearRegression().fit(X, y)
            elasticity = float(model.coef_[0])

            # R² to assess quality
            r2 = model.score(X, y)

            elasticities.append({
                "siren": siren,
                "client_price_elasticity": elasticity,
                "elasticity_r2": r2,
                "elasticity_status": "success",
                "elasticity_n_samples": len(g)
            })
        except Exception as e:
            elasticities.append({
                "siren": siren,
                "client_price_elasticity": 0.0,
                "elasticity_status": f"error_{str(e)[:50]}"
            })

    if not elasticities:
        unique_sirens = df['siren'].unique() if 'siren' in df.columns else []
        elasticity_df = pd.DataFrame({
            'siren': unique_sirens,
            'client_price_elasticity': 0.0,
            'elasticity_status': 'no_calculation'
        })
    else:
        elasticity_df = pd.DataFrame(elasticities)

    logger.info(f"✅ Elasticities computed - {len(elasticity_df)} clients")

    return elasticity_df


def create_client_features(
    df: pd.DataFrame,
    min_samples_elasticity: int = 8,
    min_price_cv: float = 0.05
) -> pd.DataFrame:
    """
    Complete pipeline for creating client features

    Args:
        df: Raw DataFrame with orders
        min_samples_elasticity: Minimum threshold for elasticity calculation
        min_price_cv: Minimum price coefficient of variation

    Returns:
        DataFrame with client features by siren
    """
    logger.info("=" * 60)
    logger.info("🚀 CREATING CLIENT FEATURES")
    logger.info("=" * 60)

    # Display available columns
    logger.info(f"Available columns: {list(df.columns)}")

    # Check for siren presence
    if 'siren' not in df.columns:
        logger.error("❌ 'siren' column missing - cannot create client features")
        return pd.DataFrame()

    # Step 1: Normalization
    df_norm = normalize_price_columns(df)

    # Step 2: Client indicators
    df_with_indicators = compute_client_indicators(df_norm)

    # Step 3: Aggregation by client
    client_agg = aggregate_client_features(df_with_indicators)

    if client_agg.empty:
        logger.warning("⚠️ No aggregated client features")
        # Return at least the sirens
        unique_sirens = df['siren'].unique()
        return pd.DataFrame({'siren': unique_sirens})

    # Step 4: Temporal features
    client_with_dates = process_client_dates(client_agg)

    # Step 5: Price elasticity
    elasticity_df = compute_price_elasticity(
        df_with_indicators,
        min_samples=min_samples_elasticity,
        min_cv=min_price_cv
    )

    # Step 6: Merge
    if not elasticity_df.empty:
        client_features = client_with_dates.merge(
            elasticity_df[['siren', 'client_price_elasticity', 'elasticity_status']],
            on="siren",
            how="left"
        )
    else:
        client_features = client_with_dates.copy()
        client_features['client_price_elasticity'] = 0.0
        client_features['elasticity_status'] = 'not_calculated'

    # Step 7: Fill NaNs
    fill_cols = ['client_price_volatility', 'client_price_std_ht', 'client_relative_price',
                 'client_price_elasticity', 'client_seniority_years', 'client_recency_days']
    for col in fill_cols:
        if col in client_features.columns:
            client_features[col] = client_features[col].fillna(0)

    # Replace infinities
    client_features = client_features.replace([np.inf, -np.inf], 0)

    logger.info(f"✅ Final client features - {len(client_features)} clients")
    logger.info(f"📊 Columns: {list(client_features.columns)}")

    return client_features


def add_client_features_to_orders(
    orders_df: pd.DataFrame,
    client_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds client features to each order

    Args:
        orders_df: Orders DataFrame
        client_features_df: Client features DataFrame

    Returns:
        Enriched orders DataFrame
    """
    if client_features_df.empty or 'siren' not in client_features_df.columns:
        logger.warning("Empty client features, returning original data")
        return orders_df

    if 'siren' not in orders_df.columns:
        logger.warning("siren column missing from orders")
        return orders_df

    # Merge with orders
    enriched_df = orders_df.merge(
        client_features_df,
        on="siren",
        how="left"
    )

    # Fill NaN for clients without features
    client_feature_cols = [col for col in client_features_df.columns if col != 'siren']
    for col in client_feature_cols:
        if col in enriched_df.columns:
            enriched_df[col] = enriched_df[col].fillna(0)

    logger.info(f"✅ Client features added to orders - {len(enriched_df)} rows")

    return enriched_df


# ============================================================
# NEW FUNCTIONS FOR JOBLIB SAVING
# ============================================================

def save_client_features_joblib(
    client_features: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    Saves client features in joblib format (faster and preserves types)

    Args:
        client_features: Client features DataFrame
        output_path: Output path (without extension, .joblib will be added)

    Returns:
        Path to saved file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add .joblib extension if needed
    if not str(output_path).endswith('.joblib'):
        joblib_path = output_path.with_suffix('.joblib')
    else:
        joblib_path = output_path

    # Save as joblib
    joblib.dump(client_features, joblib_path)

    logger.info(f"✅ Client features saved (joblib): {joblib_path}")
    logger.info(f"   Size: {joblib_path.stat().st_size / 1024:.2f} KB")

    return joblib_path


def load_client_features_joblib(file_path: Path) -> pd.DataFrame:
    """
    Loads client features from a joblib file

    Args:
        file_path: Path to joblib file

    Returns:
        Client features DataFrame
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()

    try:
        df = joblib.load(file_path)
        logger.info(f"✅ Client features loaded (joblib): {file_path} - {len(df)} clients")
        return df
    except Exception as e:
        logger.error(f"Error loading joblib: {e}")
        return pd.DataFrame()


def save_client_features_all_formats(
    client_features: pd.DataFrame,
    base_path: Path
) -> Dict[str, Path]:
    """
    Saves client features in all formats (Excel, CSV, joblib)

    Args:
        client_features: Client features DataFrame
        base_path: Base path (without extension)

    Returns:
        Dictionary of paths by format
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    # Excel
    excel_path = base_path.with_suffix('.xlsx')
    client_features.to_excel(excel_path, index=False)
    saved_paths['excel'] = excel_path
    logger.info(f"✅ Excel: {excel_path}")

    # CSV
    csv_path = base_path.with_suffix('.csv')
    client_features.to_csv(csv_path, index=False)
    saved_paths['csv'] = csv_path
    logger.info(f"✅ CSV: {csv_path}")

    # Joblib (recommended for reuse)
    joblib_path = base_path.with_suffix('.joblib')
    joblib.dump(client_features, joblib_path)
    saved_paths['joblib'] = joblib_path
    logger.info(f"✅ Joblib: {joblib_path} ({joblib_path.stat().st_size / 1024:.2f} KB)")

    return saved_paths


# ============================================================
# MODIFIED ORIGINAL FUNCTIONS
# ============================================================

def save_client_features(
    client_features: pd.DataFrame,
    output_path: Path,
    save_formats: List[str] = ['excel', 'csv', 'joblib']  # Default: all formats
) -> Dict[str, Path]:
    """
    Saves client features in different formats

    Args:
        client_features: Client features DataFrame
        output_path: Base path (without extension)
        save_formats: List of formats to save ('excel', 'csv', 'joblib')

    Returns:
        Dictionary of paths by format
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    base_path = output_path.with_suffix('')  # Remove extension if present

    if 'excel' in save_formats:
        excel_path = base_path.with_suffix('.xlsx')
        client_features.to_excel(excel_path, index=False)
        saved_paths['excel'] = excel_path
        logger.info(f"✅ Excel: {excel_path}")

    if 'csv' in save_formats:
        csv_path = base_path.with_suffix('.csv')
        client_features.to_csv(csv_path, index=False)
        saved_paths['csv'] = csv_path
        logger.info(f"✅ CSV: {csv_path}")

    if 'joblib' in save_formats:
        joblib_path = base_path.with_suffix('.joblib')
        joblib.dump(client_features, joblib_path)
        saved_paths['joblib'] = joblib_path
        file_size = joblib_path.stat().st_size / 1024
        logger.info(f"✅ Joblib: {joblib_path} ({file_size:.2f} KB)")

    return saved_paths


def load_client_features(file_path: Path) -> pd.DataFrame:
    """
    Loads client features from a file (Excel, CSV, or joblib)

    Args:
        file_path: Path to file

    Returns:
        Client features DataFrame
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()

    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            logger.info(f"✅ CSV loaded: {file_path}")
        elif file_path.suffix == '.joblib':
            df = joblib.load(file_path)
            logger.info(f"✅ Joblib loaded: {file_path}")
        else:  # default excel
            df = pd.read_excel(file_path)
            logger.info(f"✅ Excel loaded: {file_path}")

        logger.info(f"   {len(df)} clients, {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()