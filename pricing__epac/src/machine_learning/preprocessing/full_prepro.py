#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EPAC PRICING DATA PREPROCESSING PIPELINE
================================================================================

A comprehensive, production-ready preprocessing pipeline that transforms raw EPAC
pricing data into clean, normalized, and model-ready datasets.

This pipeline handles:
- Initial data cleaning (column selection, flag creation, basic imputation)
- Advanced cleaning (duplicate removal, constant column detection, value mapping)
- Date normalization with sentinel values (31-12-9999)
- Machine learning-based imputation for specific features
- Final normalization and quality reporting

================================================================================
PROCESSING FLOWCHART
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │                         INPUT                                         │
    │              Consolidated Excel/CSV file                             │
    │         (dataset_complet.xlsx or similar)                            │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 1: LOAD DATA                                                     │
    │ - Validate file existence and size                                    │
    │ - Auto-detect format (Excel/CSV)                                      │
    │ - Log initial shape                                                   │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 2: INITIAL CLEANING                                              │
    │ ┌────────────────────────────────────────────────────────────────┐   │
    │ │ 2.1: Column Selection                                          │   │
    │ │     - Keep only relevant columns for pricing analysis          │   │
    │ │     - Drop irrelevant or redundant columns                     │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 2.2: ISBN Processing                                           │   │
    │ │     - Convert ISBN10/ISBN13 to binary flags (0/1)              │   │
    │ │     - Handle missing values                                     │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 2.3: Flag Creation (has_*)                                     │   │
    │ │     - has_insert, has_tab, has_backcover, has_coil             │   │
    │ │     - Based on presence of component columns                   │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 2.4: Missing Value Imputation (Basic)                          │   │
    │ │     - Categorical → "MISSING"                                  │   │
    │ │     - Numeric → -1 or 0 depending on context                   │   │
    │ │     - Version → 1                                              │   │
    │ └────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 3: ADVANCED CLEANING                                            │
    │ ┌────────────────────────────────────────────────────────────────┐   │
    │ │ 3.1: Duplicate Removal                                         │   │
    │ │     - Remove completely duplicate rows                         │   │
    │ │     - Log number removed                                       │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 3.2: Constant Column Detection                                 │   │
    │ │     - Identify columns with single value                       │   │
    │ │     - Drop if not in preservation list                         │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 3.3: Value Mapping                                             │   │
    │ │     - Load mappings from YAML configuration                    │   │
    │ │     - Normalize categorical values                             │   │
    │ │     - Apply specific text_paper_type mapping                   │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 3.4: String Normalization                                      │   │
    │ │     - Convert all strings to uppercase                         │   │
    │ │     - Strip whitespace                                         │   │
    │ └────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 4: DATE PROCESSING                                              │
    │ ┌────────────────────────────────────────────────────────────────┐   │
    │ │ 4.1: Date Parsing                                              │   │
    │ │     - Use dateutil.parser for robust parsing                   │   │
    │ │     - Fallback to pandas to_datetime                           │   │
    │ │     - Log invalid dates                                        │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 4.2: Date Formatting                                           │   │
    │ │     - Standardize to "DD-MM-YYYY" format                       │   │
    │ │     - Handle NaT values                                        │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 4.3: Sentinel Date Replacement                                 │   │
    │ │     - Replace NaT with sentinel (31-12-9999)                   │   │
    │ │     - Ensures no missing dates in output                       │   │
    │ └────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 5: ML-BASED IMPUTATION                                          │
    │ ┌────────────────────────────────────────────────────────────────┐   │
    │ │ 5.1: Model Loading                                             │   │
    │ │     - Load cover_size prediction pipeline                      │   │
    │ │     - Fallback to mode imputation if model missing             │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 5.2: Feature Selection                                         │   │
    │ │     - Numeric: width, height, weight, production_page, thickness│   │
    │ │     - Categorical: cover_color, cover_paper_type, text_color,  │   │
    │ │                   priority_level                               │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 5.3: Prediction                                                │   │
    │ │     - Apply only to Saddle Stitch bindings                     │   │
    │ │     - Predict missing cover_size values                        │   │
    │ └────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 6: FINAL NORMALIZATION                                          │
    │ - Replace all remaining NaN/None with "MISSING" (except dates)      │
    │ - Final string normalization (uppercase, strip)                     │
    │ - Ensure consistent data types                                      │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 7: QUALITY REPORTING & EXPORT                                   │
    │ ┌────────────────────────────────────────────────────────────────┐   │
    │ │ 7.1: Quality Metrics                                           │   │
    │ │     - Shape, missing values, column types                      │   │
    │ │     - Numerical statistics                                     │   │
    │ │     - Unique value summary (if verbose)                        │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 7.2: Save Processed Data                                       │   │
    │ │     - Excel format with timestamp                              │   │
    │ │     - Save to data/processed/ directory                        │   │
    │ ├────────────────────────────────────────────────────────────────┤   │
    │ │ 7.3: Cleanup                                                   │   │
    │ │     - Remove temporary SQL dump files                          │   │
    │ │     - Log completion                                           │   │
    │ └────────────────────────────────────────────────────────────────┘   │
    └────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         OUTPUT                                        │
    │              Cleaned, normalized dataset                             │
    │         (pricing_fully_cleaned_YYYYMMDD_HHMMSS.xlsx)                 │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
KEY FEATURES
================================================================================

✓ Modular Architecture     - Each step in separate, testable functions
✓ Robust Error Handling    - Graceful fallbacks, comprehensive logging
✓ ML-Based Imputation      - Intelligent cover_size prediction for saddle stitch
✓ Date Normalization       - Handles corrupted dates with multiple strategies
✓ Sentinel Date            - NaT replaced with 31-12-9999
✓ Configurable Mappings    - YAML-based value normalization
✓ Quality Reporting        - Detailed metrics and unique value summaries
✓ Memory Efficient         - In-place operations where possible
✓ Production Ready         - Comprehensive logging, error handling, validation

================================================================================
ENVIRONMENT VARIABLES
================================================================================

Optional:
    PROJECT_ROOT           - Override project root directory
    LOG_LEVEL              - Logging level (DEBUG, INFO, WARNING, ERROR)

================================================================================
USAGE EXAMPLES
================================================================================

Basic usage:
    from full_preprocess import full_preprocessing, save_processed
    from pathlib import Path

    input_path = Path("data/consolidated/dataset_complet.xlsx")
    df = full_preprocessing(input_path, verbose=True)
    save_processed(df)

Command line:
    python full_preprocess.py --verbose

With custom input:
    python full_preprocess.py --input custom_data.xlsx

================================================================================
"""

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import joblib
import pandas as pd
import yaml
from dateutil import parser
from pricing__epac.src.shared.logging import configure_logging


# ========== CONFIGURATION ==========
@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Missing value handling
    missing_value: str = "MISSING"
    sentinel_date_str: str = "31-12-9999"  # Sentinel date format DD-MM-YYYY
    sentinel_date_timestamp: str = "9999-12-31"  # Timestamp format for pandas
    max_unique_display: int = 10

    # Column categories
    date_columns: List[str] = field(default_factory=lambda:
    ["expected_date", "delivery_date", "reception_date"])
    isbn_columns: List[str] = field(default_factory=lambda: ["isbn10", "isbn13"])
    size_numeric_columns: List[str] = field(default_factory=lambda:
    ["insert_size", "tab_size", "back_cover_flat_size", "trim_size"])
    size_categorical_columns: List[str] = field(default_factory=lambda: ["cover_size"])
    numeric_zero_columns: List[str] = field(default_factory=lambda:
    ["three_hole_drill", "perf", "double_sided_cover", "security_label", "shrinkwrap"])

    # Columns to preserve even if constant
    constant_columns_to_keep: List[str] = field(default_factory=lambda: [
        'has_coil', 'has_insert', 'has_tab', 'has_backcover',
        'insert_paper_type', 'unit_price', 'tva'
    ])

    # Required columns validation
    required_columns: List[str] = field(default_factory=lambda: ['binding_type', 'unit_price'])

    # Empty values to replace
    empty_values: List[str] = field(default_factory=lambda:
    ["NAN", "NONE", "MISSING", "N/A", "", " ", "NaT", "<NA>", "NULL"])

    # ML imputation configuration
    ml_features_numeric: List[str] = field(default_factory=lambda:
    ["width", "height", "weight", "production_page", "thickness"])
    ml_features_categorical: List[str] = field(default_factory=lambda:
    ["cover_color", "cover_paper_type", "text_color", "priority_level"])
    ml_model_path: Optional[Path] = field(default=None)
    ml_binding_type: str = "SS"  # Saddle Stitch

    # File size limits (MB)
    max_file_size_mb: int = 500

    @property
    def sentinel_date(self) -> pd.Timestamp:
        """Return sentinel date as pandas Timestamp"""
        return pd.Timestamp(self.sentinel_date_timestamp)

    def __post_init__(self):
        if self.ml_model_path is None:
            self.ml_model_path = get_project_root() / "pricing__epac" / "src" / "machine_learning" / "training"


# ========== LOGGING SETUP ==========
def setup_logging(log_level: str = 'INFO', log_file: Optional[Path] = None):
    """Configure logging with environment variable support"""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    effective_level = os.getenv('LOG_LEVEL', log_level).upper()
    return configure_logging(level=effective_level, log_file=log_file, logger_name=__name__)


# ========== PATH CONFIGURATION ==========
def get_project_root() -> Path:
    """Return the project root directory."""
    env_root = os.getenv('PROJECT_ROOT')
    if env_root:
        return Path(env_root)

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "pricing__epac").exists():
            return parent

    return current.parents[4]


def get_package_root() -> Path:
    """Return the package root directory."""
    return get_project_root() / "pricing__epac"


def get_data_root() -> Path:
    """Return the package data directory used by the pipeline."""
    return get_package_root() / "data"


# Load configuration
PROJECT_ROOT = get_project_root()
CONFIG = PreprocessingConfig()

# Setup logging
logger = setup_logging()


# ========== DATA LOADING ==========
def load_data(input_path: Path, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """
    Load data with validation and type checking.

    Args:
        input_path: Path to input file (Excel or CSV)
        config: Preprocessing configuration

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or format unsupported
    """
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Check file size
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    if file_size_mb > config.max_file_size_mb:
        raise ValueError(f"File too large: {file_size_mb:.2f} MB > {config.max_file_size_mb} MB")

    if input_path.stat().st_size == 0:
        raise ValueError(f"Empty file: {input_path}")

    logger.info(f"Loading: {input_path} ({file_size_mb:.2f} MB)")

    try:
        if input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path, engine="openpyxl")
        elif input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise

    logger.info(f"Raw shape: {df.shape}")
    return df


# ========== MAPPING MANAGEMENT ==========
def load_mappings(config: PreprocessingConfig = CONFIG) -> Dict[str, Dict[str, str]]:
    """
    Load value mappings from YAML configuration file.

    Args:
        config: Preprocessing configuration

    Returns:
        Dictionary of column mappings
    """
    mappings_path = get_package_root() / "src" / "config" / "mappings.yaml"

    if not mappings_path.exists():
        logger.warning(f"Mapping file not found: {mappings_path}")
        return {}

    try:
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = yaml.safe_load(f)

        # Clean mappings: convert all keys to strings
        cleaned_mappings = {}
        for col, mapping in mappings.items():
            cleaned_mappings[col] = {str(k): v for k, v in mapping.items()}

        logger.info(f"Mappings loaded from {mappings_path}")
        return cleaned_mappings

    except Exception as e:
        logger.error(f"Error loading mappings: {e}")
        return {}


def validate_mappings(mappings: Dict[str, Dict[str, str]]) -> bool:
    """Validate mapping structure before applying"""
    for col, mapping in mappings.items():
        if not isinstance(mapping, dict):
            logger.error(f"Invalid mapping for {col}: not a dict")
            return False
        for key, value in mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                logger.error(f"Invalid mapping entry for {col}: {key} -> {value}")
                return False
    return True


# ========== FLAG CREATION ==========
def safe_create_flag(
        df: pd.DataFrame,
        source_cols: List[str],
        flag_name: str,
        config: PreprocessingConfig = CONFIG
) -> pd.DataFrame:
    """
    Create a has_* flag safely.

    Args:
        df: Source DataFrame
        source_cols: List of source columns
        flag_name: Name of flag to create
        config: Preprocessing configuration

    Returns:
        DataFrame with flag added
    """
    existing_cols = [col for col in source_cols if col in df.columns]

    if existing_cols:
        df[flag_name] = df[existing_cols].notna().any(axis=1).astype(int)
        logger.debug(f"Flag {flag_name} created from {existing_cols}")
    else:
        df[flag_name] = 0
        logger.debug(f"Columns {source_cols} missing, {flag_name} set to 0")

        for col in source_cols:
            if col not in df.columns:
                df[col] = config.missing_value
                logger.debug(f"Created column {col} with default value")

    return df


# ========== COLUMN VALIDATION ==========
def validate_required_columns(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> None:
    """Validate that all required columns are present."""
    missing_cols = [col for col in config.required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.debug("Required columns validation: OK")


# ========== INITIAL CLEANING ==========
def initial_cleaning(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """
    Initial cleaning: column selection, flag creation, basic imputation.

    Args:
        df: Source DataFrame
        config: Preprocessing configuration

    Returns:
        Cleaned DataFrame
    """
    logger.info("=== Step 1: Initial cleaning started ===")

    # Column selection
    cols_to_keep = [
        'binding_type', 'height', 'isbn10', 'isbn13', 'perf', 'production_page',
        'security_label', 'self_cover', 'shrinkwrap', 'status', 'thickness',
        'three_hole_drill', 'unit_price', 'version', 'weight', 'width',
        'label_location', 'label_type', 'cover_finish_type', 'text_color',
        'text_paper_type', 'priority_level', 'quantity', 'quantity_min',
        'quantity_max', 'siren', 'coil_type', 'cover_paper_type',
        'double_sided_cover', 'cover_color', 'cover_size', 'insert_lamination',
        'insert_paper_type', 'insert_color', 'insert_size', 'tab_page_number',
        'trim_size', 'tab_color', 'tab_lamination', 'tab_size', 'tab_paper_type',
        'case_finish_type', 'case_paper_type', 'cover_case_color',
        'back_cover_flat_size', 'spine_type', 'head_and_tail', 'tva', 'reception_date'
    ]

    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].copy()
    logger.info(f"Columns kept: {len(existing_cols)} / {len(cols_to_keep)}")

    # ISBN processing (convert to binary flags)
    for col in config.isbn_columns:
        if col in df.columns:
            df[col] = (df[col].notna() & (df[col].astype(str).str.strip() != "")).astype(int)

    # TVA: set missing to 0
    if 'tva' in df.columns:
        df['tva'] = df['tva'].fillna(0)

    # Flag creation
    df = safe_create_flag(df, ['insert_lamination', 'insert_paper_type', 'insert_color', 'insert_size'], 'has_insert')
    df = safe_create_flag(df, ['tab_page_number', 'tab_color', 'tab_lamination', 'tab_size', 'tab_paper_type'],
                          'has_tab')
    df = safe_create_flag(df, ['case_finish_type', 'case_paper_type', 'cover_case_color', 'back_cover_flat_size',
                               'spine_type'], 'has_backcover')
    df = safe_create_flag(df, ['coil_type'], 'has_coil')

    # Numeric size columns → -1
    for col in config.size_numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # Categorical size columns → MISSING_VALUE
    for col in config.size_categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(config.missing_value)

    # Categorical columns → MISSING_VALUE
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
            df[col] = df[col].fillna(config.missing_value)

    # Numeric zero columns → 0
    for col in config.numeric_zero_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Version → 1
    if 'version' in df.columns:
        df['version'] = df['version'].fillna(1)

    validate_required_columns(df, config)

    logger.info(f"Step 1 completed. Shape: {df.shape}")
    return df


# ========== ADVANCED CLEANING ==========
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    initial = len(df)
    df = df.drop_duplicates()
    removed = initial - len(df)
    if removed > 0:
        logger.info(f"Duplicates removed: {removed:,} rows")
    return df


def drop_constant_columns(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """Drop constant columns except those specified."""
    all_constants = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    constants_to_drop = [col for col in all_constants if col not in config.constant_columns_to_keep]

    if constants_to_drop:
        logger.info(f"Constant columns dropped: {constants_to_drop}")
        constants_preserved = [col for col in all_constants if col in config.constant_columns_to_keep]
        if constants_preserved:
            logger.info(f"Constant columns preserved: {constants_preserved}")
        df = df.drop(columns=constants_to_drop)

    return df


def uppercase_string_columns(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        config: PreprocessingConfig = CONFIG
) -> pd.DataFrame:
    """
    Convert string columns to uppercase.

    Args:
        df: Source DataFrame
        columns: Specific columns to convert (None = all string columns)
        config: Preprocessing configuration

    Returns:
        DataFrame with uppercase strings
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

    for col in columns:
        if col in df.columns and (df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])):
            df[col] = df[col].astype(str).str.strip().str.upper()

    logger.debug(f"Columns uppercased: {len(columns)} columns")
    return df


def normalize_column(
        df: pd.DataFrame,
        col: str,
        mapping: Optional[Dict[str, str]] = None,
        default: Optional[str] = None,
        replace_empty: bool = True,
        config: PreprocessingConfig = CONFIG
) -> pd.DataFrame:
    """
    Normalize a column with optional mapping.

    Args:
        df: Source DataFrame
        col: Column name to normalize
        mapping: Optional value mapping dictionary
        default: Default value for empty values
        replace_empty: Whether to replace empty values
        config: Preprocessing configuration

    Returns:
        DataFrame with normalized column
    """
    if col not in df.columns:
        return df

    default = default or config.missing_value

    s = df[col].astype(str)
    if replace_empty:
        s = s.replace(config.empty_values, default)
    if mapping:
        s = s.replace(mapping)

    df[col] = s
    logger.debug(f"{col:24} → {df[col].value_counts(dropna=False).head(10).to_dict()}")
    return df


# ========== DATE PROCESSING ==========
def fix_corrupted_dates(series: pd.Series, col_name: Optional[str] = None) -> pd.Series:
    """
    Fix corrupted dates using dateutil.parser.

    Args:
        series: Pandas series with dates
        col_name: Column name for logging

    Returns:
        Series with parsed dates
    """

    def safe_parse(date_str):
        if pd.isna(date_str) or date_str in ['', ' ', 'NONE', 'MISSING']:
            return pd.NaT
        try:
            return pd.to_datetime(parser.parse(str(date_str), fuzzy=False))
        except (ValueError, TypeError, OverflowError):
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except Exception:
                return pd.NaT

    result = series.apply(safe_parse)
    invalid_count = result.isna().sum() - series.isna().sum()
    if invalid_count > 0 and col_name:
        logger.warning(f"{col_name}: {invalid_count} dates could not be parsed")

    return result


def clean_dates(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """Clean and standardize date columns."""
    for col in config.date_columns:
        if col not in df.columns:
            continue

        df[col] = fix_corrupted_dates(df[col], col_name=col)
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].dt.strftime("%d-%m-%Y").where(df[col].notna(), df[col])

    return df


def replace_nat_with_sentinel_date(
        df: pd.DataFrame,
        config: PreprocessingConfig = CONFIG
) -> pd.DataFrame:
    """
    Replace NaT dates with sentinel date (31-12-9999).

    Args:
        df: Source DataFrame
        config: Preprocessing configuration

    Returns:
        DataFrame with sentinel dates
    """
    sentinel_str = config.sentinel_date_str  # "31-12-9999"

    for col in config.date_columns:
        if col in df.columns:
            nat_count = df[col].isna().sum()
            if nat_count > 0:
                df[col] = df[col].fillna(sentinel_str)
                logger.info(f"{col}: {nat_count} NaT replaced with sentinel date ({sentinel_str})")

    return df


def validate_dates(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """
    Validate dates are within reasonable range.

    Args:
        df: Source DataFrame
        config: Preprocessing configuration

    Returns:
        DataFrame with validated dates
    """
    for col in config.date_columns:
        if col not in df.columns:
            continue

        # Convert to datetime for validation
        df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors='coerce')

        # Check for future dates beyond sentinel
        future_dates = df[df[col] > config.sentinel_date][col]
        if not future_dates.empty:
            logger.warning(f"{col}: {len(future_dates)} dates beyond sentinel date")

        # Check for very old dates (before 1900)
        old_dates = df[df[col] < '1900-01-01'][col]
        if not old_dates.empty:
            logger.warning(f"{col}: {len(old_dates)} dates before 1900")

        # Convert back to string format
        df[col] = df[col].dt.strftime("%d-%m-%Y").where(df[col].notna(), df[col])

    return df


# ========== ML IMPUTATION ==========
_model_cache = {}


def load_ml_model(model_path: Path):
    """Load ML model with caching."""
    if model_path not in _model_cache:
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        try:
            _model_cache[model_path] = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    return _model_cache[model_path]


def impute_cover_size_saddle_stitch(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """
    Impute cover_size for saddle stitch bindings using ML model.

    Args:
        df: Source DataFrame
        config: Preprocessing configuration

    Returns:
        DataFrame with imputed cover_size
    """

    # Fallback: impute with mode
    def fallback_imputation(df_subset):
        if 'cover_size' in df_subset.columns:
            mode_value = df_subset[df_subset['binding_type'] == config.ml_binding_type]['cover_size'].mode()
            if not mode_value.empty:
                mask_ss = (df["binding_type"] == config.ml_binding_type) & (df["cover_size"] == config.missing_value)
                df.loc[mask_ss, "cover_size"] = mode_value.iloc[0]
                logger.info(f"Fallback imputation with mode: {mode_value.iloc[0]}")
        return df

    model_path = config.ml_model_path / "cover_size_pipeline.pkl"
    pipeline = load_ml_model(model_path)

    if pipeline is None:
        logger.warning("ML model not available, using fallback imputation")
        return fallback_imputation(df)

    # Identify rows to impute
    mask_ss_to_impute = (
            (df["binding_type"] == config.ml_binding_type) &
            (df["cover_size"].isin([config.missing_value, "SDL", "", pd.NA, None]))
    )

    if not mask_ss_to_impute.any():
        logger.info("No saddle stitch rows to impute for cover_size")
        return df

    # Prepare features
    available_num = [f for f in config.ml_features_numeric if f in df.columns]
    available_cat = [f for f in config.ml_features_categorical if f in df.columns]

    if not available_num and not available_cat:
        logger.warning("No features available for imputation")
        return fallback_imputation(df)

    # Create feature matrix
    df_to_impute = df.loc[mask_ss_to_impute].copy()
    X = df_to_impute[available_num + available_cat]

    try:
        preds = pipeline.predict(X)
        df.loc[mask_ss_to_impute, "cover_size"] = preds
        logger.info(f"ML imputation completed: {len(preds)} values imputed")
    except Exception as e:
        logger.error(f"Error during ML imputation: {e}")
        return fallback_imputation(df)

    return df


# ========== FINAL NORMALIZATION ==========
def final_normalization(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """
    Final normalization: replace NaN/None with MISSING_VALUE.

    Args:
        df: Source DataFrame
        config: Preprocessing configuration

    Returns:
        Normalized DataFrame
    """
    for col in df.columns:
        if col in config.date_columns:
            continue

        # Replace NaN with MISSING_VALUE
        df[col] = df[col].fillna(config.missing_value)

        # For string columns, uppercase and strip
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.upper()

    logger.info("Final normalization: NaN/None replaced with 'MISSING' (except dates)")
    return df


# ========== QUALITY REPORTING ==========
def print_unique_values_summary(df: pd.DataFrame, config: PreprocessingConfig = CONFIG):
    """Print summary of unique values per column."""
    print("\n" + "=" * 80)
    print("UNIQUE VALUES SUMMARY (top 10 + total count)")
    print("=" * 80)

    for col in df.columns:
        unique_vals = df[col].value_counts(dropna=False)
        total_unique = len(unique_vals)

        print(f"\nColumn: {col} ({df[col].dtype})")
        print(f" - Total unique values: {total_unique}")

        if total_unique == 0:
            print(" → Empty column")
            continue

        if total_unique == 1:
            print(f" → Single value: {unique_vals.index[0]} ({unique_vals.iloc[0]} times)")
            continue

        top_n = unique_vals.head(config.max_unique_display)
        for val, count in top_n.items():
            print(f" - {val!r:30} : {count:,} ({count / len(df):.1%})")

        if total_unique > config.max_unique_display:
            print(f" ... and {total_unique - config.max_unique_display} other values")

    print("=" * 80)


def quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate quality report and return metrics.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with quality metrics
    """
    print("\n=== FINAL QUALITY REPORT ===")
    print(f"Shape: {df.shape}")
    print("\nColumn types:")
    print(df.dtypes)

    missing = df.isna().sum()
    if missing.sum() > 0:
        print("\nRemaining missing values:")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("No missing values → OK")

    print("\nNumerical statistics:")
    print(df.describe().round(2))

    return {
        'shape': df.shape,
        'missing_count': missing.sum(),
        'missing_by_column': missing[missing > 0].to_dict(),
        'column_types': df.dtypes.astype(str).to_dict()
    }


def save_quality_metrics(metrics: Dict[str, Any], output_path: Optional[Path] = None):
    """Save quality metrics to JSON file."""
    import json

    if output_path is None:
        output_path = get_data_root() / "processed" / "quality_metrics.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Quality metrics saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save quality metrics: {e}")


# ========== MAPPING APPLICATION ==========
def apply_all_mappings(df: pd.DataFrame, config: PreprocessingConfig = CONFIG) -> pd.DataFrame:
    """
    Apply all value mappings to the DataFrame.

    Args:
        df: Source DataFrame
        config: Preprocessing configuration

    Returns:
        DataFrame with applied mappings
    """
    mappings = load_mappings(config)

    if not validate_mappings(mappings):
        logger.warning("Mapping validation failed, skipping")
        return df

    for col, mapping in mappings.items():
        df = normalize_column(df, col, mapping=mapping, config=config)

    # Specific mapping for text_paper_type
    if "text_paper_type" in df.columns:
        text_paper_mapping = {
            "NONE": config.missing_value,
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
        df = normalize_column(df, "text_paper_type", mapping=text_paper_mapping, config=config)

    return df


# ========== MAIN PIPELINE ==========
def full_preprocessing(
        input_path: Path,
        verbose: bool = False,
        config: PreprocessingConfig = CONFIG
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.

    Args:
        input_path: Path to input file (Excel or CSV)
        verbose: If True, display detailed reports
        config: Preprocessing configuration

    Returns:
        Preprocessed DataFrame

    Example:
        >>> from pathlib import Path
        >>> df = full_preprocessing(Path("data/consolidated/dataset_complet.xlsx"))
        >>> print(f"Preprocessed shape: {df.shape}")
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("🚀 COMPLETE PREPROCESSING PIPELINE STARTED")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\n📥 STEP 1: Loading data")
    df = load_data(input_path, config)

    # Step 2: Initial cleaning
    logger.info("\n🧹 STEP 2: Initial cleaning")
    df = initial_cleaning(df, config)

    # Step 3: Advanced cleaning
    logger.info("\n🔧 STEP 3: Advanced cleaning")
    df = remove_duplicates(df)
    df = drop_constant_columns(df, config)

    # Step 4: Uppercase strings first
    logger.info("\n🔠 STEP 4: String normalization")
    df = uppercase_string_columns(df, config=config)

    # Step 5: Apply mappings
    logger.info("\n📋 STEP 5: Applying value mappings")
    df = apply_all_mappings(df, config)

    # Step 6: Date processing
    logger.info("\n📅 STEP 6: Date processing")
    df = clean_dates(df, config)
    df = replace_nat_with_sentinel_date(df, config)  # Replace NaT with 31-12-9999
    df = validate_dates(df, config)

    # Step 7: ML imputation
    logger.info("\n🤖 STEP 7: ML-based imputation")
    df = impute_cover_size_saddle_stitch(df, config)

    # Step 8: Final normalization
    logger.info("\n✨ STEP 8: Final normalization")
    df = final_normalization(df, config)

    # Step 9: Quality reporting (if verbose)
    if verbose:
        logger.info("\n📊 STEP 9: Quality reporting")
        print("\n=== 20 FIRST ROWS OF FINAL DATAFRAME ===")
        print(df.head(20).to_string(index=False))
        print_unique_values_summary(df, config)
        metrics = quality_check(df)
        save_quality_metrics(metrics)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ PREPROCESSING COMPLETED in {elapsed:.2f} seconds")
    logger.info(f"📊 Final shape: {df.shape}")
    logger.info("=" * 60)

    return df


def save_processed(
        df: pd.DataFrame,
        filename: Optional[str] = None,
        config: PreprocessingConfig = CONFIG
) -> Path:
    """
    Save processed DataFrame.

    Args:
        df: DataFrame to save
        filename: Optional filename (auto-generated if None)
        config: Preprocessing configuration

    Returns:
        Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pricing_fully_cleaned_{timestamp}.xlsx"

    out_dir = get_data_root() / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    df.to_excel(out_path, index=False, engine="openpyxl")
    logger.info(f"Final file saved → {out_path}")

    return out_path


def cleanup_temp_files():
    """Clean up temporary files."""
    sql_dumps_path = get_data_root() / "raw" / "dumps" / "sql"

    if sql_dumps_path.exists():
        for path in sql_dumps_path.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        logger.info(f"Temporary SQL dump folder cleaned: {sql_dumps_path}")


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description='EPAC Pricing Data Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Use default input path
    %(prog)s --input custom_data.xlsx     # Use custom input file
    %(prog)s --verbose                    # Show detailed reports
    %(prog)s --clean                      # Clean temporary files only
        """
    )
    parser.add_argument('--input', type=Path, default=None,
                        help='Input file path (Excel or CSV)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Display detailed reports')
    parser.add_argument('--clean', action='store_true',
                        help='Clean temporary files and exit')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (auto-generated if not specified)')

    args = parser.parse_args()

    if args.clean:
        cleanup_temp_files()
        print("✅ Temporary files cleaned")
        sys.exit(0)

    try:
        # Determine input path
        if args.input:
            input_path = args.input
        else:
            input_path = get_data_root() / "consolidated" / "dataset_complet.xlsx"

        # Run preprocessing
        df_final = full_preprocessing(input_path, verbose=args.verbose)

        # Save processed data
        output_path = save_processed(df_final, filename=args.output)

        # Cleanup temporary files
        cleanup_temp_files()

        print(f"\n✅ Preprocessing completed successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Shape: {df_final.shape}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
