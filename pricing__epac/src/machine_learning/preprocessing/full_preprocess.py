"""
Compatibility wrapper around ``full_prepro``.

The project now relies on ``full_prepro`` as the source of truth for
preprocessing. This module keeps legacy imports working while exposing the
same high-level entry points.
"""

from pricing__epac.src.machine_learning.preprocessing.full_prepro import (
    apply_all_mappings,
    clean_dates,
    cleanup_temp_files,
    drop_constant_columns,
    final_normalization,
    fix_corrupted_dates,
    full_preprocessing,
    get_data_root,
    get_package_root,
    get_project_root,
    impute_cover_size_saddle_stitch,
    initial_cleaning,
    load_data,
    load_mappings,
    normalize_column,
    print_unique_values_summary,
    quality_check,
    remove_duplicates,
    replace_nat_with_sentinel_date,
    save_processed,
    uppercase_string_columns,
    validate_dates,
)


def uppercase_all_string_columns(*args, **kwargs):
    return uppercase_string_columns(*args, **kwargs)


def replace_nat_nan_none(*args, **kwargs):
    return final_normalization(*args, **kwargs)


__all__ = [
    "apply_all_mappings",
    "clean_dates",
    "cleanup_temp_files",
    "drop_constant_columns",
    "final_normalization",
    "fix_corrupted_dates",
    "full_preprocessing",
    "get_data_root",
    "get_package_root",
    "get_project_root",
    "impute_cover_size_saddle_stitch",
    "initial_cleaning",
    "load_data",
    "load_mappings",
    "normalize_column",
    "print_unique_values_summary",
    "quality_check",
    "remove_duplicates",
    "replace_nat_nan_none",
    "replace_nat_with_sentinel_date",
    "save_processed",
    "uppercase_all_string_columns",
    "uppercase_string_columns",
    "validate_dates",
]
