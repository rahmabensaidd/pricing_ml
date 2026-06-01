"""Client features training public facade.

Readable API exposing client-features utilities used by pipelines and tests.
Heavy implementation lives in
`_client_features_core/client_features_training_impl.py`.
"""

from pricing__epac.src.machine_learning.training._client_features_core.client_features_training_impl import (
    add_client_features_to_orders,
    aggregate_client_features,
    compute_client_indicators,
    compute_price_elasticity,
    create_client_features,
    load_client_features,
    load_client_features_joblib,
    normalize_price_columns,
    normalize_reception_dates,
    process_client_dates,
    save_client_features,
    save_client_features_all_formats,
    save_client_features_joblib,
)


__all__ = [
    "normalize_reception_dates",
    "normalize_price_columns",
    "compute_client_indicators",
    "aggregate_client_features",
    "process_client_dates",
    "compute_price_elasticity",
    "create_client_features",
    "add_client_features_to_orders",
    "save_client_features_joblib",
    "load_client_features_joblib",
    "save_client_features_all_formats",
    "save_client_features",
    "load_client_features",
]
