"""Feature preprocessing construction."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(config) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.numeric_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                config.categorical_columns,
            ),
        ]
    )
