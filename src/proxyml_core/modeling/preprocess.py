"""Build the sklearn ColumnTransformer for a feature schema.

Requires ``proxyml-core[modeling]``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
import numpy as np

from proxyml_core.schema import (
    CategoricalFeature,
    CategoricalOrdinalFeature,
    ContinuousFeature,
    CountFeature,
    Feature,
    NumericOrdinalFeature,
)


def build_preprocessor(features: list[Feature]) -> ColumnTransformer:
    numeric_indices = [i for i, f in enumerate(features) if isinstance(f, ContinuousFeature)]
    categorical_indices = [i for i, f in enumerate(features) if isinstance(f, CategoricalFeature)]
    ordinal_indices = [
        i for i, f in enumerate(features)
        if isinstance(f, (CategoricalOrdinalFeature, NumericOrdinalFeature))
    ]
    count_indices = [i for i, f in enumerate(features) if isinstance(f, CountFeature)]

    transformers = []

    if numeric_indices:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_indices))

    if categorical_indices:
        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_transformer, categorical_indices))

    if ordinal_indices:
        ordinal_features = [
            f for f in features if isinstance(f, (CategoricalOrdinalFeature, NumericOrdinalFeature))
        ]
        categories = [f.categories for f in ordinal_features]
        ordinal_transformer = Pipeline(steps=[
            ("encoder", OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])
        transformers.append(("ord", ordinal_transformer, ordinal_indices))

    if count_indices:
        count_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p)),  # log1p handles zeros safely
            ("scaler", StandardScaler())
        ])
        transformers.append(("count", count_transformer, count_indices))

    return ColumnTransformer(transformers=transformers)
