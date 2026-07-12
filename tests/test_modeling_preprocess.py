import numpy as np

from proxyml_core.modeling.preprocess import build_preprocessor
from proxyml_core.schema import (
    CategoricalFeature,
    CategoricalOrdinalFeature,
    ContinuousFeature,
    CountFeature,
    NumericOrdinalFeature,
)


def _all_types_schema():
    return [
        ContinuousFeature(name="x1", mean=0.0, std=1.0, min=-10.0, max=10.0),
        CategoricalFeature(name="c1", valid_categories={"a": 0.5, "b": 0.5}),
        CategoricalOrdinalFeature(name="o1", categories=["low", "high"], probabilities=[0.5, 0.5]),
        NumericOrdinalFeature(name="o2", categories=[1, 2, 3], mean=2.0, std=1.0),
        CountFeature(name="n1", lambda_=3.0),
    ]


def test_build_preprocessor_transforms_all_feature_types():
    features = _all_types_schema()
    preprocessor = build_preprocessor(features)

    X = np.array([
        [1.0, "a", "low", 1, 2],
        [2.0, "b", "high", 3, 5],
        [-1.0, "a", "high", 2, 0],
    ], dtype=object)
    out = preprocessor.fit_transform(X)

    # 1 (continuous) + 2 (OHE for c1) + 1 (ordinal o1) + 1 (ordinal o2) + 1 (count) = 6 cols
    assert out.shape == (3, 6)

    transformer_names = {name for name, _, _ in preprocessor.transformers_ if name != "remainder"}
    assert transformer_names == {"num", "cat", "ord", "count"}


def test_build_preprocessor_handles_missing_feature_types():
    features = [ContinuousFeature(name="x1", mean=0.0, std=1.0, min=-10.0, max=10.0)]
    preprocessor = build_preprocessor(features)
    X = np.array([[1.0], [2.0], [3.0]])
    out = preprocessor.fit_transform(X)
    assert out.shape == (3, 1)
