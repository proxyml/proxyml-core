import numpy as np
import pytest

from proxyml_core.schema import (
    CategoricalFeature,
    CategoricalOrdinalFeature,
    ContinuousFeature,
    CountFeature,
    Feature,
    FeatureSchema,
    FeatureValidationError,
    NumericOrdinalFeature,
)


def test_continuous_feature_sample_is_clipped():
    f = ContinuousFeature(name="x", mean=0.0, std=1.0, min=-0.5, max=0.5)
    samples = f.sample(size=1000)
    assert samples.min() >= -0.5
    assert samples.max() <= 0.5


def test_continuous_feature_round_trip():
    f = ContinuousFeature(name="x", mean=1.0, std=2.0, min=-10.0, max=10.0)
    d = f.to_dict()
    assert d == {
        "type": "continuous",
        "name": "x",
        "immutable": False,
        "mean": 1.0,
        "std": 2.0,
        "min": -10.0,
        "max": 10.0,
    }
    assert Feature.from_dict(d) == f


def test_categorical_feature_requires_probabilities_sum_to_one():
    with pytest.raises(FeatureValidationError):
        CategoricalFeature(name="c", valid_categories={"a": 0.5, "b": 0.4})


def test_categorical_feature_round_trip():
    f = CategoricalFeature(name="c", valid_categories={"a": 0.5, "b": 0.5})
    d = f.to_dict()
    assert Feature.from_dict(d) == f


def test_categorical_ordinal_requires_matching_lengths():
    with pytest.raises(FeatureValidationError):
        CategoricalOrdinalFeature(
            name="o", categories=["low", "high"], probabilities=[1.0]
        )


def test_categorical_ordinal_requires_probabilities_sum_to_one():
    with pytest.raises(FeatureValidationError):
        CategoricalOrdinalFeature(
            name="o", categories=["low", "high"], probabilities=[0.9, 0.2]
        )


def test_categorical_ordinal_round_trip():
    f = CategoricalOrdinalFeature(
        name="o", categories=["low", "high"], probabilities=[0.3, 0.7]
    )
    d = f.to_dict()
    assert Feature.from_dict(d) == f


def test_numeric_ordinal_sample_snaps_to_category():
    f = NumericOrdinalFeature(name="stars", categories=[1, 2, 3, 4, 5], mean=3.0, std=0.01)
    samples = f.sample(size=100)
    assert set(np.unique(samples)).issubset({1, 2, 3, 4, 5})


def test_numeric_ordinal_round_trip():
    f = NumericOrdinalFeature(name="stars", categories=[1, 2, 3], mean=2.0, std=1.0)
    d = f.to_dict()
    assert Feature.from_dict(d) == f


def test_count_feature_sample_respects_max():
    f = CountFeature(name="n", lambda_=50.0, max=5)
    samples = f.sample(size=100)
    assert samples.max() <= 5
    assert samples.min() >= 0


def test_count_feature_lambda_alias_round_trip():
    f = CountFeature(name="n", lambda_=3.5, max=10)
    d = f.to_dict()
    assert d["lambda"] == 3.5
    assert "lambda_" not in d
    assert Feature.from_dict(d) == f


def test_feature_from_dict_unknown_type_raises():
    with pytest.raises(FeatureValidationError):
        Feature.from_dict({"type": "bogus", "name": "x"})


def test_feature_schema_round_trip():
    schema = FeatureSchema(
        features=[
            ContinuousFeature(name="x", mean=0.0, std=1.0, min=-1.0, max=1.0),
            CategoricalFeature(name="c", valid_categories={"a": 1.0}),
        ]
    )
    d = schema.to_dict()
    restored = FeatureSchema.from_dict(d)
    assert restored == schema


def test_feature_schema_covariance_matrix_shape_validation():
    x1 = ContinuousFeature(name="x1", mean=0.0, std=1.0, min=-1.0, max=1.0)
    x2 = ContinuousFeature(name="x2", mean=0.0, std=1.0, min=-1.0, max=1.0)
    bad_cov = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(FeatureValidationError):
        FeatureSchema(features=[x1, x2], covariance_matrix=bad_cov)


def test_feature_schema_covariance_matrix_must_be_symmetric():
    x1 = ContinuousFeature(name="x1", mean=0.0, std=1.0, min=-1.0, max=1.0)
    x2 = ContinuousFeature(name="x2", mean=0.0, std=1.0, min=-1.0, max=1.0)
    asymmetric = np.array([[1.0, 0.5], [0.1, 1.0]])
    with pytest.raises(FeatureValidationError):
        FeatureSchema(features=[x1, x2], covariance_matrix=asymmetric)


def test_feature_schema_covariance_matrix_must_be_psd():
    x1 = ContinuousFeature(name="x1", mean=0.0, std=1.0, min=-1.0, max=1.0)
    x2 = ContinuousFeature(name="x2", mean=0.0, std=1.0, min=-1.0, max=1.0)
    not_psd = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(FeatureValidationError):
        FeatureSchema(features=[x1, x2], covariance_matrix=not_psd)


def test_feature_schema_valid_covariance_matrix_accepted():
    x1 = ContinuousFeature(name="x1", mean=0.0, std=1.0, min=-1.0, max=1.0)
    x2 = ContinuousFeature(name="x2", mean=0.0, std=1.0, min=-1.0, max=1.0)
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    schema = FeatureSchema(features=[x1, x2], covariance_matrix=cov)
    assert schema.covariance_matrix is cov
