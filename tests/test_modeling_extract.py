"""The most important test in this package: fit a real pipeline, extract its
export contract, and confirm predict_from_export reproduces the pipeline's
own predictions. This closes a gap that existed in both source repos —
extraction and reconstruction had never been tested against each other.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from proxyml_core.export import predict_from_export, score_export
from proxyml_core.modeling.estimators import get_default_classifier, get_default_regressor
from proxyml_core.modeling.extract import extract_export_data
from proxyml_core.modeling.preprocess import build_preprocessor
from proxyml_core.schema import (
    CategoricalFeature,
    CategoricalOrdinalFeature,
    ContinuousFeature,
    CountFeature,
    NumericOrdinalFeature,
)


def _schema():
    return [
        ContinuousFeature(name="income", mean=50000.0, std=15000.0, min=0.0, max=200000.0),
        CategoricalFeature(name="region", valid_categories={"east": 0.5, "west": 0.5}),
        CategoricalOrdinalFeature(name="tier", categories=["bronze", "silver", "gold"], probabilities=[0.34, 0.33, 0.33]),
        NumericOrdinalFeature(name="rating", categories=[1, 2, 3, 4, 5], mean=3.0, std=1.0),
        CountFeature(name="visits", lambda_=4.0),
    ]


def _synthetic_data(features, n=200, seed=0):
    rng = np.random.RandomState(seed)
    columns = []
    for f in features:
        np.random.seed(seed + hash(f.name) % 1000)
        columns.append(f.sample(size=n))
    X = np.empty((n, len(features)), dtype=object)
    for i, col in enumerate(columns):
        X[:, i] = col
    return X


def _row_to_sample(row, features):
    return {f.name: row[i] for i, f in enumerate(features)}


def test_regression_export_reproduces_pipeline_predictions():
    features = _schema()
    X = _synthetic_data(features, n=200, seed=1)
    rng = np.random.RandomState(1)
    y = X[:, 0].astype(float) * 0.001 + rng.normal(scale=0.1, size=len(X))

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(features)),
        ("estimator", get_default_regressor()),
    ])
    pipeline.fit(X, y)

    export = extract_export_data(pipeline, features, task="regression")
    preds = pipeline.predict(X)

    for i in range(10):
        sample = _row_to_sample(X[i], features)
        reconstructed = predict_from_export(export, sample)
        assert reconstructed == pytest.approx(preds[i], abs=1e-6)


def test_binary_classification_export_reproduces_pipeline_predictions():
    features = _schema()
    X = _synthetic_data(features, n=300, seed=2)
    y = np.where(X[:, 0].astype(float) > 50000.0, "high", "low")

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(features)),
        ("estimator", get_default_classifier()),
    ])
    pipeline.fit(X, y)

    export = extract_export_data(pipeline, features, task="classification")
    preds = pipeline.predict(X)

    matches = 0
    for i in range(30):
        sample = _row_to_sample(X[i], features)
        reconstructed = predict_from_export(export, sample)
        if reconstructed == str(preds[i]):
            matches += 1
    assert matches == 30


def test_multiclass_export_reproduces_pipeline_predictions():
    features = _schema()
    X = _synthetic_data(features, n=300, seed=3)
    income = X[:, 0].astype(float)
    y = np.select(
        [income < 40000, income < 70000],
        ["low", "mid"],
        default="high",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(features)),
        ("estimator", get_default_classifier()),
    ])
    pipeline.fit(X, y)

    export = extract_export_data(pipeline, features, task="classification")
    assert export.per_class_intercepts is not None  # confirms multiclass path taken
    preds = pipeline.predict(X)

    matches = 0
    for i in range(30):
        sample = _row_to_sample(X[i], features)
        reconstructed = predict_from_export(export, sample)
        if reconstructed == str(preds[i]):
            matches += 1
    assert matches == 30


def test_export_round_trips_through_json_and_still_scores():
    features = _schema()
    X = _synthetic_data(features, n=150, seed=4)
    y = X[:, 0].astype(float) * 0.001

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(features)),
        ("estimator", get_default_regressor()),
    ])
    pipeline.fit(X, y)

    export = extract_export_data(pipeline, features, task="regression")
    from proxyml_core.export import SurrogateExport
    restored = SurrogateExport.from_dict(export.to_dict())

    sample = _row_to_sample(X[0], features)
    assert score_export(restored, sample) == pytest.approx(score_export(export, sample))
