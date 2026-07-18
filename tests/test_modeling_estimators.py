import warnings

import numpy as np
import pytest

from proxyml_core.modeling.estimators import (
    binarize_if_probabilities,
    extract_hyperparameters,
    get_default_classifier,
    get_default_regressor,
    is_classification,
    to_json_safe,
)


def test_is_classification_object_dtype():
    assert is_classification(np.array(["a", "b", "a"], dtype=object)) is True


def test_is_classification_low_cardinality_numeric():
    assert is_classification(np.array([0, 1, 0, 1, 1])) is True


def test_is_classification_high_cardinality_numeric_is_regression():
    assert is_classification(np.linspace(0, 100, 50)) is False


def test_binarize_if_probabilities_thresholds_at_half():
    probs = np.array([0.1, 0.6, 0.49, 0.5])
    out = binarize_if_probabilities(probs)
    assert list(out) == [0, 1, 0, 1]


def test_binarize_if_probabilities_passes_through_discrete_labels():
    labels = np.array([0, 1, 0, 1])
    out = binarize_if_probabilities(labels)
    assert list(out) == [0, 1, 0, 1]


def test_binarize_if_probabilities_passes_through_object_labels():
    labels = np.array(["yes", "no"], dtype=object)
    out = binarize_if_probabilities(labels)
    assert list(out) == ["yes", "no"]


def test_binarize_if_probabilities_rejects_nan():
    # Casting NaN to int is undefined and previously silently mis-binarized
    # to 0 instead of raising — must fail loudly instead.
    with pytest.raises(ValueError, match="NaN or infinite"):
        binarize_if_probabilities(np.array([0.1, 0.6, np.nan, 0.5]))


def test_binarize_if_probabilities_rejects_inf():
    with pytest.raises(ValueError, match="NaN or infinite"):
        binarize_if_probabilities(np.array([0.1, 0.6, np.inf, 0.5]))


def test_to_json_safe_handles_numpy_types():
    assert to_json_safe(np.array([1, 2, 3])) == [1, 2, 3]
    assert to_json_safe(np.float64(1.5)) == 1.5
    assert to_json_safe([np.int64(1), np.int64(2)]) == [1, 2]
    assert to_json_safe(5) == 5


def test_extract_hyperparameters_includes_cv_selected_values():
    X = np.random.RandomState(0).normal(size=(50, 3))
    y = X[:, 0] * 2 + 1
    reg = get_default_regressor()
    reg.fit(X, y)
    params = extract_hyperparameters(reg)
    assert "alpha_" in params
    assert isinstance(params["alpha_"], float)


def test_get_default_classifier_fits_without_sklearn_deprecation_warnings():
    # scoring="accuracy" and use_legacy_attributes=False pin today's sklearn
    # defaults explicitly, so upgrading sklearn can't silently change fit
    # behavior and users don't see FutureWarning noise on every fit.
    X = np.random.RandomState(0).normal(size=(50, 3))
    y = (X[:, 0] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        get_default_classifier().fit(X, y)


def test_get_default_classifier_and_regressor_are_fittable():
    X = np.random.RandomState(0).normal(size=(50, 3))
    y_class = (X[:, 0] > 0).astype(int)
    y_reg = X[:, 0] * 2 + 1

    clf = get_default_classifier()
    clf.fit(X, y_class)
    assert hasattr(clf, "coef_")

    reg = get_default_regressor()
    reg.fit(X, y_reg)
    assert hasattr(reg, "coef_")
