import numpy as np
import pytest

from proxyml_core.modeling.scoring import score_predictions


def test_score_predictions_classification_shape():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    result = score_predictions(y_true, y_pred, task="classification")
    assert set(result) == {"f1", "accuracy"}
    assert result["accuracy"] == pytest.approx(4 / 5)


def test_score_predictions_classification_perfect_score():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    result = score_predictions(y_true, y_pred, task="classification")
    assert result["f1"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)


def test_score_predictions_regression_shape():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = score_predictions(y_true, y_pred, task="regression")
    assert set(result) == {"r2"}
    assert result["r2"] == pytest.approx(1.0)


def test_score_predictions_classification_handles_zero_division():
    # All one class, predictions also all one class -> f1_score's zero_division=0
    # guards against a warning/exception when a class is never predicted.
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    result = score_predictions(y_true, y_pred, task="classification")
    assert result["accuracy"] == pytest.approx(1.0)
