"""Default estimators and task-type inference. Requires ``proxyml-core[modeling]``."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegressionCV, RidgeCV


def to_json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [to_json_safe(x) for x in v]
    return v


def extract_hyperparameters(estimator: BaseEstimator) -> dict:
    """Return configured and CV-selected hyperparameters from a fitted estimator."""
    params = {k: to_json_safe(v) for k, v in estimator.get_params().items()}
    for attr in ("alpha_", "C_", "l1_ratio_"):
        if hasattr(estimator, attr):
            params[attr] = to_json_safe(getattr(estimator, attr))
    return params


def is_classification(predictions: np.ndarray) -> bool:
    """Infer task type from predictions.

    Cardinality is the only signal available — there's no way to distinguish a
    classifier's probability outputs from a genuine continuous regression target
    by value inspection alone, so callers should let users override this guess.
    """
    return predictions.dtype == object or len(np.unique(predictions)) <= 20


def binarize_if_probabilities(predictions: np.ndarray) -> np.ndarray:
    """Threshold probability outputs into hard 0/1 labels for classification.

    A classifier needs discrete labels — if a caller submits raw probabilities
    (e.g. predict_proba output) for a binary task, every unique probability would
    otherwise be treated as its own class. Treat >=0.5 as the positive class.
    Already-discrete labels (ints, or non-numeric class names) pass through unchanged.
    """
    if predictions.dtype.kind not in "iuf":  # not int/uint/float — already discrete labels
        return predictions
    as_float = predictions.astype(float)
    if np.all(as_float == as_float.astype(int)):
        return predictions
    return (as_float >= 0.5).astype(int)


def get_default_classifier() -> BaseEstimator:
    return LogisticRegressionCV(
        solver="lbfgs",
        l1_ratios=(0,),
        class_weight="balanced",
        max_iter=500,
        cv=5,
        n_jobs=-1,
        use_legacy_attributes=False,
    )


def get_default_regressor() -> BaseEstimator:
    return RidgeCV(alphas=np.logspace(-3, 4, 15), cv=5)
