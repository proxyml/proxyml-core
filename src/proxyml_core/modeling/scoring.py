"""Score predictions against labels. Requires ``proxyml-core[modeling]``."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import f1_score, r2_score


def score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, *, task: Literal["classification", "regression"]
) -> dict[str, float]:
    """Score predictions against ground truth with a fixed, reproducible metric set.

    Classification returns weighted F1 + accuracy; regression returns R².
    Shared by ``train_challenger()`` (scoring its own held-out split) and
    ``score_champion()`` (scoring a champion's predictions against the same
    real labels), so both sides of a champion-vs-challenger comparison are
    computed with identical code.
    """
    if task == "classification":
        return {
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        }
    return {"r2": float(r2_score(y_true, y_pred))}
