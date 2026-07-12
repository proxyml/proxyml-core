"""Extract the export contract from a fitted pipeline. Requires ``proxyml-core[modeling]``."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from proxyml_core.export import FeatureExportEntry, PerClassCoefficients, PerClassIntercept, SurrogateExport
from proxyml_core.schema import Feature


def extract_export_data(pipeline: Pipeline, features: list[Feature], task: str) -> SurrogateExport:
    """Extract everything needed to reconstruct scoring outside the API.

    Returns a ``SurrogateExport`` with only the scoring-relevant fields
    populated (``task``, ``classes``, ``intercept``, ``per_class_intercepts``,
    ``features``); a caller with run metadata (version, trained_at, etc.)
    should fill in the rest via ``dataclasses.replace``.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]

    coef = np.atleast_2d(np.asarray(estimator.coef_))
    intercepts = np.atleast_1d(estimator.intercept_)

    is_multiclass = task == "classification" and coef.shape[0] > 1
    classes = [str(c) for c in estimator.classes_] if task == "classification" else None

    # Walk the ColumnTransformer in output order to build per-feature segments.
    segments: list[dict] = []
    for t_name, t_obj, col_indices in preprocessor.transformers_:
        if t_name == "remainder":
            continue
        if t_name == "cat":
            encoder = t_obj.named_steps["encoder"]
            for local_i, global_i in enumerate(col_indices):
                segments.append({
                    "feature": features[global_i],
                    "n_cols": len(encoder.categories_[local_i]),
                    "ohe_categories": [str(c) for c in encoder.categories_[local_i]],
                    "scaler_mean": None,
                    "scaler_scale": None,
                    "ordinal_categories": None,
                })
        elif t_name in ("num", "count"):
            scaler = t_obj.named_steps["scaler"]
            for local_i, global_i in enumerate(col_indices):
                segments.append({
                    "feature": features[global_i],
                    "n_cols": 1,
                    "ohe_categories": None,
                    "scaler_mean": float(scaler.mean_[local_i]),
                    "scaler_scale": float(scaler.scale_[local_i]),
                    "ordinal_categories": None,
                })
        elif t_name == "ord":
            encoder = t_obj.named_steps["encoder"]
            for local_i, global_i in enumerate(col_indices):
                segments.append({
                    "feature": features[global_i],
                    "n_cols": 1,
                    "ohe_categories": None,
                    "scaler_mean": None,
                    "scaler_scale": None,
                    "ordinal_categories": [str(c) for c in encoder.categories_[local_i]],
                })

    feature_entries: list[FeatureExportEntry] = []
    pos = 0
    for seg in segments:
        n_cols = seg["n_cols"]
        feat = seg["feature"]
        entry_kwargs: dict = {
            "name": feat.name,
            "type": feat.type,
            "scaler_mean": seg["scaler_mean"],
            "scaler_scale": seg["scaler_scale"],
            "ohe_categories": seg["ohe_categories"],
            "ordinal_categories": seg["ordinal_categories"],
        }
        if is_multiclass:
            entry_kwargs["coefficient"] = None
            entry_kwargs["category_coefficients"] = None
            entry_kwargs["per_class_coefficients"] = [
                PerClassCoefficients(
                    class_label=classes[i],
                    coefficient=float(coef[i, pos]) if n_cols == 1 else None,
                    category_coefficients=coef[i, pos:pos + n_cols].tolist() if n_cols > 1 else None,
                )
                for i in range(len(classes))
            ]
        else:
            chunk = coef[0, pos:pos + n_cols]
            entry_kwargs["per_class_coefficients"] = None
            if n_cols == 1:
                entry_kwargs["coefficient"] = float(chunk[0])
                entry_kwargs["category_coefficients"] = None
            else:
                entry_kwargs["coefficient"] = None
                entry_kwargs["category_coefficients"] = chunk.tolist()
        feature_entries.append(FeatureExportEntry(**entry_kwargs))
        pos += n_cols

    if is_multiclass:
        intercept = None
        per_class_intercepts = [
            PerClassIntercept(class_label=classes[i], intercept=float(intercepts[i]))
            for i in range(len(classes))
        ]
    else:
        intercept = float(intercepts[0])
        per_class_intercepts = None

    return SurrogateExport(
        task=task,
        classes=classes,
        intercept=intercept,
        per_class_intercepts=per_class_intercepts,
        features=feature_entries,
    )
