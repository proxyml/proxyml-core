"""The export JSON contract and its arithmetic scorer.

Pure: stdlib + numpy only. ``extract_export_data`` (in
``proxyml_core.modeling.extract``, which requires sklearn) *produces* a
``SurrogateExport``; everything in this module — reading, serializing, and
scoring one — needs nothing beyond arithmetic. That's deliberate: it's the
"your model is yours" scorer — a user can reconstruct predictions with zero
sklearn, the same arithmetic on the same JSON, whether the artifact came
from a server-trained surrogate or a locally-trained challenger.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from proxyml_core._version import (
    EXPORT_SCHEMA_VERSION,
    IncompatibleExportVersionError,
    check_compatible,
)

SCORING_NOTE = (
    "Coefficients are in the preprocessed feature space. "
    "Continuous: x_scaled = (x - scaler_mean) / scaler_scale. "
    "Count: x_scaled = (log1p(x) - scaler_mean) / scaler_scale. "
    "Categorical: one-hot encoded — ohe_categories[i] is the category for coefficient[i]. "
    "Ordinal: encoded as rank (0-based) per ordinal_categories order. "
    "score = dot(preprocessed_x, coefficients) + intercept. "
    "For binary classification score is a logit (sigmoid for probability). "
    "For multiclass, compute one score per class and take the argmax."
)


class ExportError(ValueError):
    """Raised for malformed export payloads or scoring inputs."""


@dataclass(kw_only=True)
class PerClassCoefficients:
    class_label: str
    coefficient: float | None = None
    category_coefficients: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerClassCoefficients":
        return cls(**d)


@dataclass(kw_only=True)
class FeatureExportEntry:
    name: str
    type: str
    scaler_mean: float | None = None
    scaler_scale: float | None = None
    ohe_categories: list[str] | None = None
    ordinal_categories: list[str] | None = None
    coefficient: float | None = None
    category_coefficients: list[float] | None = None
    per_class_coefficients: list[PerClassCoefficients] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.per_class_coefficients is not None:
            d["per_class_coefficients"] = [c.to_dict() for c in self.per_class_coefficients]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureExportEntry":
        d = dict(d)
        per_class = d.get("per_class_coefficients")
        if per_class is not None:
            d["per_class_coefficients"] = [PerClassCoefficients.from_dict(c) for c in per_class]
        return cls(**d)


@dataclass(kw_only=True)
class PerClassIntercept:
    class_label: str
    intercept: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerClassIntercept":
        return cls(**d)


@dataclass(kw_only=True)
class SurrogateExport:
    """The full export contract.

    ``extract_export_data`` populates only the scoring-relevant fields
    (``task``, ``classes``, ``intercept``, ``per_class_intercepts``,
    ``features``) — a caller with run metadata (version, trained_at, etc.,
    e.g. a backend endpoint) fills in the rest via ``dataclasses.replace``.
    """

    task: str
    features: list[FeatureExportEntry]
    classes: list[str] | None = None
    intercept: float | None = None
    per_class_intercepts: list[PerClassIntercept] | None = None

    version: str | None = None
    trained_at: str | None = None
    schema_name: str | None = None
    name: str | None = None
    comments: str | None = None
    metrics: dict[str, float] | None = None
    hyperparameters: dict | None = None
    run_id: str | None = None
    schema_definition: list[dict] | None = None
    schema_warning: str | None = None
    note: str = SCORING_NOTE
    export_schema_version: int = EXPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = [f.to_dict() for f in self.features]
        if self.per_class_intercepts is not None:
            d["per_class_intercepts"] = [p.to_dict() for p in self.per_class_intercepts]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SurrogateExport":
        payload_version = d.get("export_schema_version", 1)
        if not check_compatible(payload_version):
            raise IncompatibleExportVersionError(
                f"Export payload is at version {payload_version}, but this copy of "
                f"proxyml-core only understands up to version {EXPORT_SCHEMA_VERSION}. "
                "Upgrade proxyml-core to read it."
            )
        d = dict(d)
        d["features"] = [FeatureExportEntry.from_dict(f) for f in d["features"]]
        per_class_intercepts = d.get("per_class_intercepts")
        if per_class_intercepts is not None:
            d["per_class_intercepts"] = [PerClassIntercept.from_dict(p) for p in per_class_intercepts]
        return cls(**d)


def _feature_value(sample: Mapping[str, Any], name: str, strict: bool) -> Any:
    if name not in sample:
        if strict:
            raise ExportError(f"Missing value for feature {name!r}")
        return None
    return sample[name]


def _contribution(entry: FeatureExportEntry, value: Any, class_index: int | None) -> float:
    if entry.type in ("continuous", "count"):
        x = float(value)
        if entry.type == "count":
            x = math.log1p(x)
        x_scaled = (x - entry.scaler_mean) / entry.scaler_scale
        coef = (
            entry.per_class_coefficients[class_index].coefficient
            if class_index is not None
            else entry.coefficient
        )
        return coef * x_scaled
    if entry.type == "categorical":
        categories = entry.ohe_categories or []
        try:
            idx = categories.index(str(value))
        except ValueError:
            return 0.0  # unknown category -> all-zero OHE row, matches handle_unknown="ignore"
        coefs = (
            entry.per_class_coefficients[class_index].category_coefficients
            if class_index is not None
            else entry.category_coefficients
        )
        return coefs[idx]
    if entry.type in ("categorical_ordinal", "numeric_ordinal"):
        categories = entry.ordinal_categories or []
        try:
            rank = categories.index(str(value))
        except ValueError:
            rank = -1  # unknown -> OrdinalEncoder(unknown_value=-1)
        coef = (
            entry.per_class_coefficients[class_index].coefficient
            if class_index is not None
            else entry.coefficient
        )
        return coef * rank
    raise ExportError(f"Unknown feature type: {entry.type!r}")


def score_export(
    export: SurrogateExport, sample: Mapping[str, Any], *, strict: bool = True
) -> float | dict[str, float]:
    """Compute the raw linear score(s) for ``sample`` against ``export``.

    Regression / binary classification return a single float (a logit, in
    the binary case). Multiclass returns a ``{class_label: score}`` dict.
    Missing feature values raise by default (``strict=True``); pass
    ``strict=False`` to silently skip them (their contribution is treated
    as 0), e.g. when intentionally scoring on a reduced feature subset.
    """
    is_multiclass = export.per_class_intercepts is not None
    if is_multiclass:
        scores: dict[str, float] = {}
        for i, pci in enumerate(export.per_class_intercepts):
            total = pci.intercept
            for entry in export.features:
                value = _feature_value(sample, entry.name, strict)
                if value is None:
                    continue
                total += _contribution(entry, value, i)
            scores[pci.class_label] = total
        return scores

    total = export.intercept or 0.0
    for entry in export.features:
        value = _feature_value(sample, entry.name, strict)
        if value is None:
            continue
        total += _contribution(entry, value, None)
    return total


def predict_from_export(
    export: SurrogateExport, sample: Mapping[str, Any], *, strict: bool = True
) -> float | str:
    """Reconstruct the prediction ``/surrogate/predict`` would return, purely from the export."""
    score = score_export(export, sample, strict=strict)
    if export.task == "regression":
        return score
    if isinstance(score, dict):
        return max(score, key=score.get)
    prob = 1.0 / (1.0 + math.exp(-score))
    classes = export.classes or ["0", "1"]
    return classes[1] if prob >= 0.5 else classes[0]


def _entry_importance(entry: FeatureExportEntry, class_index: int | None) -> tuple[float, float]:
    """Return (coefficient, abs_coefficient) for one feature, one class (or the whole model if not multiclass).

    Categorical (OHE) features collapse to mean(abs(category_coefficients)) — sign
    isn't meaningful across one-hot columns, so ``coefficient`` and ``abs_coefficient``
    are the same value. Other feature types keep the signed coefficient.
    """
    if entry.type == "categorical":
        coefs = (
            entry.per_class_coefficients[class_index].category_coefficients
            if class_index is not None
            else entry.category_coefficients
        ) or []
        mean_abs = sum(abs(c) for c in coefs) / len(coefs) if coefs else 0.0
        return mean_abs, mean_abs
    coef = (
        entry.per_class_coefficients[class_index].coefficient
        if class_index is not None
        else entry.coefficient
    ) or 0.0
    return coef, abs(coef)


def feature_importances_from_export(export: SurrogateExport) -> dict[str, Any]:
    """Compute per-feature importances directly from a ``SurrogateExport`` — pure arithmetic,
    no fitted sklearn model needed. Lets a report render challenger importances without
    ever loading a live ``Pipeline``, unlike a server-trained surrogate's report path.

    Mirrors the collapsing/aggregation rules a live-model extraction would use: OHE
    categorical coefficients collapse via mean(abs(...)); multiclass features aggregate
    mean(abs(...)) across per-class coefficients for the top-level list.

    Returns ``{"feature_importances": [{"feature","coefficient","abs_coefficient"}, ...],
    "per_class_importances": [{"class_label","importances":[...]}, ...] | None}``,
    each list sorted by ``abs_coefficient`` descending.
    """
    is_multiclass = export.per_class_intercepts is not None

    if not is_multiclass:
        rows = []
        for entry in export.features:
            coef, abs_coef = _entry_importance(entry, None)
            rows.append({"feature": entry.name, "coefficient": coef, "abs_coefficient": abs_coef})
        rows.sort(key=lambda e: -e["abs_coefficient"])
        return {"feature_importances": rows, "per_class_importances": None}

    per_class = []
    for i, pci in enumerate(export.per_class_intercepts):
        rows = []
        for entry in export.features:
            coef, abs_coef = _entry_importance(entry, i)
            rows.append({"feature": entry.name, "coefficient": coef, "abs_coefficient": abs_coef})
        rows.sort(key=lambda e: -e["abs_coefficient"])
        per_class.append({"class_label": pci.class_label, "importances": rows})

    feat_abs: dict[str, list[float]] = {}
    for cp in per_class:
        for entry in cp["importances"]:
            feat_abs.setdefault(entry["feature"], []).append(entry["abs_coefficient"])
    feature_importances = sorted(
        (
            {"feature": f, "coefficient": sum(vals) / len(vals), "abs_coefficient": sum(vals) / len(vals)}
            for f, vals in feat_abs.items()
        ),
        key=lambda e: -e["abs_coefficient"],
    )
    return {"feature_importances": feature_importances, "per_class_importances": per_class}
