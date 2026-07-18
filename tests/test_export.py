import math

import pytest

from proxyml_core._version import EXPORT_SCHEMA_VERSION
from proxyml_core.export import (
    ExportError,
    FeatureExportEntry,
    IncompatibleExportVersionError,
    PerClassCoefficients,
    PerClassIntercept,
    SurrogateExport,
    feature_importances_from_export,
    predict_from_export,
    score_export,
)


def _regression_export() -> SurrogateExport:
    return SurrogateExport(
        task="regression",
        intercept=1.0,
        features=[
            FeatureExportEntry(
                name="x", type="continuous", scaler_mean=10.0, scaler_scale=2.0, coefficient=3.0
            ),
            FeatureExportEntry(
                name="c",
                type="categorical",
                ohe_categories=["a", "b"],
                category_coefficients=[0.5, -0.5],
            ),
        ],
    )


def test_regression_score_and_predict():
    export = _regression_export()
    # x=12 -> x_scaled=(12-10)/2=1.0 -> 3.0*1.0=3.0; c="a" -> 0.5; intercept 1.0 => 4.5
    score = score_export(export, {"x": 12, "c": "a"})
    assert score == pytest.approx(4.5)
    assert predict_from_export(export, {"x": 12, "c": "a"}) == pytest.approx(4.5)


def test_categorical_unknown_value_contributes_zero():
    export = _regression_export()
    score = score_export(export, {"x": 10, "c": "unseen"})
    # x_scaled = 0 -> 0 contribution; unknown category -> 0; intercept 1.0
    assert score == pytest.approx(1.0)


def test_count_feature_uses_log1p_transform():
    export = SurrogateExport(
        task="regression",
        intercept=0.0,
        features=[
            FeatureExportEntry(
                name="n", type="count", scaler_mean=0.0, scaler_scale=1.0, coefficient=1.0
            )
        ],
    )
    score = score_export(export, {"n": 5})
    assert score == pytest.approx(math.log1p(5))


def test_ordinal_feature_uses_unscaled_rank():
    export = SurrogateExport(
        task="regression",
        intercept=0.0,
        features=[
            FeatureExportEntry(
                name="o",
                type="categorical_ordinal",
                ordinal_categories=["low", "mid", "high"],
                coefficient=2.0,
            )
        ],
    )
    assert score_export(export, {"o": "high"}) == pytest.approx(2.0 * 2)
    # unknown ordinal category -> OrdinalEncoder(unknown_value=-1) equivalent
    assert score_export(export, {"o": "nope"}) == pytest.approx(2.0 * -1)


def test_missing_feature_strict_raises():
    export = _regression_export()
    with pytest.raises(ExportError):
        score_export(export, {"x": 1})


def test_missing_feature_non_strict_skips():
    export = _regression_export()
    score = score_export(export, {"x": 12}, strict=False)
    assert score == pytest.approx(4.0)  # 3.0 (x) + 1.0 (intercept), c skipped


def test_binary_classification_predict():
    export = SurrogateExport(
        task="classification",
        classes=["no", "yes"],
        intercept=0.0,
        features=[
            FeatureExportEntry(name="x", type="continuous", scaler_mean=0.0, scaler_scale=1.0, coefficient=10.0)
        ],
    )
    assert predict_from_export(export, {"x": 1.0}) == "yes"
    assert predict_from_export(export, {"x": -1.0}) == "no"


def test_multiclass_predict_argmax():
    export = SurrogateExport(
        task="classification",
        classes=["a", "b", "c"],
        per_class_intercepts=[
            PerClassIntercept(class_label="a", intercept=0.0),
            PerClassIntercept(class_label="b", intercept=0.0),
            PerClassIntercept(class_label="c", intercept=5.0),
        ],
        features=[
            FeatureExportEntry(
                name="x",
                type="continuous",
                scaler_mean=0.0,
                scaler_scale=1.0,
                per_class_coefficients=[
                    PerClassCoefficients(class_label="a", coefficient=0.0),
                    PerClassCoefficients(class_label="b", coefficient=0.0),
                    PerClassCoefficients(class_label="c", coefficient=0.0),
                ],
            )
        ],
    )
    scores = score_export(export, {"x": 0.0})
    assert scores == {"a": 0.0, "b": 0.0, "c": 5.0}
    assert predict_from_export(export, {"x": 0.0}) == "c"


def test_to_dict_stamps_current_version():
    export = _regression_export()
    d = export.to_dict()
    assert d["export_schema_version"] == EXPORT_SCHEMA_VERSION


def test_round_trip_via_dict():
    export = _regression_export()
    restored = SurrogateExport.from_dict(export.to_dict())
    assert restored == export


def test_from_dict_rejects_future_version():
    export = _regression_export()
    d = export.to_dict()
    d["export_schema_version"] = EXPORT_SCHEMA_VERSION + 1
    with pytest.raises(IncompatibleExportVersionError):
        SurrogateExport.from_dict(d)


def test_from_dict_accepts_older_version():
    export = _regression_export()
    d = export.to_dict()
    d["export_schema_version"] = EXPORT_SCHEMA_VERSION  # same/older is fine
    SurrogateExport.from_dict(d)  # should not raise


def test_feature_importances_from_export_binary():
    export = _regression_export()
    result = feature_importances_from_export(export)
    assert result["per_class_importances"] is None
    importances = {e["feature"]: e for e in result["feature_importances"]}
    # continuous: signed coefficient preserved
    assert importances["x"]["coefficient"] == pytest.approx(3.0)
    assert importances["x"]["abs_coefficient"] == pytest.approx(3.0)
    # categorical (OHE): mean(abs(category_coefficients)) -> mean(0.5, 0.5) = 0.5
    assert importances["c"]["coefficient"] == pytest.approx(0.5)
    assert importances["c"]["abs_coefficient"] == pytest.approx(0.5)
    # sorted by abs_coefficient descending
    assert [e["feature"] for e in result["feature_importances"]] == ["x", "c"]


def test_feature_importances_from_export_negative_coefficient_stays_signed():
    export = SurrogateExport(
        task="regression",
        intercept=0.0,
        features=[
            FeatureExportEntry(name="x", type="continuous", scaler_mean=0.0, scaler_scale=1.0, coefficient=-4.0)
        ],
    )
    result = feature_importances_from_export(export)
    entry = result["feature_importances"][0]
    assert entry["coefficient"] == pytest.approx(-4.0)
    assert entry["abs_coefficient"] == pytest.approx(4.0)


def test_feature_importances_from_export_multiclass():
    export = SurrogateExport(
        task="classification",
        classes=["a", "b"],
        per_class_intercepts=[
            PerClassIntercept(class_label="a", intercept=0.0),
            PerClassIntercept(class_label="b", intercept=0.0),
        ],
        features=[
            FeatureExportEntry(
                name="x",
                type="continuous",
                scaler_mean=0.0,
                scaler_scale=1.0,
                per_class_coefficients=[
                    PerClassCoefficients(class_label="a", coefficient=2.0),
                    PerClassCoefficients(class_label="b", coefficient=-6.0),
                ],
            )
        ],
    )
    result = feature_importances_from_export(export)
    assert result["per_class_importances"] is not None
    per_class = {cp["class_label"]: cp["importances"] for cp in result["per_class_importances"]}
    assert per_class["a"][0]["coefficient"] == pytest.approx(2.0)
    assert per_class["b"][0]["coefficient"] == pytest.approx(-6.0)
    # top-level aggregate: mean(abs(2.0), abs(-6.0)) = 4.0
    top = result["feature_importances"][0]
    assert top["feature"] == "x"
    assert top["coefficient"] == pytest.approx(4.0)
    assert top["abs_coefficient"] == pytest.approx(4.0)


def test_feature_importances_from_export_multiclass_categorical_collapses_via_mean_abs():
    export = SurrogateExport(
        task="classification",
        classes=["a", "b"],
        per_class_intercepts=[
            PerClassIntercept(class_label="a", intercept=0.0),
            PerClassIntercept(class_label="b", intercept=0.0),
        ],
        features=[
            FeatureExportEntry(
                name="c",
                type="categorical",
                ohe_categories=["x", "y"],
                per_class_coefficients=[
                    PerClassCoefficients(class_label="a", category_coefficients=[1.0, -3.0]),
                    PerClassCoefficients(class_label="b", category_coefficients=[0.0, 0.0]),
                ],
            )
        ],
    )
    result = feature_importances_from_export(export)
    per_class = {cp["class_label"]: cp["importances"] for cp in result["per_class_importances"]}
    # class "a": mean(abs(1.0), abs(-3.0)) = 2.0
    assert per_class["a"][0]["coefficient"] == pytest.approx(2.0)
    assert per_class["a"][0]["abs_coefficient"] == pytest.approx(2.0)
    assert per_class["b"][0]["coefficient"] == pytest.approx(0.0)
    # top-level: mean(2.0, 0.0) = 1.0
    assert result["feature_importances"][0]["coefficient"] == pytest.approx(1.0)
