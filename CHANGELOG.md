# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-18

### Added
- `proxyml_core.modeling.scoring.score_predictions()` — scores predictions
  against ground truth (weighted F1 + accuracy for classification, R² for
  regression). Shared by `proxyml.local.train_challenger()`'s internal
  fidelity metrics and the new `proxyml.local.score_champion()`, so both
  sides of a champion-vs-challenger comparison are computed identically.
- `proxyml_core.export.feature_importances_from_export()` — computes
  per-feature importances directly from a `SurrogateExport`'s coefficients,
  pure arithmetic with no fitted sklearn model required. Lets a backend
  render a client-trained challenger's feature importances at report time
  without ever loading a live `Pipeline`.
- `proxyml_core.__version__`, exposed via `importlib.metadata`.

## [0.1.0] - 2026-07-12

### Added
- `proxyml_core.schema` — `Feature` dataclass hierarchy (`ContinuousFeature`,
  `CategoricalFeature`, `CategoricalOrdinalFeature`, `NumericOrdinalFeature`,
  `CountFeature`) and `FeatureSchema`, pure (stdlib + numpy only)
- `proxyml_core.export` — the export JSON contract (`SurrogateExport` and
  friends), `EXPORT_SCHEMA_VERSION` / `check_compatible`, and a complete
  `predict_from_export()` / `score_export()` scorer covering all five
  feature types and multiclass — pure, no sklearn required
- `proxyml_core.modeling` (behind the `proxyml-core[modeling]` extra) —
  `build_preprocessor`, default estimators (`get_default_classifier`,
  `get_default_regressor`), task-type inference, and `extract_export_data`
  for turning a fitted scikit-learn pipeline into a `SurrogateExport`
