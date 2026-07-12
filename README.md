# proxyml-core

Shared library for [proxyml](https://proxyml.ai) — schema types, the export
JSON contract, and (behind the `modeling` extra) the sklearn-based training
and preprocessing code used by both the proxyml backend and the `proxyml`
SDK's local challenger training.

## Layout

- `proxyml_core.schema` — pure. `Feature` hierarchy, `FeatureSchema`.
- `proxyml_core.export` — pure. Export dataclasses, `EXPORT_SCHEMA_VERSION`,
  `predict_from_export()`.
- `proxyml_core.modeling` — requires `proxyml-core[modeling]` (scikit-learn,
  scipy). Preprocessing, default estimators, export extraction.

The pure base (`schema`, `export`, `_version`) depends only on `numpy`, so a
REST-only `proxyml` install never needs scikit-learn.

## Install

```
pip install proxyml-core            # pure: schema + export
pip install proxyml-core[modeling]  # + scikit-learn, scipy
```
