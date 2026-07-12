"""Regression guard for the export JSON contract's versioning.

Keeps a frozen v1 payload that from_dict() must always be able to parse,
so a future version bump can't silently break backward-compatible loading.
"""

from proxyml_core._version import EXPORT_SCHEMA_VERSION
from proxyml_core.export import SurrogateExport

FROZEN_V1_PAYLOAD = {
    "task": "classification",
    "features": [
        {
            "name": "age",
            "type": "continuous",
            "scaler_mean": 40.0,
            "scaler_scale": 12.0,
            "ohe_categories": None,
            "ordinal_categories": None,
            "coefficient": 1.5,
            "category_coefficients": None,
            "per_class_coefficients": None,
        },
        {
            "name": "region",
            "type": "categorical",
            "scaler_mean": None,
            "scaler_scale": None,
            "ohe_categories": ["east", "west"],
            "ordinal_categories": None,
            "coefficient": None,
            "category_coefficients": [0.2, -0.2],
            "per_class_coefficients": None,
        },
    ],
    "classes": ["no", "yes"],
    "intercept": -0.3,
    "per_class_intercepts": None,
    "version": "v1",
    "trained_at": "2025-01-01T00:00:00Z",
    "schema_name": "default",
    "name": None,
    "comments": None,
    "metrics": {"f1": 0.9},
    "hyperparameters": None,
    "run_id": "run-1",
    "schema_definition": [],
    "schema_warning": None,
    "note": "some note",
    "export_schema_version": 1,
}


def test_to_dict_always_stamps_current_version():
    export = SurrogateExport(task="regression", features=[], intercept=0.0)
    assert export.to_dict()["export_schema_version"] == EXPORT_SCHEMA_VERSION


def test_frozen_v1_payload_always_parses():
    restored = SurrogateExport.from_dict(FROZEN_V1_PAYLOAD)
    assert restored.task == "classification"
    assert restored.classes == ["no", "yes"]
    assert len(restored.features) == 2
    assert restored.features[0].name == "age"
