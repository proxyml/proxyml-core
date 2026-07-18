from importlib.metadata import version as _pkg_version

from proxyml_core._version import (
    EXPORT_SCHEMA_VERSION,
    IncompatibleExportVersionError,
    check_compatible,
)
from proxyml_core.export import (
    ExportError,
    FeatureExportEntry,
    PerClassCoefficients,
    PerClassIntercept,
    SurrogateExport,
    feature_importances_from_export,
    predict_from_export,
    score_export,
)
from proxyml_core.schema import (
    CategoricalFeature,
    CategoricalOrdinalFeature,
    ContinuousFeature,
    CountFeature,
    Feature,
    FeatureSchema,
    FeatureValidationError,
    NumericOrdinalFeature,
)

__version__ = _pkg_version("proxyml-core")

__all__ = [
    "__version__",
    "Feature",
    "FeatureSchema",
    "FeatureValidationError",
    "ContinuousFeature",
    "CategoricalFeature",
    "CategoricalOrdinalFeature",
    "NumericOrdinalFeature",
    "CountFeature",
    "SurrogateExport",
    "FeatureExportEntry",
    "PerClassCoefficients",
    "PerClassIntercept",
    "ExportError",
    "score_export",
    "predict_from_export",
    "feature_importances_from_export",
    "EXPORT_SCHEMA_VERSION",
    "IncompatibleExportVersionError",
    "check_compatible",
]
