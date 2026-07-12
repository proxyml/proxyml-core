"""Feature schema types.

Pure: stdlib + numpy only. These are plain dataclasses rather than pydantic
models so that a REST-only ``proxyml`` install never pulls in a compiled
validation dependency. Validation that pydantic gave for free (probabilities
summing to 1, matching lengths, etc.) is replicated by hand in
``__post_init__``.

All dataclasses are keyword-only (``kw_only=True``) so that subclasses can
add required fields after the base class's defaulted ``immutable`` field
without Python's dataclass field-ordering constraint forcing spurious
defaults onto otherwise-required fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import numpy as np


class FeatureValidationError(ValueError):
    """Raised when a Feature or FeatureSchema fails validation."""


@dataclass(kw_only=True)
class Feature:
    """Base type for a single feature in a schema.

    ``type`` is a class-level discriminator (not a constructor field) so
    every instance carries its wire-format tag without it participating in
    equality/repr/construction.
    """

    name: str
    immutable: bool = False

    type: ClassVar[str] = "feature"

    def sample(self, size: int = 1) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        for f in fields(self):
            d[f.name] = getattr(self, f.name)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Feature":
        ftype = d.get("type")
        subclass = _FEATURE_TYPES.get(ftype)
        if subclass is None:
            raise FeatureValidationError(f"Unknown feature type: {ftype!r}")
        return subclass._from_dict_fields(d)

    @classmethod
    def _from_dict_fields(cls, d: dict[str, Any]) -> "Feature":
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        return cls(**kwargs)


@dataclass(kw_only=True)
class ContinuousFeature(Feature):
    mean: float
    std: float
    min: float
    max: float

    type: ClassVar[str] = "continuous"

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.normal(loc=self.mean, scale=self.std, size=size)
        return np.clip(samples, self.min, self.max)


@dataclass(kw_only=True)
class CategoricalFeature(Feature):
    valid_categories: dict[str, float]

    type: ClassVar[str] = "categorical"

    def __post_init__(self) -> None:
        total = sum(self.valid_categories.values())
        if not np.isclose(total, 1.0):
            raise FeatureValidationError(
                f"Category probabilities must sum to 1.0, got {total}"
            )

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.choice(
            a=list(self.valid_categories.keys()),
            p=list(self.valid_categories.values()),
            size=size,
        )


@dataclass(kw_only=True)
class CategoricalOrdinalFeature(Feature):
    categories: list[str | int]
    probabilities: list[float]

    type: ClassVar[str] = "categorical_ordinal"

    def __post_init__(self) -> None:
        if len(self.categories) != len(self.probabilities):
            raise FeatureValidationError(
                "categories and probabilities must be same length"
            )
        if not np.isclose(sum(self.probabilities), 1.0):
            raise FeatureValidationError("probabilities must sum to 1.0")

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.choice(a=self.categories, p=self.probabilities, size=size)


@dataclass(kw_only=True)
class NumericOrdinalFeature(Feature):
    categories: list[int]
    mean: float
    std: float

    type: ClassVar[str] = "numeric_ordinal"

    def sample(self, size: int = 1) -> np.ndarray:
        continuous = np.random.normal(loc=self.mean, scale=self.std, size=size)
        categories = np.array(self.categories)
        indices = np.argmin(
            np.abs(continuous[:, None] - categories[None, :]), axis=1
        )
        return categories[indices]


@dataclass(kw_only=True)
class CountFeature(Feature):
    lambda_: float
    max: int | None = None

    type: ClassVar[str] = "count"

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.poisson(lam=self.lambda_, size=size)
        if self.max is not None:
            samples = np.clip(samples, 0, self.max)
        return samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "immutable": self.immutable,
            "lambda": self.lambda_,
            "max": self.max,
        }

    @classmethod
    def _from_dict_fields(cls, d: dict[str, Any]) -> "CountFeature":
        return cls(
            name=d["name"],
            immutable=d.get("immutable", False),
            lambda_=d["lambda"],
            max=d.get("max"),
        )


_FEATURE_TYPES: dict[str, type[Feature]] = {
    ContinuousFeature.type: ContinuousFeature,
    CategoricalFeature.type: CategoricalFeature,
    CategoricalOrdinalFeature.type: CategoricalOrdinalFeature,
    NumericOrdinalFeature.type: NumericOrdinalFeature,
    CountFeature.type: CountFeature,
}


@dataclass
class FeatureSchema:
    features: list[Feature]
    covariance_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.covariance_matrix is not None:
            n_continuous = sum(1 for f in self.features if isinstance(f, ContinuousFeature))
            if self.covariance_matrix.shape != (n_continuous, n_continuous):
                raise FeatureValidationError(
                    f"Covariance matrix must be ({n_continuous}, {n_continuous}), "
                    f"got {self.covariance_matrix.shape}"
                )
            if not np.allclose(self.covariance_matrix, self.covariance_matrix.T):
                raise FeatureValidationError("Covariance matrix must be symmetric")
            if not np.all(np.linalg.eigvals(self.covariance_matrix) >= 0):
                raise FeatureValidationError("Covariance matrix must be positive semi-definite")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"features": [f.to_dict() for f in self.features]}
        if self.covariance_matrix is not None:
            d["covariance_matrix"] = self.covariance_matrix.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureSchema":
        features = [Feature.from_dict(f) for f in d["features"]]
        covariance_matrix = d.get("covariance_matrix")
        if covariance_matrix is not None:
            covariance_matrix = np.asarray(covariance_matrix)
        return cls(features=features, covariance_matrix=covariance_matrix)
