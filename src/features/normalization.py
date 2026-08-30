"""Development-only normalization bundles with deterministic serialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class NormalizationBundle:
    means: Mapping[str, float]
    scales: Mapping[str, float]
    fit_split: str = "development"

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, object]], features: Sequence[str], *, split: str):
        if split != "development":
            raise ValueError("normalization may be fitted on development only")
        means, scales = {}, {}
        for feature in features:
            observed = [
                float(row[feature]) for row in rows
                if int(row.get(f"{feature}__missing", 0)) == 0
            ]
            if not observed:
                raise ValueError(f"no observed development values for {feature}")
            mean = sum(observed) / len(observed)
            variance = sum((value - mean) ** 2 for value in observed) / len(observed)
            means[feature] = mean
            scales[feature] = math.sqrt(variance) or 1.0
        return cls(means=means, scales=scales, fit_split=split)

    def transform(self, row: Mapping[str, object]) -> dict[str, object]:
        output = dict(row)
        for feature, mean in self.means.items():
            if int(row.get(f"{feature}__missing", 0)):
                output[feature] = 0.0
            else:
                output[feature] = (float(row[feature]) - mean) / self.scales[feature]
        return output

    def canonical_json(self) -> str:
        return json.dumps({
            "fit_split": self.fit_split,
            "means": dict(sorted(self.means.items())),
            "scales": dict(sorted(self.scales.items())),
        }, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

