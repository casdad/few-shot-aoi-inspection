"""Shared baseline output contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AnomalyPrediction:
    score: float
    anomaly_map: np.ndarray
    threshold: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_anomaly(self) -> bool | None:
        if self.threshold is None:
            return None
        return self.score >= self.threshold

    def validate(self, expected_shape: tuple[int, int]) -> None:
        if self.anomaly_map.shape != expected_shape:
            raise ValueError(f"map shape {self.anomaly_map.shape} != image shape {expected_shape}")
        if not np.isfinite(self.score) or not np.isfinite(self.anomaly_map).all():
            raise ValueError("prediction contains non-finite values")
