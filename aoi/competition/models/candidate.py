"""Stable lifecycle contract for AOI candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


class CandidateState(str, Enum):
    CREATED = "created"
    BASE_FITTED = "base_fitted"
    ADAPTED = "adapted"


@dataclass(frozen=True)
class CandidatePrediction:
    """Serializable image prediction shared by training, evaluation, and demo."""

    sample_id: str
    score: float
    threshold: float
    anomaly_map: np.ndarray
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_anomaly(self) -> bool:
        return self.score >= self.threshold

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if self.anomaly_map.ndim != 2:
            raise ValueError("anomaly_map must be HxW")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not np.isfinite(self.score) or not np.isfinite(self.threshold):
            raise ValueError("score and threshold must be finite")
        if not np.isfinite(self.anomaly_map).all():
            raise ValueError("anomaly_map contains non-finite values")

    def to_record(self, *, include_map: bool = False) -> dict[str, object]:
        self.validate()
        record = asdict(self)
        record["is_anomaly"] = self.is_anomaly
        anomaly_map = record.pop("anomaly_map")
        record["map_shape"] = list(self.anomaly_map.shape)
        if include_map:
            record["anomaly_map"] = np.asarray(anomaly_map).tolist()
        return record


class AOICandidate(ABC):
    """Required candidate API; subclasses own model-specific serialization."""

    candidate_id: str
    family: str
    state: CandidateState

    def __init__(self, candidate_id: str, family: str) -> None:
        if not candidate_id or not family:
            raise ValueError("candidate_id and family are required")
        self.candidate_id = candidate_id
        self.family = family
        self.state = CandidateState.CREATED

    @abstractmethod
    def fit_base(self, normal_images: Sequence[object]) -> dict[str, object]:
        """Fit only the category-independent/source representation."""

    @abstractmethod
    def adapt(
        self,
        normal_100: Sequence[object],
        defect_30: Sequence[object],
        *,
        defect_labels: Sequence[object] | None = None,
    ) -> dict[str, object]:
        """Adapt under the competition 100-normal plus 30-defect protocol."""

    @abstractmethod
    def predict_image(self, image: object, *, sample_id: str = "image") -> CandidatePrediction:
        """Predict one image with an image score and two-dimensional map."""

    def predict_batch(
        self, images: Sequence[object], *, sample_ids: Sequence[str] | None = None
    ) -> list[CandidatePrediction]:
        ids = list(sample_ids) if sample_ids is not None else [f"image-{i:06d}" for i in range(len(images))]
        if len(ids) != len(images):
            raise ValueError("sample_ids length must match images")
        return [self.predict_image(image, sample_id=sample_id) for image, sample_id in zip(images, ids, strict=True)]

    def predict_video(self, frames: Iterable[object]) -> Iterator[CandidatePrediction]:
        for index, frame in enumerate(frames):
            yield self.predict_image(frame, sample_id=f"frame-{index:06d}")

    @abstractmethod
    def export(self, output_path: str | Path, *, format: str = "onnx") -> Path:
        """Export a deployment representation or fail explicitly if unsupported."""

    @abstractmethod
    def save(self, output_path: str | Path) -> Path:
        """Persist all candidate state required for exact reload."""

    @classmethod
    @abstractmethod
    def load(cls, input_path: str | Path, **kwargs: object) -> "AOICandidate":
        """Reload a previously saved candidate."""
