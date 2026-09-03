"""Frozen official DINOv2 ViT-S spatial-prototype few-shot AOI candidate."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from aoi.competition.evaluation import best_f1_threshold
from aoi.competition.models.anomaly_detectors.multiscale_patch_memory import select_operating_point
from aoi.competition.models.candidate import AOICandidate, CandidatePrediction, CandidateState
from aoi.paths import portable_path, resolve_artifact


DINOV2_VITS14_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOV2_REPOSITORY_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def relative_knn_scores(
    query: torch.Tensor, normal_bank: torch.Tensor, defect_bank: torch.Tensor
) -> torch.Tensor:
    """Return maximum defect similarity minus maximum normal similarity."""

    if query.ndim != 2 or normal_bank.ndim != 2 or defect_bank.ndim != 2:
        raise ValueError("query and banks must be rank-2 tensors")
    if not (query.shape[1] == normal_bank.shape[1] == defect_bank.shape[1]):
        raise ValueError("query and bank embedding dimensions must match")
    if normal_bank.shape[0] == 0 or defect_bank.shape[0] == 0:
        raise ValueError("normal and defect banks must be non-empty")
    normal_similarity = query @ normal_bank.T
    defect_similarity = query @ defect_bank.T
    return defect_similarity.max(dim=1).values - normal_similarity.max(dim=1).values


class DINOv2SpatialPrototype(AOICandidate):
    def __init__(
        self,
        candidate_id: str,
        *,
        repository_path: str | Path,
        weight_path: str | Path,
        input_size: int = 252,
        score_topk: int = 16,
        target_normal_fpr: float = 0.20,
        seed: int = 20261531,
        device: str = "cpu",
    ) -> None:
        super().__init__(candidate_id, "C_frozen_foundation_fewshot")
        if input_size % 14:
            raise ValueError("DINOv2 input size must be divisible by patch size 14")
        self.repository_path = Path(repository_path).resolve(strict=True)
        self.weight_path = Path(weight_path).resolve(strict=True)
        if hashlib.sha256(self.weight_path.read_bytes()).hexdigest() != DINOV2_VITS14_SHA256:
            raise ValueError("DINOv2 weight SHA-256 mismatch")
        self.input_size = input_size
        self.score_topk = score_topk
        self.target_normal_fpr = target_normal_fpr
        self.seed = seed
        self.device = torch.device(device)
        self.backbone = torch.hub.load(
            str(self.repository_path), "dinov2_vits14", source="local", pretrained=False
        ).to(self.device).eval()
        state = torch.load(self.weight_path, map_location=self.device, weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.prototype: torch.Tensor | None = None
        self.location: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.threshold: float | None = None
        self.pixel_threshold: float | None = None
        self.normal_global_bank: torch.Tensor | None = None
        self.defect_global_bank: torch.Tensor | None = None
        self.known_threshold: float | None = None
        self.calibration: dict[str, object] = {}

    def fit_base(self, normal_images: Sequence[object]) -> dict[str, object]:
        self.state = CandidateState.BASE_FITTED
        return {
            "backbone": "official_dinov2_vits14",
            "parameters": sum(parameter.numel() for parameter in self.backbone.parameters()),
            "weight_sha256": DINOV2_VITS14_SHA256,
            "repository_commit": DINOV2_REPOSITORY_COMMIT,
            "frozen": True,
            "source_normal_count": len(normal_images),
        }

    def _tensor(self, image: object) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        array = np.ascontiguousarray(array[..., :3])
        tensor = torch.from_numpy(array).to(self.device, dtype=torch.float32).permute(2, 0, 1)[None]
        if float(tensor.max()) > 1.5:
            tensor = tensor / 255.0
        tensor = F.interpolate(tensor, (self.input_size, self.input_size), mode="bilinear", align_corners=False)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]
        return (tensor - mean) / std

    @torch.inference_mode()
    def _features(self, images: Sequence[object], batch_size: int = 16) -> torch.Tensor:
        output = []
        for start in range(0, len(images), batch_size):
            batch = torch.cat([self._tensor(image) for image in images[start : start + batch_size]])
            tokens = self.backbone.forward_features(batch)["x_norm_patchtokens"]
            output.append(F.normalize(tokens, dim=2))
        if not output:
            raise ValueError("at least one image is required")
        return torch.cat(output)

    def _maps_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.prototype is None or self.location is None or self.scale is None:
            raise RuntimeError("candidate has no adapted spatial prototype")
        distance = 1.0 - torch.sum(features * self.prototype[None], dim=2)
        normalized = torch.clamp((distance - self.location[None]) / self.scale[None], min=0.0)
        grid = int(np.sqrt(features.shape[1]))
        maps = normalized.reshape(len(features), 1, grid, grid)
        return F.interpolate(maps, (self.input_size, self.input_size), mode="bilinear", align_corners=False)[:, 0]

    def _scores(self, maps: torch.Tensor) -> np.ndarray:
        flat = maps.flatten(1)
        return flat.topk(min(self.score_topk, flat.shape[1]), dim=1).values.mean(dim=1).cpu().numpy()

    def adapt(
        self,
        normal_100: Sequence[object],
        defect_30: Sequence[object],
        *,
        defect_labels: Sequence[object] | None = None,
    ) -> dict[str, object]:
        if len(normal_100) != 100 or len(defect_30) != 30:
            raise ValueError("competition adaptation requires exactly 100 normal and 30 defect images")
        started = time.perf_counter()
        normal_features = self._features(normal_100)
        self.prototype = F.normalize(normal_features.mean(dim=0), dim=1)
        normal_distance = 1.0 - torch.sum(normal_features * self.prototype[None], dim=2)
        self.location = normal_distance.median(dim=0).values
        deviation = torch.abs(normal_distance - self.location[None])
        self.scale = torch.clamp(deviation.median(dim=0).values * 1.4826, min=0.01)
        defect_features = self._features(defect_30)
        normal_maps = self._maps_from_features(normal_features)
        defect_maps = self._maps_from_features(defect_features)
        normal_scores = self._scores(normal_maps)
        defect_scores = self._scores(defect_maps)
        self.threshold, operating_point = select_operating_point(
            normal_scores,
            defect_scores,
            fpr_targets=(self.target_normal_fpr,),
            minimum_recall=0.0,
        )
        self.normal_global_bank = F.normalize(normal_features.mean(dim=1), dim=1)
        self.defect_global_bank = F.normalize(defect_features.mean(dim=1), dim=1)
        normal_similarity = self.normal_global_bank @ self.normal_global_bank.T
        normal_similarity.fill_diagonal_(-torch.inf)
        normal_known_scores = (
            (self.normal_global_bank @ self.defect_global_bank.T).max(dim=1).values
            - normal_similarity.max(dim=1).values
        ).cpu().numpy()
        defect_similarity = self.defect_global_bank @ self.defect_global_bank.T
        defect_similarity.fill_diagonal_(-torch.inf)
        defect_known_scores = (
            defect_similarity.max(dim=1).values
            - (self.defect_global_bank @ self.normal_global_bank.T).max(dim=1).values
        ).cpu().numpy()
        self.known_threshold, known_operating_point = select_operating_point(
            normal_known_scores,
            defect_known_scores,
            fpr_targets=(0.05,),
            minimum_recall=0.0,
        )
        if defect_labels is not None:
            masks = np.stack([np.asarray(mask, dtype=np.uint8) for mask in defect_labels])
            mask_tensor = torch.from_numpy(masks[:, None].astype(np.float32)).to(self.device)
            mask_tensor = F.interpolate(mask_tensor, (self.input_size, self.input_size), mode="nearest")[:, 0]
            labels = np.concatenate(
                [np.zeros(normal_maps.numel(), dtype=np.uint8), mask_tensor.cpu().numpy().reshape(-1).astype(np.uint8)]
            )
            values = np.concatenate([normal_maps.cpu().numpy().reshape(-1), defect_maps.cpu().numpy().reshape(-1)])
            if len(values) > 1_500_000:
                stride = int(np.ceil(len(values) / 1_500_000))
                keep = np.concatenate((np.flatnonzero(labels == 1), np.arange(0, len(values), stride)))
                keep = np.unique(keep)
                labels, values = labels[keep], values[keep]
            self.pixel_threshold, pixel_evidence = best_f1_threshold(labels, values, max_candidates=256)
        else:
            self.pixel_threshold = float(np.quantile(normal_maps.cpu().numpy(), 0.999))
            pixel_evidence = {"available": False}
        self.state = CandidateState.ADAPTED
        self.calibration = {
            "operating_point": operating_point,
            "pixel": pixel_evidence,
            "normal_score_min": float(normal_scores.min()),
            "normal_score_max": float(normal_scores.max()),
            "known_score_min": float(defect_scores.min()),
            "known_score_max": float(defect_scores.max()),
            "known_head": {
                "method": "global_token_relative_knn_leave_one_sample_out",
                "operating_point": known_operating_point,
                "normal_score_min": float(normal_known_scores.min()),
                "normal_score_max": float(normal_known_scores.max()),
                "defect_score_min": float(defect_known_scores.min()),
                "defect_score_max": float(defect_known_scores.max()),
            },
        }
        return {
            "adaptation_seconds": time.perf_counter() - started,
            "threshold": self.threshold,
            "pixel_threshold": self.pixel_threshold,
            "calibration": self.calibration,
            "prototype_items": int(self.prototype.shape[0]),
        }

    def predict_batch(
        self, images: Sequence[object], *, sample_ids: Sequence[str] | None = None
    ) -> list[CandidatePrediction]:
        if self.state != CandidateState.ADAPTED or self.threshold is None:
            raise RuntimeError("candidate must be adapted before prediction")
        ids = list(sample_ids) if sample_ids is not None else [f"image-{index:06d}" for index in range(len(images))]
        if len(ids) != len(images):
            raise ValueError("sample_ids length must match images")
        started = time.perf_counter()
        features = self._features(images)
        return self.predict_features(
            features,
            sample_ids=ids,
            started_at=started,
        )

    @torch.inference_mode()
    def predict_features(
        self,
        features: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
        started_at: float | None = None,
    ) -> list[CandidatePrediction]:
        """Predict from already extracted patch tokens.

        The runtime uses this to share one DINO batch between the full image and
        its nine local tiles. The scoring path is identical to ``predict_batch``.
        """

        if self.state != CandidateState.ADAPTED or self.threshold is None:
            raise RuntimeError("candidate must be adapted before prediction")
        if features.ndim != 3:
            raise ValueError("DINO features must have shape [N,P,D]")
        ids = (
            list(sample_ids)
            if sample_ids is not None
            else [f"image-{index:06d}" for index in range(len(features))]
        )
        if len(ids) != len(features):
            raise ValueError("sample_ids length must match features")
        started = time.perf_counter() if started_at is None else started_at
        maps = self._maps_from_features(features)
        scores = self._scores(maps)
        if (
            self.normal_global_bank is None
            or self.defect_global_bank is None
            or self.known_threshold is None
        ):
            raise RuntimeError("candidate has no adapted known-defect head")
        global_features = F.normalize(features.mean(dim=1), dim=1)
        known_scores = relative_knn_scores(
            global_features, self.normal_global_bank, self.defect_global_bank
        ).cpu().numpy()
        elapsed = (time.perf_counter() - started) * 1000 / len(features)
        maps_np = maps.cpu().numpy().astype(np.float32)
        return [
            CandidatePrediction(
                sample_id=sample_id,
                score=float(scores[index]),
                threshold=float(self.threshold),
                anomaly_map=maps_np[index],
                latency_ms=elapsed,
                metadata={
                    "candidate_id": self.candidate_id,
                    "input_size": self.input_size,
                    "backbone": "dinov2_vits14",
                    "pixel_threshold": self.pixel_threshold,
                    "known_score": float(known_scores[index]),
                    "known_threshold": float(self.known_threshold),
                    "known_decision": bool(known_scores[index] >= self.known_threshold),
                    "batched_latency": True,
                },
            )
            for index, sample_id in enumerate(ids)
        ]

    def predict_image(self, image: object, *, sample_id: str = "image") -> CandidatePrediction:
        return self.predict_batch([image], sample_ids=[sample_id])[0]

    def export(self, output_path: str | Path, *, format: str = "onnx") -> Path:
        raise NotImplementedError(f"{format} export is a Level-4 task and has not been claimed")

    def save(self, output_path: str | Path) -> Path:
        if self.prototype is None or self.threshold is None:
            raise RuntimeError("only an adapted candidate can be saved")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "candidate_id": self.candidate_id,
                "repository_path": portable_path(self.repository_path),
                "repository_commit": DINOV2_REPOSITORY_COMMIT,
                "weight_path": portable_path(self.weight_path),
                "weight_sha256": DINOV2_VITS14_SHA256,
                "input_size": self.input_size,
                "score_topk": self.score_topk,
                "target_normal_fpr": self.target_normal_fpr,
                "seed": self.seed,
                "prototype": self.prototype.cpu(),
                "location": self.location.cpu(),
                "scale": self.scale.cpu(),
                "threshold": self.threshold,
                "pixel_threshold": self.pixel_threshold,
                "calibration": self.calibration,
                "normal_global_bank": self.normal_global_bank.cpu(),
                "defect_global_bank": self.defect_global_bank.cpu(),
                "known_threshold": self.known_threshold,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, input_path: str | Path, **kwargs: object) -> "DINOv2SpatialPrototype":
        device = str(kwargs.get("device", "cpu"))
        payload = torch.load(Path(input_path), map_location=device, weights_only=True)
        candidate = cls(
            str(payload["candidate_id"]),
            repository_path=resolve_artifact(payload["repository_path"]),
            weight_path=resolve_artifact(payload["weight_path"]),
            input_size=int(payload["input_size"]),
            score_topk=int(payload["score_topk"]),
            target_normal_fpr=float(payload["target_normal_fpr"]),
            seed=int(payload["seed"]),
            device=device,
        )
        candidate.prototype = payload["prototype"].to(candidate.device)
        candidate.location = payload["location"].to(candidate.device)
        candidate.scale = payload["scale"].to(candidate.device)
        candidate.threshold = float(payload["threshold"])
        candidate.pixel_threshold = float(payload["pixel_threshold"])
        candidate.calibration = payload["calibration"]
        if "normal_global_bank" in payload:
            candidate.normal_global_bank = payload["normal_global_bank"].to(candidate.device)
            candidate.defect_global_bank = payload["defect_global_bank"].to(candidate.device)
            candidate.known_threshold = float(payload["known_threshold"])
        candidate.state = CandidateState.ADAPTED
        return candidate
