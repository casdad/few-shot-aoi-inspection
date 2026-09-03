"""Deterministic multi-scale ResNet18 patch-memory AOI candidate."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from aoi.competition.evaluation import best_f1_threshold
from aoi.competition.models.candidate import AOICandidate, CandidatePrediction, CandidateState


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OFFICIAL_RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class MultiScaleResNet18(nn.Module):
    def __init__(self, weight_path: Path | None) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        if weight_path is not None:
            if hashlib.sha256(weight_path.read_bytes()).hexdigest() != OFFICIAL_RESNET18_SHA256:
                raise ValueError("official ResNet18 weight SHA-256 mismatch")
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("std", torch.tensor(IMAGENET_STD)[None, :, None, None])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = (value - self.mean) / self.std
        value = self.stem(value)
        layer1 = self.layer1(value)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        target = layer2.shape[-2:]
        layer1 = F.adaptive_avg_pool2d(layer1, target)
        layer3 = F.interpolate(layer3, size=target, mode="bilinear", align_corners=False)
        return torch.cat((layer1, layer2, layer3), dim=1)


def select_operating_point(
    normal_scores: np.ndarray,
    defect_scores: np.ndarray,
    *,
    fpr_targets: Sequence[float],
    minimum_recall: float,
) -> tuple[float, dict[str, object]]:
    normals = np.asarray(normal_scores, dtype=np.float64)
    defects = np.asarray(defect_scores, dtype=np.float64)
    if not len(normals) or not len(defects):
        raise ValueError("normal and known-defect scores are required")
    points: list[dict[str, float]] = []
    descending = np.sort(normals)[::-1]
    for target in fpr_targets:
        maximum_fp = int(np.floor(float(target) * len(normals) + 1e-9))
        if maximum_fp <= 0:
            threshold = float(np.nextafter(descending[0], np.inf))
        elif maximum_fp >= len(normals):
            threshold = float(descending[-1])
        else:
            threshold = float(descending[maximum_fp - 1])
            while float(np.mean(normals >= threshold)) > float(target) + 1e-12:
                threshold = float(np.nextafter(threshold, np.inf))
        points.append(
            {
                "target_fpr": float(target),
                "threshold": threshold,
                "actual_fpr": float(np.mean(normals >= threshold)),
                "known_recall": float(np.mean(defects >= threshold)),
            }
        )
    feasible = [point for point in points if point["known_recall"] >= minimum_recall]
    if feasible:
        chosen = min(feasible, key=lambda point: (point["actual_fpr"], -point["known_recall"], point["threshold"]))
        fallback = False
    else:
        chosen = max(points, key=lambda point: (point["known_recall"], -point["actual_fpr"], point["threshold"]))
        fallback = True
    return float(chosen["threshold"]), {
        "points": points,
        "chosen": chosen,
        "minimum_known_recall": minimum_recall,
        "recall_constraint_unmet": fallback,
    }


class MultiScalePatchMemory(AOICandidate):
    """Frozen CNN features plus category-specific normal patch memory."""

    def __init__(
        self,
        candidate_id: str,
        *,
        weight_path: str | Path | None,
        input_size: int = 256,
        projection_dimension: int = 96,
        maximum_patch_pool: int = 8192,
        coreset_size: int = 512,
        score_topk: int = 32,
        seed: int = 20261511,
        device: str = "cpu",
    ) -> None:
        super().__init__(candidate_id, "A_multiscale_patch_memory")
        if min(input_size, projection_dimension, maximum_patch_pool, coreset_size, score_topk) <= 0:
            raise ValueError("candidate dimensions must be positive")
        self.input_size = input_size
        self.projection_dimension = projection_dimension
        self.maximum_patch_pool = maximum_patch_pool
        self.coreset_size = coreset_size
        self.score_topk = score_topk
        self.seed = seed
        self.device = torch.device(device)
        self.weight_path = Path(weight_path).resolve() if weight_path is not None else None
        self.extractor = MultiScaleResNet18(self.weight_path).to(self.device).eval()
        for parameter in self.extractor.parameters():
            parameter.requires_grad_(False)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        projection = torch.randn(448, projection_dimension, generator=generator, device=self.device)
        self.projection = F.normalize(projection, dim=0)
        self.memory: torch.Tensor | None = None
        self.threshold: float | None = None
        self.pixel_threshold: float | None = None
        self.calibration: dict[str, object] = {}

    def fit_base(self, normal_images: Sequence[object]) -> dict[str, object]:
        self.state = CandidateState.BASE_FITTED
        return {
            "backbone": "torchvision_resnet18_imagenet1k_v1",
            "weight_sha256": OFFICIAL_RESNET18_SHA256,
            "source_normal_count": len(normal_images),
            "frozen": True,
        }

    def _tensor(self, image: object) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("image must be HxWx3/4 or HxW")
        array = np.ascontiguousarray(array[..., :3])
        tensor = torch.from_numpy(array).to(self.device, dtype=torch.float32).permute(2, 0, 1)[None]
        if float(tensor.max()) > 1.5:
            tensor = tensor / 255.0
        return F.interpolate(tensor, (self.input_size, self.input_size), mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def _features(self, images: Sequence[object], batch_size: int = 16) -> torch.Tensor:
        batches = []
        for start in range(0, len(images), batch_size):
            values = torch.cat([self._tensor(image) for image in images[start : start + batch_size]], dim=0)
            maps = self.extractor(values)
            patches = maps.permute(0, 2, 3, 1).reshape(len(values), -1, maps.shape[1])
            batches.append(F.normalize(patches @ self.projection, dim=2))
        if not batches:
            raise ValueError("at least one image is required")
        return torch.cat(batches, dim=0)

    @staticmethod
    def _greedy_coreset(features: torch.Tensor, count: int) -> torch.Tensor:
        if len(features) <= count:
            return torch.arange(len(features), device=features.device)
        center = features.mean(dim=0, keepdim=True)
        first = int(torch.argmin(torch.sum((features - center) ** 2, dim=1)))
        selected = [first]
        minimum = torch.sum((features - features[first]) ** 2, dim=1)
        for _ in range(1, count):
            index = int(torch.argmax(minimum))
            selected.append(index)
            minimum = torch.minimum(minimum, torch.sum((features - features[index]) ** 2, dim=1))
        return torch.tensor(selected, device=features.device, dtype=torch.long)

    @torch.inference_mode()
    def _maps_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.memory is None:
            raise RuntimeError("candidate has no adapted memory")
        output = []
        for batch in features.split(8):
            distance = torch.cdist(batch.reshape(-1, batch.shape[-1]), self.memory).amin(dim=1)
            grid = int(np.sqrt(batch.shape[1]))
            output.append(distance.reshape(len(batch), 1, grid, grid))
        maps = torch.cat(output)
        return F.interpolate(maps, (self.input_size, self.input_size), mode="bilinear", align_corners=False)[:, 0]

    def _scores(self, maps: torch.Tensor) -> np.ndarray:
        flat = maps.flatten(1)
        count = min(self.score_topk, flat.shape[1])
        return flat.topk(count, dim=1).values.mean(dim=1).detach().cpu().numpy()

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
        pool = normal_features.reshape(-1, normal_features.shape[-1])
        if len(pool) > self.maximum_patch_pool:
            indices = torch.linspace(0, len(pool) - 1, self.maximum_patch_pool, device=self.device).round().long()
            pool = pool[indices]
        selected = self._greedy_coreset(pool, min(self.coreset_size, len(pool)))
        self.memory = pool[selected].contiguous()
        normal_maps = self._maps_from_features(normal_features)
        defect_maps = self._maps_from_features(self._features(defect_30))
        normal_scores = self._scores(normal_maps)
        defect_scores = self._scores(defect_maps)
        self.threshold, operating_point = select_operating_point(
            normal_scores,
            defect_scores,
            fpr_targets=(0.05, 0.10, 0.15, 0.20),
            minimum_recall=0.80,
        )
        pixel_evidence: dict[str, object] = {"available": False}
        if defect_labels is not None:
            masks = np.stack([np.asarray(value, dtype=np.uint8) for value in defect_labels])
            if masks.shape[0] != 30:
                raise ValueError("defect_labels must contain 30 masks")
            mask_tensor = torch.from_numpy(masks[:, None].astype(np.float32)).to(self.device)
            mask_tensor = F.interpolate(mask_tensor, (self.input_size, self.input_size), mode="nearest")[:, 0]
            labels = np.concatenate(
                [np.zeros(normal_maps.numel(), dtype=np.uint8), mask_tensor.detach().cpu().numpy().reshape(-1).astype(np.uint8)]
            )
            values = np.concatenate(
                [normal_maps.detach().cpu().numpy().reshape(-1), defect_maps.detach().cpu().numpy().reshape(-1)]
            )
            if len(values) > 1_000_000:
                stride = int(np.ceil(len(values) / 1_000_000))
                keep = np.concatenate((np.flatnonzero(labels == 1), np.arange(0, len(values), stride)))
                keep = np.unique(keep)
                labels, values = labels[keep], values[keep]
            self.pixel_threshold, evidence = best_f1_threshold(labels, values, max_candidates=128)
            pixel_evidence = {"available": True, **evidence}
        else:
            self.pixel_threshold = float(np.quantile(normal_maps.detach().cpu().numpy(), 0.999))
        self.state = CandidateState.ADAPTED
        self.calibration = {
            "operating_point": operating_point,
            "pixel": pixel_evidence,
            "normal_score_min": float(normal_scores.min()),
            "normal_score_max": float(normal_scores.max()),
            "known_score_min": float(defect_scores.min()),
            "known_score_max": float(defect_scores.max()),
        }
        return {
            "adaptation_seconds": time.perf_counter() - started,
            "memory_items": len(self.memory),
            "threshold": self.threshold,
            "pixel_threshold": self.pixel_threshold,
            "calibration": self.calibration,
        }

    def predict_batch(
        self, images: Sequence[object], *, sample_ids: Sequence[str] | None = None
    ) -> list[CandidatePrediction]:
        if self.state != CandidateState.ADAPTED or self.threshold is None:
            raise RuntimeError("candidate must be adapted before prediction")
        ids = list(sample_ids) if sample_ids is not None else [f"image-{i:06d}" for i in range(len(images))]
        if len(ids) != len(images):
            raise ValueError("sample_ids length must match images")
        started = time.perf_counter()
        maps = self._maps_from_features(self._features(images))
        scores = self._scores(maps)
        elapsed = (time.perf_counter() - started) * 1000 / len(images)
        maps_np = maps.detach().cpu().numpy().astype(np.float32)
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
                    "memory_items": len(self.memory) if self.memory is not None else 0,
                    "batched_latency": True,
                    "pixel_threshold": self.pixel_threshold,
                },
            )
            for index, sample_id in enumerate(ids)
        ]

    def predict_image(self, image: object, *, sample_id: str = "image") -> CandidatePrediction:
        return self.predict_batch([image], sample_ids=[sample_id])[0]

    def export(self, output_path: str | Path, *, format: str = "onnx") -> Path:
        raise NotImplementedError(f"{format} export is a Level-4 task and has not been claimed")

    def save(self, output_path: str | Path) -> Path:
        if self.memory is None or self.threshold is None:
            raise RuntimeError("only an adapted candidate can be saved")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "candidate_id": self.candidate_id,
                "input_size": self.input_size,
                "projection_dimension": self.projection_dimension,
                "maximum_patch_pool": self.maximum_patch_pool,
                "coreset_size": self.coreset_size,
                "score_topk": self.score_topk,
                "seed": self.seed,
                "extractor": self.extractor.state_dict(),
                "projection": self.projection.detach().cpu(),
                "memory": self.memory.detach().cpu(),
                "threshold": self.threshold,
                "pixel_threshold": self.pixel_threshold,
                "calibration": self.calibration,
                "weight_sha256": OFFICIAL_RESNET18_SHA256,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, input_path: str | Path, **kwargs: object) -> "MultiScalePatchMemory":
        device = str(kwargs.get("device", "cpu"))
        payload = torch.load(Path(input_path), map_location=device, weights_only=True)
        candidate = cls(
            str(payload["candidate_id"]),
            weight_path=None,
            input_size=int(payload["input_size"]),
            projection_dimension=int(payload["projection_dimension"]),
            maximum_patch_pool=int(payload["maximum_patch_pool"]),
            coreset_size=int(payload["coreset_size"]),
            score_topk=int(payload.get("score_topk", 32)),
            seed=int(payload["seed"]),
            device=device,
        )
        candidate.extractor.load_state_dict(payload["extractor"])
        candidate.projection = payload["projection"].to(candidate.device)
        candidate.memory = payload["memory"].to(candidate.device)
        candidate.threshold = float(payload["threshold"])
        candidate.pixel_threshold = float(payload["pixel_threshold"])
        candidate.calibration = payload["calibration"]
        candidate.state = CandidateState.ADAPTED
        return candidate
