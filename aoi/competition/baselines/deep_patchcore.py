"""Method-faithful compact PatchCore-style baseline with ResNet-18 features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .base import AnomalyPrediction
from .image_ops import to_tensor


class _ResNet18Features(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = (value - self.mean) / self.std
        value = self.stem(value)
        value = self.layer1(value)
        layer2 = self.layer2(value)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([layer2, layer3], dim=1)
        return F.normalize(features, p=2, dim=1)


class DeepPatchCoreLite:
    def __init__(
        self,
        input_size: tuple[int, int] = (128, 128),
        max_memory: int = 512,
        projection_dim: int = 64,
        seed: int = 20260803,
        *,
        pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        if max_memory <= 0 or projection_dim <= 0:
            raise ValueError("memory and projection dimensions must be positive")
        self.input_size = input_size
        self.max_memory = max_memory
        self.projection_dim = projection_dim
        self.seed = seed
        self.pretrained = pretrained
        self.device = torch.device(device)
        self.extractor = _ResNet18Features(pretrained).to(self.device).eval()
        for parameter in self.extractor.parameters():
            parameter.requires_grad_(False)
        self.memory: torch.Tensor | None = None
        self.threshold: float | None = None

    @torch.inference_mode()
    def _batch_features(self, images: list[np.ndarray]) -> tuple[torch.Tensor, tuple[int, int]]:
        tensors = torch.cat([to_tensor(image, self.input_size, self.device) for image in images], dim=0)
        feature_map = self.extractor(tensors)
        grid_shape = tuple(feature_map.shape[-2:])
        features = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
        return features, (int(grid_shape[0]), int(grid_shape[1]))

    @staticmethod
    def _greedy_coreset(projected: torch.Tensor, count: int) -> torch.Tensor:
        if len(projected) <= count:
            return torch.arange(len(projected), device=projected.device)
        center = projected.mean(dim=0, keepdim=True)
        first = int(torch.argmin(torch.sum((projected - center) ** 2, dim=1)))
        selected = [first]
        min_distance = torch.sum((projected - projected[first]) ** 2, dim=1)
        for _ in range(1, count):
            index = int(torch.argmax(min_distance))
            selected.append(index)
            distance = torch.sum((projected - projected[index]) ** 2, dim=1)
            min_distance = torch.minimum(min_distance, distance)
        return torch.tensor(selected, device=projected.device, dtype=torch.long)

    @torch.inference_mode()
    def fit(self, normal_images: list[np.ndarray], *, batch_size: int = 8) -> None:
        if not normal_images or batch_size <= 0:
            raise ValueError("normal images and positive batch size are required")
        feature_batches = []
        for start in range(0, len(normal_images), batch_size):
            features, _ = self._batch_features(normal_images[start : start + batch_size])
            feature_batches.append(features)
        features = torch.cat(feature_batches, dim=0)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        projection = torch.randn(
            features.shape[1],
            min(self.projection_dim, features.shape[1]),
            generator=generator,
            device=self.device,
        ) / np.sqrt(features.shape[1])
        projected = features @ projection
        selected = self._greedy_coreset(projected, self.max_memory)
        self.memory = features[selected].contiguous()
        normal_scores = [self.predict(image, use_threshold=False).score for image in normal_images]
        self.threshold = float(np.quantile(normal_scores, 0.995))

    @torch.inference_mode()
    def predict(self, image: np.ndarray, *, use_threshold: bool = True) -> AnomalyPrediction:
        if self.memory is None:
            raise RuntimeError("baseline is not fitted")
        features, grid_shape = self._batch_features([image])
        distance = torch.cdist(features, self.memory).amin(dim=1).reshape(1, 1, *grid_shape)
        height, width = image.shape[:2]
        anomaly = F.interpolate(distance, size=(height, width), mode="bilinear", align_corners=False)
        anomaly_map = anomaly[0, 0].detach().cpu().numpy().astype(np.float32)
        score = float(np.quantile(anomaly_map, 0.995))
        prediction = AnomalyPrediction(
            score=score,
            anomaly_map=anomaly_map,
            threshold=self.threshold if use_threshold else None,
            metadata={"backend": "resnet18_patchcore_lite", "memory_items": len(self.memory)},
        )
        prediction.validate((height, width))
        return prediction

    def save(self, path: Path) -> None:
        if self.memory is None:
            raise RuntimeError("baseline is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "input_size": self.input_size,
                "max_memory": self.max_memory,
                "projection_dim": self.projection_dim,
                "seed": self.seed,
                "pretrained": self.pretrained,
                "extractor": self.extractor.state_dict(),
                "memory": self.memory.detach().cpu(),
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, map_location: str = "cpu") -> "DeepPatchCoreLite":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(
            tuple(payload["input_size"]),
            int(payload["max_memory"]),
            int(payload["projection_dim"]),
            int(payload["seed"]),
            pretrained=False,
            device=map_location,
        )
        model.extractor.load_state_dict(payload["extractor"])
        model.memory = payload["memory"].to(model.device)
        model.threshold = payload["threshold"]
        return model
