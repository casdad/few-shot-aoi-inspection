"""Full-coverage tiled DINO residual prototypes for small known defects."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .dinov2_entity_residual import DinoEntityResidualKnownHead


class DinoTiledEntityResidualKnownHead(DinoEntityResidualKnownHead):
    """Score aligned high-resolution tiles with entity residual prototypes."""

    def __init__(self, *, tile_count: int = 4, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if tile_count <= 0:
            raise ValueError("tile count must be positive")
        self.tile_count = tile_count

    def _score_tiled_with(
        self,
        features: torch.Tensor,
        prototype: torch.Tensor,
        residual_bank: torch.Tensor,
    ) -> torch.Tensor:
        if self.magnitude_center is None or self.magnitude_scale is None:
            raise RuntimeError("tiled residual magnitude calibration is unavailable")
        normalized = F.normalize(features, dim=3)
        residual = normalized - prototype[None]
        magnitude = torch.linalg.vector_norm(residual, dim=3)
        direction = F.normalize(residual, dim=3)
        similarity = torch.matmul(direction, residual_bank.T).max(dim=3).values
        magnitude_z = (magnitude - self.magnitude_center[None]) / self.magnitude_scale[
            None
        ]
        flattened = (similarity + self.magnitude_weight * magnitude_z).flatten(1)
        return flattened.topk(min(self.score_topk, flattened.shape[1]), dim=1).values.mean(
            dim=1
        )

    def _score_single_with(
        self,
        features: torch.Tensor,
        tile_indices: torch.Tensor,
        prototype: torch.Tensor,
        residual_bank: torch.Tensor,
    ) -> torch.Tensor:
        selected_prototype = prototype[tile_indices]
        selected_center = self.magnitude_center[tile_indices]
        selected_scale = self.magnitude_scale[tile_indices]
        normalized = F.normalize(features, dim=2)
        residual = normalized - selected_prototype
        magnitude = torch.linalg.vector_norm(residual, dim=2)
        direction = F.normalize(residual, dim=2)
        similarity = torch.matmul(direction, residual_bank.T).max(dim=2).values
        magnitude_z = (magnitude - selected_center) / selected_scale
        patch_score = similarity + self.magnitude_weight * magnitude_z
        return patch_score.topk(
            min(self.score_topk, patch_score.shape[1]), dim=1
        ).values.mean(dim=1)

    def fit_features(
        self,
        normal_features: torch.Tensor,
        defect_features: torch.Tensor,
        defect_masks: Sequence[object],
        defect_tile_indices: Sequence[int],
    ) -> dict[str, object]:
        started = time.perf_counter()
        if normal_features.ndim != 4:
            raise ValueError("normal tile features must have shape [N,T,P,D]")
        if normal_features.shape[1] != self.tile_count:
            raise ValueError("normal feature tile count differs from configuration")
        if (
            len(normal_features) != 100
            or len(defect_features) != 30
            or len(defect_masks) != 30
            or len(defect_tile_indices) != 30
        ):
            raise ValueError("tiled residual head requires exactly 100 normal + 30 defect")
        normal = F.normalize(normal_features.to(self.device), dim=3)
        defect = F.normalize(defect_features.to(self.device), dim=2)
        tiles = torch.as_tensor(defect_tile_indices, device=self.device, dtype=torch.long)
        if bool(((tiles < 0) | (tiles >= self.tile_count)).any()):
            raise ValueError("defect tile index is outside the configured tile grid")
        self.normal_prototype = F.normalize(normal.mean(dim=0), dim=2)
        normal_magnitude = torch.linalg.vector_norm(
            normal - self.normal_prototype[None], dim=3
        )
        self.magnitude_center = normal_magnitude.median(dim=0).values
        deviation = torch.abs(normal_magnitude - self.magnitude_center[None])
        self.magnitude_scale = torch.clamp(
            deviation.median(dim=0).values * 1.4826, min=0.01
        )
        grid = int(round(np.sqrt(defect.shape[1])))
        if grid * grid != defect.shape[1]:
            raise ValueError("DINO patch count is not a square grid")
        selected = self._mask_tokens(defect_masks, grid)
        entities = []
        for index in range(len(defect)):
            indices = torch.nonzero(selected[index], as_tuple=False).flatten()
            if not len(indices):
                raise ValueError(f"defect mask {index} contains no residual patch")
            if len(indices) > self.max_patches_per_defect:
                positions = torch.linspace(
                    0,
                    len(indices) - 1,
                    self.max_patches_per_defect,
                    device=indices.device,
                ).round().long()
                indices = indices[positions]
            residual = defect[index, indices] - self.normal_prototype[tiles[index], indices]
            entities.append(F.normalize(residual.mean(dim=0), dim=0))
        self.defect_bank = torch.stack(entities)
        self.defect_owner = torch.arange(30, device=self.device, dtype=torch.long)
        normal_sum = normal.sum(dim=0)
        normal_scores = []
        for index in range(len(normal)):
            leave_one_out = F.normalize(
                (normal_sum - normal[index]) / (len(normal) - 1), dim=2
            )
            normal_scores.append(
                self._score_tiled_with(
                    normal[index : index + 1], leave_one_out, self.defect_bank
                )[0]
            )
        defect_scores = []
        for index in range(len(defect)):
            keep = self.defect_owner != index
            defect_scores.append(
                self._score_single_with(
                    defect[index : index + 1],
                    tiles[index : index + 1],
                    self.normal_prototype,
                    self.defect_bank[keep],
                )[0]
            )
        self.calibration_normal_scores = torch.stack(normal_scores).cpu().numpy()
        self.calibration_defect_scores = torch.stack(defect_scores).cpu().numpy()
        self.fit_seconds = time.perf_counter() - started
        return {
            "fit_seconds_excluding_backbone": self.fit_seconds,
            "tile_count": self.tile_count,
            "normal_prototype_items": int(self.normal_prototype.shape[1]),
            "defect_bank_items": int(self.defect_bank.shape[0]),
            "embedding_dimension": int(self.defect_bank.shape[1]),
            "magnitude_weight": self.magnitude_weight,
            "normal_score_min": float(self.calibration_normal_scores.min()),
            "normal_score_max": float(self.calibration_normal_scores.max()),
            "defect_score_min": float(self.calibration_defect_scores.min()),
            "defect_score_max": float(self.calibration_defect_scores.max()),
        }

    @torch.inference_mode()
    def fit(
        self,
        dino: Any,
        normal_tile_images: Sequence[Sequence[object]],
        defect_tile_images: Sequence[object],
        defect_tile_masks: Sequence[object],
        defect_tile_indices: Sequence[int],
    ) -> dict[str, object]:
        started = time.perf_counter()
        flat_normal = [image for group in normal_tile_images for image in group]
        normal = dino._features(flat_normal)
        normal = normal.reshape(len(normal_tile_images), self.tile_count, *normal.shape[1:])
        result = self.fit_features(
            normal,
            dino._features(defect_tile_images),
            defect_tile_masks,
            defect_tile_indices,
        )
        result["adaptation_seconds"] = time.perf_counter() - started
        return result

    @torch.inference_mode()
    def score_tiled_features(self, features: torch.Tensor) -> np.ndarray:
        if self.normal_prototype is None or self.defect_bank is None:
            raise RuntimeError("tiled residual head must be fitted before scoring")
        return (
            self._score_tiled_with(
                features.to(self.device), self.normal_prototype, self.defect_bank
            )
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def score(self, dino: Any, tiled_images: Sequence[Sequence[object]]) -> np.ndarray:
        flat = [image for group in tiled_images for image in group]
        features = dino._features(flat)
        features = features.reshape(len(tiled_images), self.tile_count, *features.shape[1:])
        return self.score_tiled_features(features)

    def save(self, path: str | Path) -> Path:
        if (
            self.normal_prototype is None
            or self.defect_bank is None
            or self.magnitude_center is None
            or self.magnitude_scale is None
        ):
            raise RuntimeError("cannot save an unfitted tiled residual head")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "tile_count": self.tile_count,
                "score_topk": self.score_topk,
                "max_patches_per_defect": self.max_patches_per_defect,
                "mask_dilation": self.mask_dilation,
                "magnitude_weight": self.magnitude_weight,
                "normal_prototype": self.normal_prototype.cpu(),
                "magnitude_center": self.magnitude_center.cpu(),
                "magnitude_scale": self.magnitude_scale.cpu(),
                "defect_bank": self.defect_bank.cpu(),
                "defect_owner": self.defect_owner.cpu(),
                "calibration_normal_scores": torch.from_numpy(
                    self.calibration_normal_scores
                ),
                "calibration_defect_scores": torch.from_numpy(
                    self.calibration_defect_scores
                ),
            },
            output,
        )
        return output

    @classmethod
    def load(
        cls, path: str | Path, *, device: str = "cpu"
    ) -> "DinoTiledEntityResidualKnownHead":
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        head = cls(
            tile_count=int(payload["tile_count"]),
            score_topk=int(payload["score_topk"]),
            max_patches_per_defect=int(payload["max_patches_per_defect"]),
            mask_dilation=int(payload["mask_dilation"]),
            magnitude_weight=float(payload["magnitude_weight"]),
            device=device,
        )
        for name in (
            "normal_prototype",
            "magnitude_center",
            "magnitude_scale",
            "defect_bank",
            "defect_owner",
        ):
            setattr(head, name, payload[name].to(head.device))
        head.calibration_normal_scores = payload["calibration_normal_scores"].cpu().numpy()
        head.calibration_defect_scores = payload["calibration_defect_scores"].cpu().numpy()
        return head
