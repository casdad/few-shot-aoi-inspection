"""Mask-supervised defect residuals relative to aligned normal DINO patches."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .dinov2_local_patch import DinoLocalPatchKnownHead


class DinoLocalResidualKnownHead(DinoLocalPatchKnownHead):
    def __init__(self, *, magnitude_weight: float = 0.25, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if magnitude_weight < 0:
            raise ValueError("magnitude weight must be nonnegative")
        self.magnitude_weight = magnitude_weight
        self.magnitude_center: torch.Tensor | None = None
        self.magnitude_scale: torch.Tensor | None = None

    def _residual_bank(
        self,
        features: torch.Tensor,
        masks: Sequence[object],
        prototype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = int(round(np.sqrt(features.shape[1])))
        selected = self._mask_tokens(masks, grid)
        residuals: list[torch.Tensor] = []
        owners: list[torch.Tensor] = []
        for index in range(len(features)):
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
            residuals.append(features[index, indices] - prototype[indices])
            owners.append(
                torch.full((len(indices),), index, device=self.device, dtype=torch.long)
            )
        return F.normalize(torch.cat(residuals), dim=1), torch.cat(owners)

    def _residual_score_with(
        self,
        features: torch.Tensor,
        prototype: torch.Tensor,
        residual_bank: torch.Tensor,
    ) -> torch.Tensor:
        if self.magnitude_center is None or self.magnitude_scale is None:
            raise RuntimeError("residual magnitude calibration is unavailable")
        normalized = self._normalized(features)
        residual = normalized - prototype[None]
        magnitude = torch.linalg.vector_norm(residual, dim=2)
        direction = F.normalize(residual, dim=2)
        similarity = torch.matmul(direction, residual_bank.T).max(dim=2).values
        magnitude_z = (magnitude - self.magnitude_center[None]) / self.magnitude_scale[None]
        patch_score = similarity + self.magnitude_weight * magnitude_z
        return patch_score.topk(
            min(self.score_topk, patch_score.shape[1]), dim=1
        ).values.mean(dim=1)

    def fit_features(
        self,
        normal_features: torch.Tensor,
        defect_features: torch.Tensor,
        defect_masks: Sequence[object],
    ) -> dict[str, object]:
        started = time.perf_counter()
        normal = self._normalized(normal_features.to(self.device))
        defect = self._normalized(defect_features.to(self.device))
        if len(normal) != 100 or len(defect) != 30 or len(defect_masks) != 30:
            raise ValueError("local residual head requires exactly 100 normal + 30 defect")
        self.normal_prototype = F.normalize(normal.mean(dim=0), dim=1)
        normal_magnitude = torch.linalg.vector_norm(
            normal - self.normal_prototype[None], dim=2
        )
        self.magnitude_center = normal_magnitude.median(dim=0).values
        deviation = torch.abs(normal_magnitude - self.magnitude_center[None])
        self.magnitude_scale = torch.clamp(
            deviation.median(dim=0).values * 1.4826, min=0.01
        )
        self.defect_bank, self.defect_owner = self._residual_bank(
            defect, defect_masks, self.normal_prototype
        )
        normal_sum = normal.sum(dim=0)
        normal_scores = []
        for index in range(len(normal)):
            leave_one_out = F.normalize(
                (normal_sum - normal[index]) / (len(normal) - 1), dim=1
            )
            normal_scores.append(
                self._residual_score_with(
                    normal[index : index + 1], leave_one_out, self.defect_bank
                )[0]
            )
        defect_scores = []
        for index in range(len(defect)):
            keep = self.defect_owner != index
            defect_scores.append(
                self._residual_score_with(
                    defect[index : index + 1],
                    self.normal_prototype,
                    self.defect_bank[keep],
                )[0]
            )
        self.calibration_normal_scores = torch.stack(normal_scores).cpu().numpy()
        self.calibration_defect_scores = torch.stack(defect_scores).cpu().numpy()
        self.fit_seconds = time.perf_counter() - started
        return {
            "fit_seconds_excluding_backbone": self.fit_seconds,
            "normal_prototype_items": int(self.normal_prototype.shape[0]),
            "defect_bank_items": int(self.defect_bank.shape[0]),
            "embedding_dimension": int(self.defect_bank.shape[1]),
            "magnitude_weight": self.magnitude_weight,
            "normal_score_min": float(self.calibration_normal_scores.min()),
            "normal_score_max": float(self.calibration_normal_scores.max()),
            "defect_score_min": float(self.calibration_defect_scores.min()),
            "defect_score_max": float(self.calibration_defect_scores.max()),
        }

    @torch.inference_mode()
    def score_features(self, features: torch.Tensor) -> np.ndarray:
        if self.normal_prototype is None or self.defect_bank is None:
            raise RuntimeError("local residual head must be fitted before scoring")
        return (
            self._residual_score_with(
                features.to(self.device), self.normal_prototype, self.defect_bank
            )
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def fit(
        self,
        dino: Any,
        normal_images: Sequence[object],
        defect_images: Sequence[object],
        defect_masks: Sequence[object],
    ) -> dict[str, object]:
        started = time.perf_counter()
        result = self.fit_features(
            dino._features(normal_images), dino._features(defect_images), defect_masks
        )
        result["adaptation_seconds"] = time.perf_counter() - started
        return result

    def save(self, path: str | Path) -> Path:
        if (
            self.normal_prototype is None
            or self.defect_bank is None
            or self.magnitude_center is None
            or self.magnitude_scale is None
        ):
            raise RuntimeError("cannot save an unfitted residual head")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
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
    ) -> "DinoLocalResidualKnownHead":
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        head = cls(
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
