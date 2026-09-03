"""Mask-supervised local patch prototypes on frozen DINOv2 features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F


class DinoLocalPatchKnownHead:
    """Detect local known defects without adding another backbone forward."""

    def __init__(
        self,
        *,
        score_topk: int = 4,
        max_patches_per_defect: int = 16,
        mask_dilation: int = 1,
        device: str = "cpu",
    ) -> None:
        if score_topk <= 0 or max_patches_per_defect <= 0 or mask_dilation < 0:
            raise ValueError("local known-head dimensions must be positive")
        self.score_topk = score_topk
        self.max_patches_per_defect = max_patches_per_defect
        self.mask_dilation = mask_dilation
        self.device = torch.device(device)
        self.normal_prototype: torch.Tensor | None = None
        self.defect_bank: torch.Tensor | None = None
        self.defect_owner: torch.Tensor | None = None
        self.calibration_normal_scores: np.ndarray | None = None
        self.calibration_defect_scores: np.ndarray | None = None
        self.fit_seconds: float | None = None

    @staticmethod
    def _normalized(features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("DINO patch features must have shape [N,P,D]")
        return F.normalize(features, dim=2)

    def _mask_tokens(self, masks: Sequence[object], grid: int) -> torch.Tensor:
        values = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks])
        tensor = torch.from_numpy(values[:, None]).to(self.device)
        tensor = F.adaptive_max_pool2d(tensor, (grid, grid))
        if self.mask_dilation:
            kernel = 2 * self.mask_dilation + 1
            tensor = F.max_pool2d(
                tensor, kernel_size=kernel, stride=1, padding=self.mask_dilation
            )
        return tensor[:, 0].reshape(len(values), -1) > 0

    def _defect_patches(
        self, features: torch.Tensor, masks: Sequence[object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = int(round(np.sqrt(features.shape[1])))
        if grid * grid != features.shape[1]:
            raise ValueError("DINO patch count is not a square grid")
        selected = self._mask_tokens(masks, grid)
        patches: list[torch.Tensor] = []
        owners: list[torch.Tensor] = []
        for index in range(len(features)):
            indices = torch.nonzero(selected[index], as_tuple=False).flatten()
            if not len(indices):
                raise ValueError(f"defect mask {index} contains no positive patch")
            if len(indices) > self.max_patches_per_defect:
                positions = torch.linspace(
                    0,
                    len(indices) - 1,
                    self.max_patches_per_defect,
                    device=indices.device,
                ).round().long()
                indices = indices[positions]
            patches.append(features[index, indices])
            owners.append(
                torch.full((len(indices),), index, device=self.device, dtype=torch.long)
            )
        return F.normalize(torch.cat(patches), dim=1), torch.cat(owners)

    def _score_with(
        self,
        features: torch.Tensor,
        normal_prototype: torch.Tensor,
        defect_bank: torch.Tensor,
    ) -> torch.Tensor:
        features = self._normalized(features)
        defect_similarity = torch.matmul(features, defect_bank.T).max(dim=2).values
        normal_similarity = torch.sum(features * normal_prototype[None], dim=2)
        relative = defect_similarity - normal_similarity
        return relative.topk(min(self.score_topk, relative.shape[1]), dim=1).values.mean(
            dim=1
        )

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
            raise ValueError("local known head requires exactly 100 normal + 30 defect")
        self.normal_prototype = F.normalize(normal.mean(dim=0), dim=1)
        self.defect_bank, self.defect_owner = self._defect_patches(defect, defect_masks)
        normal_sum = normal.sum(dim=0)
        normal_scores = []
        for index in range(len(normal)):
            leave_one_out = F.normalize(
                (normal_sum - normal[index]) / (len(normal) - 1), dim=1
            )
            normal_scores.append(
                self._score_with(
                    normal[index : index + 1], leave_one_out, self.defect_bank
                )[0]
            )
        defect_scores = []
        for index in range(len(defect)):
            keep = self.defect_owner != index
            if not bool(keep.any()):
                raise ValueError("leave-one-defect-out bank is empty")
            defect_scores.append(
                self._score_with(
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
            "normal_score_min": float(self.calibration_normal_scores.min()),
            "normal_score_max": float(self.calibration_normal_scores.max()),
            "defect_score_min": float(self.calibration_defect_scores.min()),
            "defect_score_max": float(self.calibration_defect_scores.max()),
        }

    @torch.inference_mode()
    def fit(
        self,
        dino: Any,
        normal_images: Sequence[object],
        defect_images: Sequence[object],
        defect_masks: Sequence[object],
    ) -> dict[str, object]:
        started = time.perf_counter()
        normal_features = dino._features(normal_images)
        defect_features = dino._features(defect_images)
        result = self.fit_features(normal_features, defect_features, defect_masks)
        result["adaptation_seconds"] = time.perf_counter() - started
        return result

    @torch.inference_mode()
    def score_features(self, features: torch.Tensor) -> np.ndarray:
        if self.normal_prototype is None or self.defect_bank is None:
            raise RuntimeError("local known head must be fitted before scoring")
        return (
            self._score_with(
                features.to(self.device), self.normal_prototype, self.defect_bank
            )
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def score(self, dino: Any, images: Sequence[object]) -> np.ndarray:
        return self.score_features(dino._features(images))

    def save(self, path: str | Path) -> Path:
        if self.normal_prototype is None or self.defect_bank is None:
            raise RuntimeError("cannot save an unfitted local known head")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "score_topk": self.score_topk,
                "max_patches_per_defect": self.max_patches_per_defect,
                "mask_dilation": self.mask_dilation,
                "normal_prototype": self.normal_prototype.cpu(),
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
    def load(cls, path: str | Path, *, device: str = "cpu") -> "DinoLocalPatchKnownHead":
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        head = cls(
            score_topk=int(payload["score_topk"]),
            max_patches_per_defect=int(payload["max_patches_per_defect"]),
            mask_dilation=int(payload["mask_dilation"]),
            device=device,
        )
        head.normal_prototype = payload["normal_prototype"].to(head.device)
        head.defect_bank = payload["defect_bank"].to(head.device)
        head.defect_owner = payload["defect_owner"].to(head.device)
        head.calibration_normal_scores = payload["calibration_normal_scores"].cpu().numpy()
        head.calibration_defect_scores = payload["calibration_defect_scores"].cpu().numpy()
        return head
