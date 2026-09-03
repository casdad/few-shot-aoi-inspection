"""Entity-aggregated mask-supervised residual prototypes on DINO features."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .dinov2_local_residual import DinoLocalResidualKnownHead


class DinoEntityResidualKnownHead(DinoLocalResidualKnownHead):
    """Reduce token-bank extreme values with one residual prototype per defect."""

    def _residual_bank(
        self,
        features: torch.Tensor,
        masks: Sequence[object],
        prototype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = int(round(np.sqrt(features.shape[1])))
        if grid * grid != features.shape[1]:
            raise ValueError("DINO patch count is not a square grid")
        selected = self._mask_tokens(masks, grid)
        entities: list[torch.Tensor] = []
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
            residual = features[index, indices] - prototype[indices]
            entities.append(F.normalize(residual.mean(dim=0), dim=0))
        bank = torch.stack(entities)
        owners = torch.arange(len(entities), device=self.device, dtype=torch.long)
        return bank, owners
