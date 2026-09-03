"""Dependency-light image conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def as_rgb_float(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] not in {1, 3, 4}:
        raise ValueError(f"expected HxW, HxWx1/3/4 image, got {value.shape}")
    if value.shape[2] == 1:
        value = np.repeat(value, 3, axis=2)
    elif value.shape[2] == 4:
        value = value[:, :, :3]
    value = value.astype(np.float32, copy=False)
    if value.max(initial=0.0) > 1.0:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def to_tensor(image: np.ndarray, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    value = as_rgb_float(image)
    resized = cv2.resize(value, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
