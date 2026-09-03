from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _decode(path: str | Path, flags: int) -> np.ndarray:
    source = Path(path)
    encoded = np.frombuffer(source.read_bytes(), dtype=np.uint8)
    value = cv2.imdecode(encoded, flags)
    if value is None:
        raise ValueError(f"failed to decode image: {source}")
    return value


def read_rgb(path: str | Path) -> np.ndarray:
    return cv2.cvtColor(_decode(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    value = _decode(path, cv2.IMREAD_GRAYSCALE)
    if value.shape != shape:
        value = cv2.resize(
            value,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (value > 0).astype(np.uint8)
