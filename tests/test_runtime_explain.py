from __future__ import annotations

import numpy as np

from aoi.runtime.fusion import _display_anomaly_map


def test_display_anomaly_map_is_bounded_and_resized() -> None:
    primary = np.arange(16, dtype=np.float32).reshape(4, 4)
    dino = np.asarray([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
    result = _display_anomaly_map(primary, dino)
    assert result.shape == (4, 4)
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
