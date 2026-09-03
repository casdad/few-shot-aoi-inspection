from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ASSETS = (
    PROJECT_ROOT / "assets" / "resnet18-f37072fd.pth",
    PROJECT_ROOT / "assets" / "dinov2_vits14_pretrain.pth",
    PROJECT_ROOT / "assets" / "source_model.pt",
)


@pytest.fixture
def require_training_assets() -> None:
    """Skip tests that intentionally parse configs requiring unpublished weights."""

    missing = [path.name for path in TRAINING_ASSETS if not path.is_file()]
    if missing:
        pytest.skip(
            "public release excludes required training assets: " + ", ".join(missing)
        )
