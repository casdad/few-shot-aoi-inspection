from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aoi.feedback import apply_active_feedback, runtime_model_version


class _Dino:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.defect_global_bank = torch.tensor([[1.0, 0.0]])


class _Runtime:
    def __init__(self) -> None:
        self.bundle = {"run_id": "run-20260821"}
        self.bundle_path = Path("outputs/run-20260821/model_bundle.yaml")
        self.dino = _Dino()


class _Store:
    root = Path("outputs/feedback").resolve()

    def active_threshold(self, category: str, model_version: str) -> float | None:
        assert (category, model_version) == ("Class1", "run-20260821")
        return 0.42

    def active_feature_bank(self, category: str, model_version: str):
        assert (category, model_version) == ("Class1", "run-20260821")
        return (
            np.asarray([[0.0, 2.0], [1.0, 1.0]], dtype=np.float32),
            {"version": "features-v1", "sample_count": 2},
        )


def test_active_feedback_is_model_scoped_and_runtime_reversible() -> None:
    runtime = _Runtime()
    state = apply_active_feedback(runtime, _Store(), "Class1")

    assert state.model_version == "run-20260821"
    assert state.threshold == 0.42
    assert state.threshold_source == "active_operator_feedback"
    assert state.feature_version == "features-v1"
    assert state.feature_sample_count == 2
    assert runtime.dino.defect_global_bank.shape == (3, 2)
    assert torch.allclose(
        torch.linalg.norm(runtime.dino.defect_global_bank, dim=1),
        torch.ones(3),
    )

    disabled = apply_active_feedback(
        runtime,
        _Store(),
        "Class1",
        enabled=False,
    )
    assert disabled.threshold is None
    assert disabled.threshold_source == "model_bundle"
    assert runtime.dino.defect_global_bank.tolist() == [[1.0, 0.0]]


def test_runtime_model_version_uses_bundle_directory_as_legacy_fallback() -> None:
    runtime = _Runtime()
    runtime.bundle = {}
    assert runtime_model_version(runtime) == "run-20260821"
