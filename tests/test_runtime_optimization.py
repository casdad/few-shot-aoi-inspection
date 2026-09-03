from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from aoi.competition.deployment import SharedTorchMapBackend, TorchMapBackend
from aoi.competition.models.anomaly_detectors.dinov2_spatial import (
    DINOv2SpatialPrototype,
)
from aoi.competition.models.candidate import CandidateState
from aoi.runtime.runtime import _verify_bundle_integrity
from aoi.training.manifests import payload_hash


class _CountingExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.convolution(value)


def _patchcore_model(
    extractor: nn.Module, memory: torch.Tensor
) -> SimpleNamespace:
    return SimpleNamespace(extractor=extractor, memory=memory)


def test_shared_patchcore_backend_matches_two_independent_backends() -> None:
    torch.manual_seed(17)
    source_extractor = _CountingExtractor()
    adapted_extractor = copy.deepcopy(source_extractor)
    source = _patchcore_model(source_extractor, torch.randn(5, 4))
    adapted = _patchcore_model(adapted_extractor, torch.randn(7, 4))
    images = [
        np.random.default_rng(11).integers(0, 256, (12, 12, 3), dtype=np.uint8),
        np.random.default_rng(12).integers(0, 256, (12, 12, 3), dtype=np.uint8),
    ]

    expected_source = TorchMapBackend(source, device="cpu").predict_maps(images)
    expected_adapted = TorchMapBackend(adapted, device="cpu").predict_maps(images)
    source_extractor.calls = 0
    shared = SharedTorchMapBackend(source, adapted, device="cpu")
    actual_source, actual_adapted = shared.predict_maps(images)

    assert source_extractor.calls == 1
    assert np.allclose(actual_source, expected_source, atol=1e-6)
    assert np.allclose(actual_adapted, expected_adapted, atol=1e-6)


def test_shared_patchcore_backend_rejects_different_extractors() -> None:
    source_extractor = _CountingExtractor()
    adapted_extractor = copy.deepcopy(source_extractor)
    with torch.no_grad():
        adapted_extractor.convolution.weight.add_(1.0)
    source = _patchcore_model(source_extractor, torch.randn(5, 4))
    adapted = _patchcore_model(adapted_extractor, torch.randn(7, 4))

    with pytest.raises(ValueError, match="identical extractors"):
        SharedTorchMapBackend(source, adapted, device="cpu")


def test_dino_predict_features_preserves_prediction_contract() -> None:
    candidate = object.__new__(DINOv2SpatialPrototype)
    candidate.state = CandidateState.ADAPTED
    candidate.threshold = 0.4
    candidate.pixel_threshold = 0.2
    candidate.candidate_id = "test-dino"
    candidate.input_size = 2
    candidate.score_topk = 2
    candidate.prototype = torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
    candidate.location = torch.zeros(4)
    candidate.scale = torch.ones(4)
    candidate.normal_global_bank = torch.nn.functional.normalize(
        torch.randn(3, 3), dim=1
    )
    candidate.defect_global_bank = torch.nn.functional.normalize(
        torch.randn(2, 3), dim=1
    )
    candidate.known_threshold = 0.1
    features = torch.nn.functional.normalize(torch.randn(2, 4, 3), dim=2)

    predictions = candidate.predict_features(
        features, sample_ids=["first", "second"]
    )

    assert [prediction.sample_id for prediction in predictions] == ["first", "second"]
    assert all(prediction.anomaly_map.shape == (2, 2) for prediction in predictions)
    assert all(prediction.threshold == 0.4 for prediction in predictions)
    assert all("known_score" in prediction.metadata for prediction in predictions)


def test_bundle_integrity_verifies_every_recorded_artifact(tmp_path) -> None:
    artifacts = {
        "source.pt": b"source",
        "primary.pt": b"primary",
        "adapted.pt": b"adapted",
        "dino.pt": b"dino",
        "head.pt": b"head",
        "calibration.yaml": b"calibration",
        "support.yaml": b"support",
    }
    for name, value in artifacts.items():
        (tmp_path / name).write_bytes(value)

    def digest(name: str) -> str:
        return hashlib.sha256(artifacts[name]).hexdigest()

    bundle = {
        "source_model": "source.pt",
        "models": {
            "primary": "primary.pt",
            "adapted": "adapted.pt",
            "dino": "dino.pt",
            "head": "head.pt",
        },
        "calibration": "calibration.yaml",
        "support_receipt": "support.yaml",
        "artifact_sha256": {
            "source_model": digest("source.pt"),
            "models": {
                "primary": digest("primary.pt"),
                "adapted": digest("adapted.pt"),
                "dino": digest("dino.pt"),
                "head": digest("head.pt"),
            },
            "calibration": digest("calibration.yaml"),
            "support_receipt": digest("support.yaml"),
        },
        "bundle_hash": "pending",
    }
    bundle["bundle_hash"] = payload_hash(
        {key: value for key, value in bundle.items() if key != "bundle_hash"}
    )
    bundle_path = tmp_path / "model_bundle.yaml"
    bundle_path.write_text(
        yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8"
    )

    receipt = _verify_bundle_integrity(bundle, bundle_path)

    assert receipt["bundle_hash_verified"] is True
    assert receipt["artifact_hashes_verified"] is True
    (tmp_path / "head.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="models.head"):
        _verify_bundle_integrity(bundle, bundle_path)
