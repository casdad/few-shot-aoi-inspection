from __future__ import annotations

import copy

import pytest
import torch

from aoi.competition.models.known_defect_heads.dinov2_tiled_mlp import (
    DinoTiledMLPKnownHead,
)


def _head(*, hidden_dimension: int = 32) -> DinoTiledMLPKnownHead:
    """Construct a small fitted head without invoking the 100+30 trainer."""
    head = DinoTiledMLPKnownHead(
        tile_count=9,
        hidden_dimension=hidden_dimension,
        epochs=1,
        device="cpu",
    )
    dimension = 4
    generator = torch.Generator().manual_seed(20260830)
    head.feature_mean = torch.zeros(dimension)
    head.feature_scale = torch.ones(dimension)
    head.weight1 = torch.randn(dimension, hidden_dimension, generator=generator) * 0.05
    head.bias1 = torch.full((hidden_dimension,), 0.05)
    head.weight2 = torch.linspace(-0.03, 0.03, hidden_dimension)
    head.bias2 = torch.tensor(0.0)
    return head


def _feedback_features() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    normal = torch.zeros((2, 9, 4, 4), dtype=torch.float32)
    normal[..., 0] = 1.0
    defect = torch.zeros((2, 9, 4, 4), dtype=torch.float32)
    defect[..., 1] = 1.0
    masks = [torch.ones((8, 8)), torch.ones((8, 8))]
    return normal, defect, masks


def test_feedback_tuning_updates_exactly_33_output_parameters_deterministically() -> None:
    parent = _head()
    normal, defect, masks = _feedback_features()
    frozen = parent.frozen_parameter_fingerprint()
    parent_fingerprint = parent.parameter_fingerprint()

    first = parent.clone()
    first_receipt = first.fine_tune_feedback_output_layer(
        normal,
        defect,
        masks=masks,
        epochs=4,
        learning_rate=0.02,
        seed=17,
    )
    second = parent.clone()
    second_receipt = second.fine_tune_feedback_output_layer(
        normal,
        defect,
        masks=masks,
        epochs=4,
        learning_rate=0.02,
        seed=17,
    )

    assert parent.feedback_output_parameter_count() == 33
    assert first_receipt["updated_parameter_count"] == 33
    assert first_receipt["trainable_parameters"] == ["weight2", "bias2"]
    assert first_receipt["changed_tensors"] == {
        "feature_mean": False,
        "feature_scale": False,
        "weight1": False,
        "bias1": False,
        "weight2": True,
        "bias2": True,
    }
    assert first.frozen_parameter_fingerprint() == frozen
    assert first.parameter_fingerprint() == second.parameter_fingerprint()
    assert first_receipt["candidate_parameter_fingerprint"] == second_receipt[
        "candidate_parameter_fingerprint"
    ]
    assert first.parameter_fingerprint()["overall_sha256"] != parent_fingerprint[
        "overall_sha256"
    ]


def test_feedback_overlay_reloads_only_output_layer_and_is_atomic_on_tampering() -> None:
    parent = _head()
    normal, defect, masks = _feedback_features()
    candidate = parent.clone()
    candidate.fine_tune_feedback_output_layer(
        normal,
        defect,
        masks=masks,
        epochs=3,
        learning_rate=0.02,
        seed=23,
    )
    overlay = candidate.output_layer_overlay(
        parent_fingerprint=parent.parameter_fingerprint()
    )

    restored = parent.clone()
    restored.apply_output_layer_overlay(overlay)
    assert restored.parameter_fingerprint() == candidate.parameter_fingerprint()
    assert restored.frozen_parameter_fingerprint() == parent.frozen_parameter_fingerprint()

    tampered = copy.deepcopy(overlay)
    tampered_state = tampered["output_layer"]
    assert isinstance(tampered_state, dict)
    tampered_state["weight2"][0] += 0.25
    before = parent.clone()
    target = parent.clone()
    with pytest.raises(ValueError, match="candidate fingerprint mismatch"):
        target.apply_output_layer_overlay(tampered)
    assert target.parameter_fingerprint() == before.parameter_fingerprint()


def test_feedback_tuning_rejects_an_output_surface_other_than_33_parameters() -> None:
    head = _head(hidden_dimension=31)
    normal, defect, masks = _feedback_features()

    assert head.feedback_output_parameter_count() == 32
    with pytest.raises(RuntimeError, match="exactly 33"):
        head.fine_tune_feedback_output_layer(normal, defect, masks=masks)
