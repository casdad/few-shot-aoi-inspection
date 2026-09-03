from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aoi.competition.models.known_defect_heads import DinoTiledMLPKnownHead  # noqa: E402
from aoi.runtime.fusion import apply_continuous_fusion as runtime_apply_fusion  # noqa: E402
from aoi.training.calibration import (  # noqa: E402
    CONTINUOUS_FUSION_SCHEMA,
    apply_continuous_fusion,
    constrained_threshold,
    fit_five_fold_continuous_fusion,
    fit_runtime_score_transform,
    identity_score_transform,
    select_global_category_thresholds,
    stratified_five_fold_ids,
    transform_runtime_score,
)


def _manual_fusion() -> dict[str, object]:
    branch = {
        "threshold": 0.0,
        "scale": 1.0,
        "runtime_transform": identity_score_transform(),
    }
    return {
        "schema": CONTINUOUS_FUSION_SCHEMA,
        "branches": {name: dict(branch) for name in ("general", "local", "known")},
        "fusion": {
            "aggregation": "nonnegative_linear",
            "weights": {"general": 0.4, "local": 0.3, "known": 0.3},
            "threshold": 0.1,
            "margin_clip": 6.0,
        },
    }


def test_stratified_five_fold_is_exact_and_disjoint() -> None:
    labels = np.asarray([0] * 100 + [1] * 30, dtype=np.uint8)
    folds = stratified_five_fold_ids(labels)
    assert folds.shape == (130,)
    assert set(folds.tolist()) == set(range(5))
    for fold in range(5):
        validation = folds == fold
        assert int(np.sum(validation & (labels == 0))) == 20
        assert int(np.sum(validation & (labels == 1))) == 6
        assert int(np.sum(~validation)) == 104


def test_constrained_threshold_minimizes_fpr_then_maximizes_f1() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9, 0.7, 1.0, 1.1, 1.2])
    threshold, fit = constrained_threshold(labels, scores, minimum_recall=0.75)
    assert threshold == 1.0
    assert fit["constraint_met"] is True
    assert fit["recall"] == 0.75
    assert fit["fpr"] == 0.0


def test_support_only_five_fold_search_records_no_leakage() -> None:
    labels = np.asarray([0] * 100 + [1] * 30, dtype=np.uint8)
    sample_ids = [f"sample-{index:03d}" for index in range(130)]
    normal = np.linspace(-1.0, 0.0, 100)
    defect = np.linspace(1.0, 2.0, 30)
    result = fit_five_fold_continuous_fusion(
        labels=labels,
        sample_ids=sample_ids,
        branch_scores={
            "general": np.concatenate((normal, defect)),
            "local": np.concatenate((normal * 2.0 + 3.0, defect * 2.0 + 3.0)),
            "known": np.concatenate((normal * 0.5 - 2.0, defect * 0.5 - 2.0)),
        },
    )
    assert result["selection_scope"] == "100_normal_plus_30_defect_support_only"
    assert result["independent_evaluation_used_for_selection"] is False
    assert result["fold_count"] == 5
    assert result["fusion"]["candidate_count"] == 45
    assert result["fusion"]["oof_recall_constraint_met"] is True
    validation_ids: list[str] = []
    for fold in result["folds"]:
        assert fold["training_count"] == 104
        assert fold["validation_count"] == 26
        assert fold["validation_normal"] == 20
        assert fold["validation_defect"] == 6
        validation_ids.extend(fold["validation_sample_ids"])
    assert sorted(validation_ids) == sorted(sample_ids)
    assert len(set(validation_ids)) == 130
    assert sum(bool(row["selected"]) for row in result["candidate_ledger"]) == 1


def test_equal_oof_candidates_prefer_supervised_local_branch_without_rethresholding() -> None:
    labels = np.asarray([0] * 100 + [1] * 30, dtype=np.uint8)
    sample_ids = [f"sample-{index:03d}" for index in range(130)]
    shared = np.concatenate(
        (np.linspace(-2.0, -1.0, 100), np.linspace(1.0, 2.0, 30))
    )
    result = fit_five_fold_continuous_fusion(
        labels=labels,
        sample_ids=sample_ids,
        branch_scores={name: shared.copy() for name in ("general", "local", "known")},
    )

    assert result["fusion"]["weights"] == {
        "general": 0.0,
        "local": 1.0,
        "known": 0.0,
    }
    assert result["fusion"]["threshold"] == 0.0
    assert result["fusion"]["threshold_protocol"] == (
        "calibrated_branch_zero_margin"
    )
    assert result["fusion"]["support_refit_metrics_apparent"]["recall"] == 1.0


def test_final_threshold_can_refit_on_runtime_aligned_support_scores() -> None:
    labels = np.asarray([0] * 100 + [1] * 30, dtype=np.uint8)
    sample_ids = [f"sample-{index:03d}" for index in range(130)]
    normal = np.linspace(-1.0, 0.0, 100)
    defect = np.linspace(0.2, 1.0, 30)
    calibration_local = np.concatenate((normal, defect))
    runtime_local = np.concatenate((normal, defect + 3.0))
    result = fit_five_fold_continuous_fusion(
        labels=labels,
        sample_ids=sample_ids,
        branch_scores={
            "general": np.concatenate((normal, defect)),
            "local": calibration_local,
            "known": np.concatenate((normal, defect)),
        },
        refit_branch_scores={
            "general": np.concatenate((normal, defect)),
            "local": runtime_local,
            "known": np.concatenate((normal, defect)),
        },
        fold_fit_minimum_recall=0.98,
        final_fit_minimum_recall=0.95,
    )
    assert result["fusion"]["support_refit_score_space"] == (
        "final_runtime_aligned_support_scores"
    )
    assert result["fusion"]["support_refit_metrics_apparent"]["recall"] >= 0.95
    assert len(result["support_refit_predictions"]) == 130
    assert result["support_refit_predictions"][-1]["branch_scores"]["local"] > (
        result["oof_predictions"][-1]["branch_scores"]["local"]
    )


def test_global_threshold_allocator_keeps_pooled_recall_hard_gate() -> None:
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.uint8)
    result = select_global_category_thresholds(
        {
            "easy": (labels, np.asarray([0.0] * 10 + [2.0] * 10)),
            "overlap": (
                labels,
                np.asarray([0.0] * 8 + [1.0] * 2 + [0.5] * 2 + [2.0] * 8),
            ),
        },
        pooled_minimum_recall=0.95,
        category_minimum_recall=0.80,
    )
    assert result["selection_scope"] == "support_only"
    assert result["independent_evaluation_used_for_selection"] is False
    assert result["pooled_support_metrics"]["recall"] >= 0.95
    assert set(result["categories"]) == {"easy", "overlap"}


def test_continuous_fusion_is_not_binary_or_or_majority_vote() -> None:
    calibration = _manual_fusion()
    one_positive = apply_continuous_fusion(
        {"general": 1.0, "local": -1.0, "known": -1.0}, calibration
    )
    two_positive = apply_continuous_fusion(
        {"general": 1.0, "local": 1.0, "known": -1.0}, calibration
    )
    assert one_positive["decision"] is False
    assert two_positive["decision"] is True
    assert isinstance(one_positive["score"], float)
    assert one_positive["score"] != float(any((True, False, False)))


def test_training_and_deployment_fusion_are_exactly_equal() -> None:
    calibration = _manual_fusion()
    scores = {"general": 0.8, "local": 0.2, "known": -0.4}
    assert apply_continuous_fusion(scores, calibration) == runtime_apply_fusion(
        scores, calibration
    )


def test_runtime_alignment_maps_final_head_to_oof_normal_space() -> None:
    labels = np.asarray([0] * 100 + [1] * 30, dtype=np.uint8)
    oof = np.concatenate((np.linspace(-2.0, 2.0, 100), np.linspace(3.0, 5.0, 30)))
    runtime = oof * 4.0 + 10.0
    transform = fit_runtime_score_transform(oof, runtime, labels)
    mapped = np.asarray([transform_runtime_score(value, transform) for value in runtime])
    assert np.allclose(mapped, oof, atol=1e-10)


def test_mlp_oof_defect_calibration_uses_runtime_tile_shape() -> None:
    torch.manual_seed(7)
    head = DinoTiledMLPKnownHead(
        tile_count=2,
        score_topk=2,
        max_patches_per_defect=4,
        mask_dilation=0,
        hidden_dimension=4,
        epochs=1,
        learning_rate=0.01,
        negative_patches_per_tile=1,
        seed=11,
        device="cpu",
    )
    normal = torch.randn(100, 2, 4, 8)
    defect_training = torch.randn(30, 4, 8) + 0.5
    defect_runtime = torch.randn(30, 2, 4, 8) + 0.5
    masks = [np.ones((2, 2), dtype=np.uint8) for _ in range(30)]
    receipt = head.fit_features(
        normal,
        defect_training,
        masks,
        defect_calibration_features=defect_runtime,
    )
    assert receipt["calibration_score_protocol"] == (
        "five_fold_out_of_fold_runtime_tile_scoring"
    )
    assert head.calibration_defect_scores.shape == (30,)
    assert head.final_runtime_normal_scores.shape == (100,)
    assert head.final_runtime_defect_scores.shape == (30,)
