"""Calibration formulas used by the clean original AOI scheme."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product

import cv2
import numpy as np

from aoi.competition.evaluation import best_f1_threshold
from aoi.competition.models.anomaly_detectors.multiscale_patch_memory import (
    select_operating_point,
)


BRANCH_NAMES = ("general", "local", "known")
CONTINUOUS_FUSION_SCHEMA = "continuous_score_fusion_v2"


@dataclass(frozen=True)
class RobustNormalizer:
    center: float
    scale: float
    method: str = "normal_median_mad"

    @classmethod
    def fit(cls, normal_scores: np.ndarray) -> RobustNormalizer:
        values = np.asarray(normal_scores, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("finite non-empty normal score vector required")
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        return cls(center=center, scale=max(1e-9, 1.4826 * mad))

    def transform(self, scores: np.ndarray | list[float]) -> np.ndarray:
        return (np.asarray(scores, dtype=np.float64) - self.center) / self.scale

    def payload(self) -> dict[str, float | str]:
        return asdict(self)


def calibrate_image_scores(
    labels: np.ndarray, raw_scores: np.ndarray
) -> tuple[RobustNormalizer, float, dict[str, object]]:
    label_values = np.asarray(labels, dtype=np.uint8)
    score_values = np.asarray(raw_scores, dtype=np.float64)
    normalizer = RobustNormalizer.fit(score_values[label_values == 0])
    normalized = normalizer.transform(score_values)
    threshold, fit = best_f1_threshold(label_values, normalized)
    evidence = {
        "threshold_fit": fit,
        "score_prototype": {
            "normal": float(normalized[label_values == 0].mean()),
            "known_defect": float(normalized[label_values == 1].mean()),
            "space": "normalized_anomaly_score",
        },
    }
    return normalizer, float(threshold), evidence


def stratified_pixel_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    max_per_class: int = 100_000,
) -> tuple[float, dict[str, object]]:
    label_values = np.asarray(labels, dtype=np.uint8).reshape(-1)
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if label_values.shape != score_values.shape:
        raise ValueError("pixel labels/scores must align")
    generator = np.random.default_rng(seed)
    selected = []
    for value in (0, 1):
        indices = np.flatnonzero(label_values == value)
        if not len(indices):
            raise ValueError(f"pixel calibration is missing class {value}")
        if len(indices) > max_per_class:
            indices = np.sort(generator.choice(indices, size=max_per_class, replace=False))
        selected.append(indices)
    indices = np.concatenate(selected)
    threshold, metrics = best_f1_threshold(label_values[indices], score_values[indices])
    return float(threshold), {
        "method": "stratified_adaptation_pixels_best_f1",
        "sampled_pixels": int(len(indices)),
        "sampled_normal": int(len(selected[0])),
        "sampled_defect": int(len(selected[1])),
        "fit": metrics,
    }


def pixel_normalizer(maps: np.ndarray, labels: np.ndarray) -> RobustNormalizer:
    normal_pixels = maps[np.asarray(labels) == 0].reshape(-1)
    if len(normal_pixels) > 500_000:
        indices = np.linspace(0, len(normal_pixels) - 1, 500_000, dtype=np.int64)
        normal_pixels = normal_pixels[indices]
    return RobustNormalizer.fit(normal_pixels)


def target_point(
    normal_scores: np.ndarray,
    defect_scores: np.ndarray,
    target_fpr: float,
    minimum_known_recall: float = 0.0,
) -> dict[str, float | bool]:
    threshold, evidence = select_operating_point(
        normal_scores,
        defect_scores,
        fpr_targets=(target_fpr,),
        minimum_recall=minimum_known_recall,
    )
    point = dict(evidence["points"][0])
    point["threshold"] = float(threshold)
    point["known_constraint_met"] = bool(float(point["known_recall"]) >= minimum_known_recall)
    return point


def normal_only_threshold(normal_scores: np.ndarray, target_fpr: float) -> float:
    normals = np.asarray(normal_scores, dtype=np.float64)
    if normals.ndim != 1 or not len(normals) or not np.isfinite(normals).all():
        raise ValueError("finite one-dimensional normal scores are required")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    descending = np.sort(normals)[::-1]
    maximum_fp = int(math.floor(target_fpr * len(normals) + 1e-9))
    if maximum_fp <= 0:
        return float(np.nextafter(descending[0], np.inf))
    threshold = float(descending[maximum_fp - 1])
    while float(np.mean(normals >= threshold)) > target_fpr + 1e-12:
        threshold = float(np.nextafter(threshold, np.inf))
    return threshold


def binary_metrics(
    labels: np.ndarray | list[int], predictions: np.ndarray | list[bool]
) -> dict[str, float | int]:
    """Return the image-level metrics used by constrained calibration."""

    label_values = np.asarray(labels, dtype=np.uint8)
    prediction_values = np.asarray(predictions, dtype=bool)
    if label_values.ndim != 1 or prediction_values.shape != label_values.shape:
        raise ValueError("labels/predictions must be aligned one-dimensional vectors")
    if not len(label_values) or not np.isin(label_values, (0, 1)).all():
        raise ValueError("non-empty binary labels are required")
    tp = int(np.sum((label_values == 1) & prediction_values))
    fp = int(np.sum((label_values == 0) & prediction_values))
    fn = int(np.sum((label_values == 1) & ~prediction_values))
    tn = int(np.sum((label_values == 0) & ~prediction_values))

    def divide(numerator: int | float, denominator: int | float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": divide(fp, fp + tn),
    }


def constrained_threshold(
    labels: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    minimum_recall: float,
) -> tuple[float, dict[str, float | int | bool]]:
    """Minimize FPR, then maximize F1, subject to a recall floor."""

    label_values = np.asarray(labels, dtype=np.uint8)
    score_values = np.asarray(scores, dtype=np.float64)
    if label_values.ndim != 1 or score_values.shape != label_values.shape:
        raise ValueError("labels/scores must be aligned one-dimensional vectors")
    if not len(label_values) or not np.isfinite(score_values).all():
        raise ValueError("finite non-empty scores are required")
    if set(np.unique(label_values)) != {0, 1}:
        raise ValueError("calibration requires both normal and defect samples")
    if not 0.0 < minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be in (0, 1]")

    candidates = np.unique(score_values)
    candidates = np.concatenate(
        (candidates, [np.nextafter(float(candidates[-1]), math.inf)])
    )
    best: tuple[tuple[float, ...], float, dict[str, float | int | bool]] | None = None
    for raw_threshold in candidates:
        threshold = float(raw_threshold)
        metrics = binary_metrics(label_values, score_values >= threshold)
        constraint_met = bool(float(metrics["recall"]) + 1e-12 >= minimum_recall)
        if constraint_met:
            rank = (
                0.0,
                float(metrics["fpr"]),
                -float(metrics["f1"]),
                -float(metrics["recall"]),
                -threshold,
            )
        else:
            rank = (
                1.0,
                -float(metrics["recall"]),
                float(metrics["fpr"]),
                -float(metrics["f1"]),
                -threshold,
            )
        evidence: dict[str, float | int | bool] = {
            **metrics,
            "threshold": threshold,
            "minimum_recall": float(minimum_recall),
            "constraint_met": constraint_met,
        }
        if best is None or rank < best[0]:
            best = rank, threshold, evidence
    if best is None:  # pragma: no cover - guarded by non-empty validation above.
        raise RuntimeError("no threshold candidate was generated")
    return best[1], best[2]


def robust_score_scale(scores: np.ndarray | list[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite non-empty score vector required")
    center = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - center)))
    q25, q75 = np.quantile(values, (0.25, 0.75), method="linear")
    iqr_scale = float((q75 - q25) / 1.349)
    return max(1e-6, mad_scale, iqr_scale)


def fit_runtime_score_transform(
    calibration_scores: np.ndarray | list[float],
    runtime_scores: np.ndarray | list[float],
    labels: np.ndarray | list[int],
) -> dict[str, float | str]:
    """Align final-model runtime scores to the cross-fitted calibration space."""

    calibration_values = np.asarray(calibration_scores, dtype=np.float64)
    runtime_values = np.asarray(runtime_scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.uint8)
    if not (
        calibration_values.shape == runtime_values.shape == label_values.shape
        and calibration_values.ndim == 1
    ):
        raise ValueError("calibration/runtime scores and labels must align")
    if not np.isfinite(calibration_values).all() or not np.isfinite(runtime_values).all():
        raise ValueError("runtime score transform requires finite values")
    normal = label_values == 0
    if not np.any(normal):
        raise ValueError("runtime score transform requires normal samples")
    return {
        "method": "normal_robust_affine_to_cross_fitted_space",
        "input_center": float(np.median(runtime_values[normal])),
        "input_scale": robust_score_scale(runtime_values[normal]),
        "output_center": float(np.median(calibration_values[normal])),
        "output_scale": robust_score_scale(calibration_values[normal]),
    }


def identity_score_transform() -> dict[str, float | str]:
    return {
        "method": "identity",
        "input_center": 0.0,
        "input_scale": 1.0,
        "output_center": 0.0,
        "output_scale": 1.0,
    }


def transform_runtime_score(value: float, transform: dict[str, object]) -> float:
    input_scale = max(abs(float(transform["input_scale"])), 1e-12)
    return float(
        (float(value) - float(transform["input_center"]))
        / input_scale
        * float(transform["output_scale"])
        + float(transform["output_center"])
    )


def stratified_five_fold_ids(labels: np.ndarray | list[int]) -> np.ndarray:
    """Return deterministic 20-normal/6-defect validation folds for 100+30 support."""

    label_values = np.asarray(labels, dtype=np.uint8)
    if label_values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    counts = {value: int(np.sum(label_values == value)) for value in (0, 1)}
    if counts != {0: 100, 1: 30}:
        raise ValueError(
            f"five-fold AOI calibration requires exactly 100 normal + 30 defect; got {counts}"
        )
    folds = np.empty(len(label_values), dtype=np.int64)
    for value in (0, 1):
        indices = np.flatnonzero(label_values == value)
        folds[indices] = np.arange(len(indices), dtype=np.int64) % 5
    return folds


def _score_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _fit_branch(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    minimum_recall: float,
    runtime_transform: dict[str, object] | None = None,
) -> dict[str, object]:
    threshold, fit = constrained_threshold(
        labels, scores, minimum_recall=minimum_recall
    )
    normal_scores = scores[labels == 0]
    defect_scores = scores[labels == 1]
    return {
        "threshold": float(threshold),
        "scale": robust_score_scale(normal_scores),
        "normal_center": float(np.median(normal_scores)),
        "minimum_recall": float(minimum_recall),
        "fit_metrics": fit,
        "score_summary": {
            "normal": _score_summary(normal_scores),
            "defect": _score_summary(defect_scores),
        },
        "runtime_transform": runtime_transform or identity_score_transform(),
    }


def _calibration_margins(
    score_matrix: np.ndarray,
    branch_parameters: dict[str, dict[str, object]],
    *,
    clip: float,
) -> np.ndarray:
    columns = []
    for index, branch in enumerate(BRANCH_NAMES):
        parameters = branch_parameters[branch]
        columns.append(
            np.clip(
                (score_matrix[:, index] - float(parameters["threshold"]))
                / max(float(parameters["scale"]), 1e-12),
                -clip,
                clip,
            )
        )
    return np.column_stack(columns)


def _weight_grid(units: int) -> list[tuple[float, float, float]]:
    if units <= 0:
        raise ValueError("weight grid units must be positive")
    return [
        tuple(float(value / units) for value in candidate)
        for candidate in product(range(units + 1), repeat=3)
        if sum(candidate) == units
    ]


def _pure_branch(weights: tuple[float, float, float]) -> str | None:
    active = [
        branch
        for branch, weight in zip(BRANCH_NAMES, weights, strict=True)
        if abs(float(weight)) > 1e-12
    ]
    if len(active) != 1:
        return None
    branch = active[0]
    index = BRANCH_NAMES.index(branch)
    return branch if abs(float(weights[index]) - 1.0) <= 1e-12 else None


def _fusion_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    weights: tuple[float, float, float],
    minimum_recall: float,
) -> tuple[float, dict[str, object]]:
    """Fit a mixed score, or retain a calibrated branch's zero-margin gate."""

    pure_branch = _pure_branch(weights)
    if pure_branch is None:
        threshold, fit = constrained_threshold(
            labels, scores, minimum_recall=minimum_recall
        )
        return threshold, {
            **fit,
            "threshold_protocol": "support_constrained_mixed_fusion",
        }

    threshold = 0.0
    metrics = binary_metrics(labels, scores >= threshold)
    return threshold, {
        **metrics,
        "threshold": threshold,
        "minimum_recall": float(minimum_recall),
        "constraint_met": bool(
            float(metrics["recall"]) + 1e-12 >= minimum_recall
        ),
        "threshold_protocol": "calibrated_branch_zero_margin",
        "pure_branch": pure_branch,
    }


def fit_five_fold_continuous_fusion(
    *,
    labels: np.ndarray | list[int],
    sample_ids: Sequence[str],
    branch_scores: dict[str, np.ndarray | list[float]],
    runtime_transforms: dict[str, dict[str, object]] | None = None,
    refit_branch_scores: dict[str, np.ndarray | list[float]] | None = None,
    acceptance_minimum_recall: float = 0.95,
    branch_minimum_recall: float = 0.95,
    fold_fit_minimum_recall: float = 0.98,
    final_fit_minimum_recall: float = 0.95,
    margin_clip: float = 6.0,
    weight_grid_units: int = 8,
) -> dict[str, object]:
    """Fit per-branch thresholds and search continuous fusion using support-only CV."""

    label_values = np.asarray(labels, dtype=np.uint8)
    folds = stratified_five_fold_ids(label_values)
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(label_values) or len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be aligned and unique")
    if set(branch_scores) != set(BRANCH_NAMES):
        raise ValueError(f"branch_scores must contain exactly {BRANCH_NAMES}")
    columns = [np.asarray(branch_scores[name], dtype=np.float64) for name in BRANCH_NAMES]
    if any(values.shape != label_values.shape for values in columns):
        raise ValueError("every branch score vector must align with labels")
    score_matrix = np.column_stack(columns)
    if not np.isfinite(score_matrix).all():
        raise ValueError("branch scores must be finite")
    if not (
        0.0
        < acceptance_minimum_recall
        <= fold_fit_minimum_recall
        <= 1.0
        and acceptance_minimum_recall <= final_fit_minimum_recall <= 1.0
    ):
        raise ValueError(
            "recall targets must satisfy 0 < acceptance <= fold/final fit <= 1"
        )
    if margin_clip <= 0.0:
        raise ValueError("margin_clip must be positive")

    fold_contexts: list[dict[str, object]] = []
    for fold in range(5):
        validation = folds == fold
        training = ~validation
        parameters = {
            branch: _fit_branch(
                label_values[training],
                score_matrix[training, index],
                minimum_recall=branch_minimum_recall,
            )
            for index, branch in enumerate(BRANCH_NAMES)
        }
        fold_contexts.append(
            {
                "fold": fold,
                "training": training,
                "validation": validation,
                "branch_parameters": parameters,
                "training_margins": _calibration_margins(
                    score_matrix[training], parameters, clip=margin_clip
                ),
                "validation_margins": _calibration_margins(
                    score_matrix[validation], parameters, clip=margin_clip
                ),
            }
        )

    candidate_records: list[dict[str, object]] = []
    selected_rank: tuple[float, ...] | None = None
    selected_weights: tuple[float, float, float] | None = None
    for candidate_index, weights in enumerate(_weight_grid(weight_grid_units)):
        weight_values = np.asarray(weights, dtype=np.float64)
        oof_predictions = np.zeros(len(label_values), dtype=bool)
        for context in fold_contexts:
            training = np.asarray(context["training"], dtype=bool)
            validation = np.asarray(context["validation"], dtype=bool)
            train_scores = np.asarray(context["training_margins"]) @ weight_values
            validation_scores = np.asarray(context["validation_margins"]) @ weight_values
            threshold, _ = _fusion_threshold(
                label_values[training],
                train_scores,
                weights=weights,
                minimum_recall=fold_fit_minimum_recall,
            )
            oof_predictions[validation] = validation_scores >= threshold
        metrics = binary_metrics(label_values, oof_predictions)
        constraint_met = bool(
            float(metrics["recall"]) + 1e-12 >= acceptance_minimum_recall
        )
        if constraint_met:
            active_branches = int(np.count_nonzero(weight_values > 1e-12))
            rank = (
                0.0,
                float(metrics["fpr"]),
                -float(metrics["f1"]),
                -float(metrics["recall"]),
                active_branches,
                -float(weights[1]),
                float(np.sum(weight_values**2)),
                *weights,
            )
        else:
            rank = (
                1.0,
                -float(metrics["recall"]),
                float(metrics["fpr"]),
                -float(metrics["f1"]),
                int(np.count_nonzero(weight_values > 1e-12)),
                -float(weights[1]),
                float(np.sum(weight_values**2)),
                *weights,
            )
        candidate_records.append(
            {
                "candidate_id": f"linear-{candidate_index:03d}",
                "aggregation": "nonnegative_linear",
                "weights": dict(zip(BRANCH_NAMES, weights, strict=True)),
                "oof_metrics": metrics,
                "recall_constraint_met": constraint_met,
                "threshold_protocol": (
                    "calibrated_branch_zero_margin"
                    if _pure_branch(weights) is not None
                    else "support_constrained_mixed_fusion"
                ),
                "selected": False,
            }
        )
        if selected_rank is None or rank < selected_rank:
            selected_rank = rank
            selected_weights = weights
    if selected_weights is None:  # pragma: no cover - the weight grid is non-empty.
        raise RuntimeError("fusion search produced no candidate")

    selected_weight_values = np.asarray(selected_weights, dtype=np.float64)
    for record in candidate_records:
        record["selected"] = bool(
            tuple(float(record["weights"][name]) for name in BRANCH_NAMES)
            == selected_weights
        )

    selected_oof_scores = np.zeros(len(label_values), dtype=np.float64)
    selected_oof_thresholds = np.zeros(len(label_values), dtype=np.float64)
    selected_oof_predictions = np.zeros(len(label_values), dtype=bool)
    selected_oof_margins = np.zeros_like(score_matrix)
    fold_evidence: list[dict[str, object]] = []
    for context in fold_contexts:
        fold = int(context["fold"])
        training = np.asarray(context["training"], dtype=bool)
        validation = np.asarray(context["validation"], dtype=bool)
        training_scores = np.asarray(context["training_margins"]) @ selected_weight_values
        validation_scores = np.asarray(context["validation_margins"]) @ selected_weight_values
        threshold, training_fit = _fusion_threshold(
            label_values[training],
            training_scores,
            weights=selected_weights,
            minimum_recall=fold_fit_minimum_recall,
        )
        validation_predictions = validation_scores >= threshold
        selected_oof_scores[validation] = validation_scores
        selected_oof_thresholds[validation] = threshold
        selected_oof_predictions[validation] = validation_predictions
        selected_oof_margins[validation] = np.asarray(context["validation_margins"])
        branch_parameters = context["branch_parameters"]
        fold_evidence.append(
            {
                "fold": fold,
                "training_count": int(np.sum(training)),
                "validation_count": int(np.sum(validation)),
                "validation_normal": int(np.sum(validation & (label_values == 0))),
                "validation_defect": int(np.sum(validation & (label_values == 1))),
                "validation_sample_ids": [
                    ids[index] for index in np.flatnonzero(validation)
                ],
                "branch_parameters": {
                    branch: {
                        "threshold": float(branch_parameters[branch]["threshold"]),
                        "scale": float(branch_parameters[branch]["scale"]),
                    }
                    for branch in BRANCH_NAMES
                },
                "fusion_threshold": float(threshold),
                "training_fit": training_fit,
                "validation_metrics": binary_metrics(
                    label_values[validation], validation_predictions
                ),
            }
        )

    oof_metrics = binary_metrics(label_values, selected_oof_predictions)
    transforms = runtime_transforms or {}
    final_branches = {
        branch: _fit_branch(
            label_values,
            score_matrix[:, index],
            minimum_recall=branch_minimum_recall,
            runtime_transform=transforms.get(branch),
        )
        for index, branch in enumerate(BRANCH_NAMES)
    }
    if refit_branch_scores is None:
        refit_score_matrix = score_matrix
        refit_score_space = "cross_fitted_calibration_scores"
    else:
        if set(refit_branch_scores) != set(BRANCH_NAMES):
            raise ValueError(f"refit_branch_scores must contain exactly {BRANCH_NAMES}")
        refit_columns = [
            np.asarray(refit_branch_scores[name], dtype=np.float64)
            for name in BRANCH_NAMES
        ]
        if any(values.shape != label_values.shape for values in refit_columns):
            raise ValueError("every refit branch score vector must align with labels")
        refit_score_matrix = np.column_stack(refit_columns)
        if not np.isfinite(refit_score_matrix).all():
            raise ValueError("refit branch scores must be finite")
        refit_score_space = "final_runtime_aligned_support_scores"
    final_margins = _calibration_margins(
        refit_score_matrix, final_branches, clip=margin_clip
    )
    final_scores = final_margins @ selected_weight_values
    final_threshold, final_fit = _fusion_threshold(
        label_values,
        final_scores,
        weights=selected_weights,
        minimum_recall=final_fit_minimum_recall,
    )

    oof_rows = []
    for index, sample_id in enumerate(ids):
        oof_rows.append(
            {
                "sample_id": sample_id,
                "label": int(label_values[index]),
                "fold": int(folds[index]),
                "branch_scores": {
                    branch: float(score_matrix[index, branch_index])
                    for branch_index, branch in enumerate(BRANCH_NAMES)
                },
                "branch_margins": {
                    branch: float(selected_oof_margins[index, branch_index])
                    for branch_index, branch in enumerate(BRANCH_NAMES)
                },
                "fusion_score": float(selected_oof_scores[index]),
                "fusion_threshold": float(selected_oof_thresholds[index]),
                "prediction": int(selected_oof_predictions[index]),
            }
        )

    refit_rows = []
    for index, sample_id in enumerate(ids):
        refit_rows.append(
            {
                "sample_id": sample_id,
                "label": int(label_values[index]),
                "branch_scores": {
                    branch: float(refit_score_matrix[index, branch_index])
                    for branch_index, branch in enumerate(BRANCH_NAMES)
                },
                "branch_margins": {
                    branch: float(final_margins[index, branch_index])
                    for branch_index, branch in enumerate(BRANCH_NAMES)
                },
                "fusion_score": float(final_scores[index]),
                "fusion_threshold": float(final_threshold),
                "prediction": int(final_scores[index] >= final_threshold),
            }
        )

    candidate_records.sort(
        key=lambda record: (
            not bool(record["recall_constraint_met"]),
            float(record["oof_metrics"]["fpr"])
            if bool(record["recall_constraint_met"])
            else -float(record["oof_metrics"]["recall"]),
            -float(record["oof_metrics"]["f1"]),
            record["candidate_id"],
        )
    )
    return {
        "schema": CONTINUOUS_FUSION_SCHEMA,
        "method": "support_only_stratified_5fold_sparse_safety_search",
        "selection_scope": "100_normal_plus_30_defect_support_only",
        "independent_evaluation_used_for_selection": False,
        "fold_count": 5,
        "fold_assignment": "normal_order_mod_5_and_defect_order_mod_5",
        "acceptance_minimum_recall": float(acceptance_minimum_recall),
        "branch_minimum_recall": float(branch_minimum_recall),
        "fold_fit_minimum_recall": float(fold_fit_minimum_recall),
        "final_fit_minimum_recall": float(final_fit_minimum_recall),
        "branches": final_branches,
        "fusion": {
            "aggregation": "nonnegative_linear",
            "weights": dict(zip(BRANCH_NAMES, selected_weights, strict=True)),
            "threshold": float(final_threshold),
            "threshold_protocol": str(final_fit["threshold_protocol"]),
            "margin_clip": float(margin_clip),
            "weight_grid_units": int(weight_grid_units),
            "candidate_count": len(candidate_records),
            "oof_metrics": oof_metrics,
            "oof_recall_constraint_met": bool(
                float(oof_metrics["recall"]) + 1e-12 >= acceptance_minimum_recall
            ),
            "support_refit_score_space": refit_score_space,
            "support_refit_metrics_apparent": final_fit,
            "tie_break": (
                "fpr_then_f1_then_recall_then_sparsity_then_supervised_local_weight"
            ),
        },
        "folds": fold_evidence,
        "oof_predictions": oof_rows,
        "support_refit_predictions": refit_rows,
        "candidate_ledger": candidate_records,
    }


def apply_continuous_fusion(
    branch_scores: dict[str, float], calibration: dict[str, object]
) -> dict[str, object]:
    if calibration.get("schema") != CONTINUOUS_FUSION_SCHEMA:
        raise ValueError("continuous fusion calibration schema is missing or unsupported")
    if set(branch_scores) != set(BRANCH_NAMES):
        raise ValueError(f"branch scores must contain exactly {BRANCH_NAMES}")
    branches = calibration["branches"]
    fusion = calibration["fusion"]
    clip = float(fusion["margin_clip"])
    calibrated_scores: dict[str, float] = {}
    margins: dict[str, float] = {}
    for branch in BRANCH_NAMES:
        parameters = branches[branch]
        calibrated = transform_runtime_score(
            float(branch_scores[branch]), parameters["runtime_transform"]
        )
        calibrated_scores[branch] = calibrated
        margins[branch] = float(
            np.clip(
                (calibrated - float(parameters["threshold"]))
                / max(float(parameters["scale"]), 1e-12),
                -clip,
                clip,
            )
        )
    weights = fusion["weights"]
    score = float(sum(float(weights[name]) * margins[name] for name in BRANCH_NAMES))
    threshold = float(fusion["threshold"])
    return {
        "score": score,
        "threshold": threshold,
        "decision": bool(score >= threshold),
        "calibrated_branch_scores": calibrated_scores,
        "branch_margins": margins,
        "weights": {name: float(weights[name]) for name in BRANCH_NAMES},
    }


def select_global_category_thresholds(
    category_scores: dict[str, tuple[np.ndarray | list[int], np.ndarray | list[float]]],
    *,
    pooled_minimum_recall: float = 0.95,
    category_minimum_recall: float = 0.80,
) -> dict[str, object]:
    """Allocate the pooled recall budget across category-specific thresholds."""

    if not category_scores:
        raise ValueError("at least one category score vector is required")
    if not 0.0 < category_minimum_recall <= pooled_minimum_recall <= 1.0:
        raise ValueError("recall floors must satisfy 0 < category <= pooled <= 1")

    options: dict[str, dict[int, dict[str, object]]] = {}
    total_positive = 0
    total_negative = 0
    for category, (raw_labels, raw_scores) in sorted(category_scores.items()):
        labels = np.asarray(raw_labels, dtype=np.uint8)
        scores = np.asarray(raw_scores, dtype=np.float64)
        if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
            raise ValueError(f"{category}: labels/scores must align")
        if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
            raise ValueError(f"{category}: finite binary calibration data required")
        total_positive += int(np.sum(labels == 1))
        total_negative += int(np.sum(labels == 0))
        by_tp: dict[int, dict[str, object]] = {}
        for raw_threshold in np.unique(scores):
            threshold = float(raw_threshold)
            metrics = binary_metrics(labels, scores >= threshold)
            if float(metrics["recall"]) + 1e-12 < category_minimum_recall:
                continue
            tp = int(metrics["tp"])
            candidate = {
                "threshold": threshold,
                "metrics": metrics,
            }
            existing = by_tp.get(tp)
            rank = (
                int(metrics["fp"]),
                -float(metrics["f1"]),
                -threshold,
            )
            if existing is None:
                by_tp[tp] = candidate
            else:
                old = existing["metrics"]
                old_rank = (
                    int(old["fp"]),
                    -float(old["f1"]),
                    -float(existing["threshold"]),
                )
                if rank < old_rank:
                    by_tp[tp] = candidate
        if not by_tp:
            raise RuntimeError(f"{category}: no threshold satisfies category recall floor")
        options[category] = by_tp

    # For each exact pooled TP count, keep the path with the fewest false positives.
    states: dict[int, tuple[int, dict[str, dict[str, object]]]] = {0: (0, {})}
    for category in sorted(options):
        next_states: dict[int, tuple[int, dict[str, dict[str, object]]]] = {}
        for accumulated_tp, (accumulated_fp, selected) in states.items():
            for category_tp, candidate in options[category].items():
                pooled_tp = accumulated_tp + category_tp
                pooled_fp = accumulated_fp + int(candidate["metrics"]["fp"])
                replacement = (pooled_fp, {**selected, category: candidate})
                existing = next_states.get(pooled_tp)
                if existing is None or pooled_fp < existing[0]:
                    next_states[pooled_tp] = replacement
        states = next_states

    required_tp = int(math.ceil(pooled_minimum_recall * total_positive - 1e-12))
    feasible = []
    for tp, (fp, selected) in states.items():
        if tp < required_tp:
            continue
        fn = total_positive - tp
        tn = total_negative - fp
        pooled = binary_metrics(
            np.asarray([1] * total_positive + [0] * total_negative, dtype=np.uint8),
            np.asarray([1] * tp + [0] * fn + [1] * fp + [0] * tn, dtype=bool),
        )
        rank = (
            float(pooled["fpr"]),
            -float(pooled["f1"]),
            -float(pooled["recall"]),
            tuple(float(selected[name]["threshold"]) for name in sorted(selected)),
        )
        feasible.append((rank, selected, pooled))
    if not feasible:
        raise RuntimeError("no global threshold allocation satisfies pooled recall floor")
    _, selected, pooled = min(feasible, key=lambda item: item[0])
    return {
        "method": "support_only_dynamic_program_category_threshold_allocation",
        "pooled_minimum_recall": float(pooled_minimum_recall),
        "category_minimum_recall": float(category_minimum_recall),
        "selection_scope": "support_only",
        "independent_evaluation_used_for_selection": False,
        "pooled_support_metrics": pooled,
        "categories": selected,
        "candidate_counts": {
            category: len(category_options)
            for category, category_options in options.items()
        },
    }


def scale_padding(height: int, width: int, retained_scale: float) -> tuple[int, int, int, int]:
    inner_h = int(round(height * retained_scale))
    inner_w = int(round(width * retained_scale))
    top = (height - inner_h) // 2
    bottom = height - inner_h - top
    left = (width - inner_w) // 2
    right = width - inner_w - left
    return top, bottom, left, right


def reflection_error(image: np.ndarray, retained_scale: float) -> tuple[float, dict[str, float]]:
    height, width = image.shape[:2]
    top, bottom, left, right = scale_padding(height, width, retained_scale)
    right_boundary = width - right - 1
    bottom_boundary = height - bottom - 1
    pairs = {
        "left": (image[:, :left], image[:, left + 1 : left + 1 + left][:, ::-1]),
        "right": (
            image[:, width - right :],
            image[:, right_boundary - right : right_boundary][:, ::-1],
        ),
        "top": (image[:top, :], image[top + 1 : top + 1 + top, :][::-1]),
        "bottom": (
            image[height - bottom :, :],
            image[bottom_boundary - bottom : bottom_boundary, :][::-1],
        ),
    }
    errors = {
        name: float(cv2.norm(outer, inner, cv2.NORM_L1) / (outer.size * 255.0))
        for name, (outer, inner) in pairs.items()
    }
    return max(errors.values()), errors


def calibrate_scale_detector(
    normal_images: Sequence[np.ndarray],
) -> tuple[dict[float, dict[str, float]], list[dict[str, float]]]:
    if len(normal_images) != 100:
        raise ValueError("scale calibration requires exactly 100 adaptation-normal images")
    calibration: dict[float, dict[str, float]] = {}
    signatures = []
    for scale in (0.80, 0.85, 0.90, 0.95):
        errors = []
        for image_index, image in enumerate(normal_images):
            error, sides = reflection_error(image, scale)
            errors.append(error)
            signatures.append(
                {
                    "normal_index": float(image_index),
                    "retained_scale": scale,
                    "reflection_error": error,
                    **{f"{name}_error": value for name, value in sides.items()},
                }
            )
        values = np.asarray(errors, dtype=np.float64)
        minimum = float(np.min(values))
        quantile = float(np.quantile(values, 0.01, method="linear"))
        threshold = min(float(np.nextafter(minimum, -math.inf)), quantile * 0.25, 0.002)
        if threshold <= 0.0:
            raise RuntimeError(f"normal reference is not separable for scale {scale}")
        calibration[scale] = {
            "normal_min": minimum,
            "normal_quantile": quantile,
            "trigger_threshold": threshold,
            "reference_trigger_count": float(np.sum(values <= threshold)),
        }
    return calibration, signatures


def fusion_score(
    *,
    primary_score: float,
    primary_threshold: float,
    g2_score: float,
    g2_threshold: float,
    dino_score: float,
    dino_threshold: float,
    known_score: float,
    known_threshold: float,
) -> dict[str, float]:
    primary_margin = primary_score / max(abs(primary_threshold), 1e-6)
    dino_ratio = dino_score / max(abs(dino_threshold), 1e-6)
    primary_signal = max(
        primary_margin - 1.01,
        min(primary_margin - 1.0, dino_ratio - 0.90),
    )
    guard_signal = min(
        primary_margin - 0.95,
        (g2_score - g2_threshold) / max(abs(g2_threshold), 1.0),
        dino_ratio - 0.70,
    )
    known_raw_signal = (known_score - known_threshold) / max(abs(known_threshold), 1e-6)
    known_signal = min(primary_margin - 0.95, known_raw_signal - 0.46)
    general_score = max(primary_signal, guard_signal)
    return {
        # ``score`` retains the old D10 value for legacy bundles and audit parity.
        "score": max(general_score, known_signal),
        # V2 keeps general and known as independent continuous branches.
        "general_score": general_score,
        "known_score": known_score,
        "known_reference_threshold": known_threshold,
        "primary_margin": primary_margin,
        "primary_signal": primary_signal,
        "g2_score": g2_score,
        "g2_threshold": g2_threshold,
        "guard_signal": guard_signal,
        "dino_ratio": dino_ratio,
        "known_raw_signal": known_raw_signal,
        "known_signal": known_signal,
    }
