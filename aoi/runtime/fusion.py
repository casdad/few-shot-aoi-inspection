from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from aoi.competition.deployment import TorchMapBackend, anomaly_score


BRANCH_NAMES = ("general", "local", "known")
CONTINUOUS_FUSION_SCHEMA = "continuous_score_fusion_v2"


@dataclass(frozen=True)
class RobustNormalizer:
    center: float
    scale: float
    method: str = "normal_median_mad"

    def transform(self, scores):
        return (np.asarray(scores, dtype=np.float64) - self.center) / self.scale


def d10_locked_score(
    *,
    primary_score,
    primary_threshold,
    g2_score,
    g2_threshold,
    dino_score,
    dino_threshold,
    known_score,
    known_threshold,
):
    primary_margin = primary_score / max(abs(primary_threshold), 1e-6)
    dino_ratio = dino_score / max(abs(dino_threshold), 1e-6)
    primary_signal = max(primary_margin - 1.01, min(primary_margin - 1.0, dino_ratio - 0.90))
    guard_signal = min(primary_margin - 0.95, (g2_score - g2_threshold) / max(abs(g2_threshold), 1.0), dino_ratio - 0.70)
    known_raw_signal = (known_score - known_threshold) / max(abs(known_threshold), 1e-6)
    known_signal = min(primary_margin - 0.95, known_raw_signal - 0.46)
    general_score = max(primary_signal, guard_signal)
    return {
        "score": max(general_score, known_signal),
        "general_score": general_score,
        "known_score": known_score,
        "known_reference_threshold": known_threshold,
        "primary_margin": primary_margin,
        "known_signal": known_signal,
        "dino_ratio": dino_ratio,
        "g2_score": g2_score,
    }


def _transform_runtime_score(value: float, transform: dict[str, object]) -> float:
    input_scale = max(abs(float(transform["input_scale"])), 1e-12)
    return float(
        (float(value) - float(transform["input_center"]))
        / input_scale
        * float(transform["output_scale"])
        + float(transform["output_center"])
    )


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
        calibrated = _transform_runtime_score(
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


def grid_boxes(image: np.ndarray, grid: int = 3):
    height, width = image.shape[:2]
    side = max((height + grid - 1) // grid, (width + grid - 1) // grid)
    side = min(side, height, width)
    lefts = np.linspace(0, width - side, grid).round().astype(int)
    tops = np.linspace(0, height - side, grid).round().astype(int)
    return [
        (int(left), int(top), int(left + side), int(top + side))
        for top in tops
        for left in lefts
    ]


def nine_tiles(image: np.ndarray):
    return [
        image[top:bottom, left:right].copy()
        for left, top, right, bottom in grid_boxes(image)
    ]


def _backend_scores(backend, images):
    maps = backend.predict_maps(images)
    return np.asarray([anomaly_score(value) for value in maps], dtype=np.float64)


def _display_anomaly_map(primary_map: np.ndarray, dino_map: np.ndarray) -> np.ndarray:
    target = primary_map.shape[:2]
    dino_resized = cv2.resize(np.asarray(dino_map, dtype=np.float32), (target[1], target[0]))

    def normalize(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        low, high = np.percentile(value, (5, 99))
        return np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)

    return np.maximum(normalize(primary_map), normalize(dino_resized)).astype(np.float32)


def score_images(
    *, images, ids, source, adapted, primary, dino, head, calibration, device,
    source_backend=None, adapted_backend=None, g2_backend=None,
    include_anomaly_map=False,
):
    small = [cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA) for image in images]
    if g2_backend is None:
        source_backend = source_backend or TorchMapBackend(source, device=device)
        adapted_backend = adapted_backend or TorchMapBackend(adapted, device=device)
        source_raw = _backend_scores(source_backend, small)
        adapted_raw = _backend_scores(adapted_backend, small)
    else:
        source_maps, adapted_maps = g2_backend.predict_maps(small)
        source_raw = np.asarray(
            [anomaly_score(value) for value in source_maps], dtype=np.float64
        )
        adapted_raw = np.asarray(
            [anomaly_score(value) for value in adapted_maps], dtype=np.float64
        )
    source_norm = RobustNormalizer(**calibration["source_calibration"]["normalizer"])
    adapted_norm = RobustNormalizer(**calibration["adapted_calibration"]["normalizer"])
    g2_scores = np.maximum(source_norm.transform(source_raw), adapted_norm.transform(adapted_raw))
    primary_predictions = primary.predict_batch(images, sample_ids=ids)
    dino_inputs = []
    for image in images:
        dino_inputs.append(image)
        dino_inputs.extend(nine_tiles(image))
    dino_features = dino._features(dino_inputs)
    full_features = dino_features[::10]
    tile_features = torch.stack(
        [dino_features[index * 10 + 1 : index * 10 + 10] for index in range(len(images))]
    )
    dino_predictions = dino.predict_features(full_features, sample_ids=ids)
    with torch.inference_mode():
        local_scores_raw = head.score_tiled_features(tile_features)
    rows = []
    for image, sample_id, p, d, g2, local_score_raw in zip(
        images,
        ids,
        primary_predictions,
        dino_predictions,
        g2_scores,
        local_scores_raw,
        strict=True,
    ):
        detail = d10_locked_score(
            primary_score=float(p.score), primary_threshold=float(p.threshold),
            g2_score=float(g2), g2_threshold=float(calibration["g2_image_threshold"]),
            dino_score=float(d.score), dino_threshold=float(d.threshold),
            known_score=float(d.metadata["known_score"]), known_threshold=float(d.metadata["known_threshold"]),
        )
        local_score_raw = float(local_score_raw)
        if "continuous_fusion" in calibration:
            fusion = apply_continuous_fusion(
                {
                    "general": float(detail["general_score"]),
                    "local": local_score_raw,
                    "known": float(detail["known_score"]),
                },
                calibration["continuous_fusion"],
            )
            branch_parameters = calibration["continuous_fusion"]["branches"]
            calibrated_scores = fusion["calibrated_branch_scores"]
            general_threshold = float(branch_parameters["general"]["threshold"])
            local_threshold = float(branch_parameters["local"]["threshold"])
            known_threshold = float(branch_parameters["known"]["threshold"])
            general_decision = bool(calibrated_scores["general"] >= general_threshold)
            local_decision = bool(calibrated_scores["local"] >= local_threshold)
            known_decision = bool(calibrated_scores["known"] >= known_threshold)
            decision = bool(fusion["decision"])
            fusion_mode = "continuous_score_fusion_v2"
            fusion_score_value = float(fusion["score"])
            fusion_threshold = float(fusion["threshold"])
            fusion_weights = fusion["weights"]
            branch_margins = fusion["branch_margins"]
            general_score = float(calibrated_scores["general"])
            local_score = float(calibrated_scores["local"])
            known_score = float(calibrated_scores["known"])
        else:
            general_score = float(detail["score"])
            local_score = local_score_raw
            known_score = float(detail["known_signal"])
            general_threshold = float(calibration["general"]["general_threshold"])
            local_threshold = float(calibration["head"]["local_threshold"])
            known_threshold = 0.0
            known_decision = bool(known_score >= known_threshold)
            general_decision = bool(general_score >= general_threshold)
            local_decision = bool(local_score >= local_threshold)
            decision = bool(known_decision or general_decision or local_decision)
            fusion_mode = "legacy_binary_or"
            fusion_score_value = float(
                max(int(known_decision), int(general_decision), int(local_decision))
            )
            fusion_threshold = 1.0
            fusion_weights = {"general": 0.0, "local": 0.0, "known": 0.0}
            branch_margins = {"general": 0.0, "local": 0.0, "known": 0.0}
        row = {
            "sample_id": sample_id,
            "decision": decision,
            "fusion_mode": fusion_mode,
            "fusion_score": fusion_score_value,
            "fusion_threshold": fusion_threshold,
            "fusion_weights": fusion_weights,
            "branch_margins": branch_margins,
            "general_score_raw": float(detail["general_score"]),
            "general_score": general_score,
            "general_threshold": general_threshold,
            "local_score_raw": local_score_raw,
            "local_score": local_score,
            "local_threshold": local_threshold,
            "known_score_raw": float(detail["known_score"]),
            "known_score": known_score,
            "known_threshold": known_threshold,
            "known_reference_threshold": float(detail["known_reference_threshold"]),
            "known_signal": float(detail["known_signal"]),
            "known_decision": bool(known_decision),
            "general_decision": bool(general_decision),
            "local_decision": bool(local_decision),
            "primary_score": float(p.score),
            "dino_score": float(d.score),
            "g2_score": float(g2),
        }
        if include_anomaly_map:
            row["_anomaly_map"] = _display_anomaly_map(p.anomaly_map, d.anomaly_map)
        rows.append(row)
    return rows
