from __future__ import annotations

import cv2
import numpy as np
import torch

from aoi.competition.deployment import TorchMapBackend, anomaly_score
from .adaptation import nine_tiles
from .calibration import RobustNormalizer, apply_continuous_fusion, fusion_score


def _backend_scores(backend, images: list[np.ndarray]) -> np.ndarray:
    maps = backend.predict_maps(images)
    return np.asarray([anomaly_score(value) for value in maps], dtype=np.float64)


def score_images(*, images, ids, source, adapted, primary, dino, head, calibration, device):
    small = [cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA) for image in images]
    source_raw = _backend_scores(TorchMapBackend(source, device=device), small)
    adapted_raw = _backend_scores(TorchMapBackend(adapted, device=device), small)
    source_norm = RobustNormalizer(**calibration["source_calibration"]["normalizer"])
    adapted_norm = RobustNormalizer(**calibration["adapted_calibration"]["normalizer"])
    g2_scores = np.maximum(source_norm.transform(source_raw), adapted_norm.transform(adapted_raw))
    primary_predictions = primary.predict_batch(images, sample_ids=ids)
    dino_predictions = dino.predict_batch(images, sample_ids=ids)
    rows = []
    for image, sample_id, p, d, g2 in zip(images, ids, primary_predictions, dino_predictions, g2_scores, strict=True):
        detail = fusion_score(
            primary_score=float(p.score),
            primary_threshold=float(p.threshold),
            g2_score=float(g2),
            g2_threshold=float(calibration["g2_image_threshold"]),
            dino_score=float(d.score),
            dino_threshold=float(d.threshold),
            known_score=float(d.metadata["known_score"]),
            known_threshold=float(d.metadata["known_threshold"]),
        )
        with torch.inference_mode():
            features = dino._features(nine_tiles(image)).reshape(1, 9, -1, head.feature_mean.shape[0])
            local_score_raw = float(head.score_tiled_features(features)[0])
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
            # Historical bundles remain runnable and reproduce their original routing.
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
            fusion_score_value = float(max(int(known_decision), int(general_decision), int(local_decision)))
            fusion_threshold = 1.0
            fusion_weights = {"general": 0.0, "local": 0.0, "known": 0.0}
            branch_margins = {"general": 0.0, "local": 0.0, "known": 0.0}
        rows.append({
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
            "known_decision": known_decision,
            "general_decision": general_decision,
            "local_decision": local_decision,
            "primary_score": float(p.score),
            "dino_score": float(d.score),
            "g2_score": float(g2),
        })
    return rows
