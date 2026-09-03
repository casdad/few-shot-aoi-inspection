"""Clean 100+30 few-shot adaptation using the original AOI scheme."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from aoi.competition.baselines import DeepPatchCoreLite
from aoi.competition.deployment import TorchMapBackend, anomaly_score
from aoi.competition.evaluation import best_f1_threshold
from aoi.competition.models.anomaly_detectors import (
    DINOv2SpatialPrototype,
    MultiScalePatchMemory,
)
from aoi.competition.models.known_defect_heads import DinoTiledMLPKnownHead

from .calibration import (
    RobustNormalizer,
    calibrate_image_scores,
    calibrate_scale_detector,
    fit_five_fold_continuous_fusion,
    fit_runtime_score_transform,
    fusion_score,
    normal_only_threshold,
    pixel_normalizer,
    stratified_pixel_threshold,
    target_point,
    transform_runtime_score,
)
from .config import TrainingConfig
from .dataset import TrainingDataset, dataset_arrays
from .manifests import payload_hash, sha256_file, write_yaml


@dataclass
class AdaptationArtifacts:
    primary: MultiScalePatchMemory
    adapted: DeepPatchCoreLite
    dino: DINOv2SpatialPrototype
    head: DinoTiledMLPKnownHead
    source: DeepPatchCoreLite
    calibration: dict[str, Any]
    model_paths: dict[str, Path]
    adaptation_seconds: dict[str, float]
    input_samples: list[np.ndarray]
    input_ids: list[str]


def _save_candidate(candidate: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        candidate.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _save_patchcore(model: DeepPatchCoreLite, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def grid_boxes(image: np.ndarray, grid: int = 3) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    side = max((height + grid - 1) // grid, (width + grid - 1) // grid)
    side = min(side, height, width)
    lefts = np.linspace(0, width - side, grid).round().astype(int)
    tops = np.linspace(0, height - side, grid).round().astype(int)
    return [
        (int(left), int(top), int(left + side), int(top + side)) for top in tops for left in lefts
    ]


def nine_tiles(image: np.ndarray) -> list[np.ndarray]:
    return [image[top:bottom, left:right].copy() for left, top, right, bottom in grid_boxes(image)]


def nine_tile_defect(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = grid_boxes(image)
    mask_tiles = [mask[top:bottom, left:right] for left, top, right, bottom in boxes]
    index = int(np.argmax([int(value.sum()) for value in mask_tiles]))
    left, top, right, bottom = boxes[index]
    resized = Image.fromarray(mask_tiles[index] * 255).resize((252, 252), Image.Resampling.NEAREST)
    return image[top:bottom, left:right].copy(), (np.asarray(resized) > 0).astype(np.uint8)


def _resize_images(images: list[np.ndarray], size: int = 128) -> list[np.ndarray]:
    return [cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA) for image in images]


def _resize_masks(masks: list[np.ndarray], size: int = 128) -> list[np.ndarray]:
    return [
        cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        for mask in masks
    ]


def _backend_maps(
    backend: Any, images: list[np.ndarray], *, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    maps = []
    for start in range(0, len(images), batch_size):
        maps.append(backend.predict_maps(images[start : start + batch_size]))
    values = np.concatenate(maps).astype(np.float32)
    raw = np.asarray([anomaly_score(value) for value in values], dtype=np.float64)
    return values, raw


def _calibrate_pixel_threshold(
    masks: np.ndarray | None,
    maps: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[float, dict[str, object]]:
    if masks is not None:
        return stratified_pixel_threshold(masks, maps, seed=seed)
    normal_values = np.asarray(maps)[np.asarray(labels) == 0].reshape(-1)
    if not len(normal_values):
        raise ValueError("normal images are required for weak pixel calibration")
    threshold = float(np.quantile(normal_values, 0.999))
    return threshold, {
        "available": False,
        "method": "normal_pixels_quantile",
        "quantile": 0.999,
        "normal_pixels": int(len(normal_values)),
        "note": "No ground-truth pixel masks were supplied.",
    }


def _fit_primary(
    config: TrainingConfig,
    normals: list[np.ndarray],
    defects: list[np.ndarray],
    masks: list[np.ndarray] | None,
    output: Path,
) -> tuple[MultiScalePatchMemory, dict[str, Any], float, Path]:
    candidate = MultiScalePatchMemory(
        f"{config.candidate_id}-PRIMARY",
        weight_path=config.source_assets["resnet18_weights"].path,
        input_size=256,
        projection_dimension=96,
        maximum_patch_pool=8192,
        coreset_size=512,
        seed=20261511,
        device=config.device,
    )
    candidate.fit_base([])
    result = candidate.adapt(normals, defects, defect_labels=masks)
    points = candidate.calibration["operating_point"]["points"]
    target = [point for point in points if abs(float(point["target_fpr"]) - 0.20) < 1e-12]
    if len(target) != 1:
        raise RuntimeError("formal A2 FPR20 operating point is unavailable")
    candidate.threshold = float(target[0]["threshold"])
    candidate.calibration["a2_fixed_operating_point"] = target[0]
    path = _save_candidate(candidate, output / "models" / "primary.pt")
    return candidate, result, float(result["adaptation_seconds"]), path


def _fit_g2(
    config: TrainingConfig,
    images_128: list[np.ndarray],
    masks_128: np.ndarray | None,
    labels: np.ndarray,
    output: Path,
) -> tuple[DeepPatchCoreLite, DeepPatchCoreLite, dict[str, Any], float, Path]:
    source_path = config.source_assets["source_model"].path
    source = DeepPatchCoreLite.load(source_path, map_location=config.device)
    source_backend = TorchMapBackend(source, device=config.device)
    source_started = time.perf_counter()
    source_maps, source_raw = _backend_maps(source_backend, images_128)
    source_normalizer, source_threshold, source_evidence = calibrate_image_scores(
        labels, source_raw
    )
    source_pixel_threshold, source_pixel_evidence = _calibrate_pixel_threshold(
        masks_128, source_maps, labels, seed=config.seed + 10
    )
    source_calibration_seconds = time.perf_counter() - source_started

    adapted_started = time.perf_counter()
    adapted = DeepPatchCoreLite(
        input_size=(128, 128),
        max_memory=512,
        projection_dim=64,
        seed=config.seed + 128,
        pretrained=False,
        device=config.device,
    )
    adapted.extractor.load_state_dict(source.extractor.state_dict())
    adapted.fit(
        [image for image, label in zip(images_128, labels, strict=True) if label == 0],
        batch_size=8,
    )
    adapted_path = _save_patchcore(adapted, output / "models" / "adapted.pt")
    adapted_backend = TorchMapBackend(adapted, device=config.device)
    adapted_maps, adapted_raw = _backend_maps(adapted_backend, images_128)
    adapted_normalizer, adapted_threshold, adapted_evidence = calibrate_image_scores(
        labels, adapted_raw
    )
    adapted_pixel_threshold, adapted_pixel_evidence = _calibrate_pixel_threshold(
        masks_128, adapted_maps, labels, seed=config.seed + 20
    )
    adapted_seconds = time.perf_counter() - adapted_started

    fusion_started = time.perf_counter()
    source_scores = source_normalizer.transform(source_raw)
    adapted_scores = adapted_normalizer.transform(adapted_raw)
    g2_scores = np.maximum(source_scores, adapted_scores)
    g2_threshold, g2_image_evidence = best_f1_threshold(labels, g2_scores)
    source_pixel_normalizer = pixel_normalizer(source_maps, labels)
    adapted_pixel_normalizer = pixel_normalizer(adapted_maps, labels)
    g2_maps = np.maximum(
        source_pixel_normalizer.transform(source_maps),
        adapted_pixel_normalizer.transform(adapted_maps),
    )
    g2_pixel_threshold, g2_pixel_evidence = _calibrate_pixel_threshold(
        masks_128, g2_maps, labels, seed=config.seed + 30
    )
    fusion_seconds = time.perf_counter() - fusion_started
    calibration = {
        "source_calibration": {
            "normalizer": source_normalizer.payload(),
            "image_threshold": float(source_threshold),
            "pixel_threshold": float(source_pixel_threshold),
            "image_evidence": source_evidence,
            "pixel_evidence": source_pixel_evidence,
        },
        "adapted_calibration": {
            "normalizer": adapted_normalizer.payload(),
            "image_threshold": float(adapted_threshold),
            "pixel_threshold": float(adapted_pixel_threshold),
            "image_evidence": adapted_evidence,
            "pixel_evidence": adapted_pixel_evidence,
        },
        "g2_image_threshold": float(g2_threshold),
        "g2_pixel_threshold": float(g2_pixel_threshold),
        "source_pixel_normalizer": source_pixel_normalizer.payload(),
        "adapted_pixel_normalizer": adapted_pixel_normalizer.payload(),
        "g2_image_evidence": g2_image_evidence,
        "g2_pixel_evidence": g2_pixel_evidence,
        "source_calibration_seconds_informational": source_calibration_seconds,
        "adapted_model_sha256": sha256_file(adapted_path),
    }
    calibration["calibration_payload_sha256"] = payload_hash(calibration)
    return source, adapted, calibration, adapted_seconds + fusion_seconds, adapted_path


def _fit_dino(
    config: TrainingConfig,
    normals: list[np.ndarray],
    defects: list[np.ndarray],
    masks: list[np.ndarray] | None,
    output: Path,
) -> tuple[DINOv2SpatialPrototype, dict[str, Any], float, Path]:
    dino = DINOv2SpatialPrototype(
        f"{config.candidate_id}-DINO",
        repository_path=config.source_assets["dinov2_repository"].path,
        weight_path=config.source_assets["dinov2_weights"].path,
        input_size=252,
        score_topk=16,
        target_normal_fpr=0.20,
        seed=20261531,
        device=config.device,
    )
    dino.fit_base([])
    result = dino.adapt(normals, defects, defect_labels=masks)
    path = _save_candidate(dino, output / "models" / "dino.pt")
    return dino, result, float(result["adaptation_seconds"]), path


def _fit_head(
    config: TrainingConfig,
    dino: DINOv2SpatialPrototype,
    normals: list[np.ndarray],
    defects: list[np.ndarray],
    masks: list[np.ndarray] | None,
    output: Path,
) -> tuple[DinoTiledMLPKnownHead, dict[str, Any], float, Path, dict[str, Any]]:
    normal_tiles = [nine_tiles(image) for image in normals]
    defect_calibration_tiles = [nine_tiles(image) for image in defects]
    if masks is None:
        defect_tiles = []
        for index, image in enumerate(defects):
            tiles = nine_tiles(image)
            predictions = dino.predict_batch(
                tiles,
                sample_ids=[f"weak-{index:03d}-tile-{tile_index}" for tile_index in range(9)],
            )
            selected = int(np.argmax([prediction.score for prediction in predictions]))
            defect_tiles.append(
                (tiles[selected], np.ones((252, 252), dtype=np.uint8))
            )
        supervision_mode = "image_label_weak_auto_tile"
    else:
        defect_tiles = [
            nine_tile_defect(image, mask)
            for image, mask in zip(defects, masks, strict=True)
        ]
        supervision_mode = "pixel_mask"
    head = DinoTiledMLPKnownHead(
        tile_count=9,
        score_topk=4,
        max_patches_per_defect=16,
        mask_dilation=1,
        hidden_dimension=32,
        epochs=config.head_epochs,
        learning_rate=config.head_learning_rate,
        weight_decay=0.0001,
        negative_patches_per_tile=8,
        seed=20261608,
        device=config.device,
    )
    result = head.fit(
        dino,
        normal_tiles,
        [item[0] for item in defect_tiles],
        [item[1] for item in defect_tiles],
        defect_calibration_tile_images=defect_calibration_tiles,
    )
    result["supervision_mode"] = supervision_mode
    path = head.save(output / "models" / "head.pt")
    point = target_point(
        head.calibration_normal_scores,
        head.calibration_defect_scores,
        0.10,
        0.0,
    )
    return head, result, float(result["adaptation_seconds"]), path, point


@torch.inference_mode()
def _leave_one_out_known_scores(dino: DINOv2SpatialPrototype) -> np.ndarray:
    """Match deployment k-NN scores without letting support samples match themselves."""

    if dino.normal_global_bank is None or dino.defect_global_bank is None:
        raise RuntimeError("DINO known-score banks are unavailable")
    normal = torch.nn.functional.normalize(dino.normal_global_bank, dim=1)
    defect = torch.nn.functional.normalize(dino.defect_global_bank, dim=1)
    if len(normal) != 100 or len(defect) != 30:
        raise ValueError("known-score calibration requires exactly 100 normal + 30 defect")
    normal_similarity = normal @ normal.T
    defect_similarity = defect @ defect.T
    normal_similarity.fill_diagonal_(-torch.inf)
    defect_similarity.fill_diagonal_(-torch.inf)
    normal_scores = (
        (normal @ defect.T).max(dim=1).values
        - normal_similarity.max(dim=1).values
    )
    defect_scores = (
        defect_similarity.max(dim=1).values
        - (defect @ normal.T).max(dim=1).values
    )
    return torch.cat((normal_scores, defect_scores)).cpu().numpy().astype(np.float64)


def _general_scores(
    images: list[np.ndarray],
    ids: list[str],
    *,
    source: DeepPatchCoreLite,
    adapted: DeepPatchCoreLite,
    primary: MultiScalePatchMemory,
    dino: DINOv2SpatialPrototype,
    calibration: dict[str, Any],
    device: str,
    known_score_overrides: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    small = _resize_images(images)
    source_maps, source_raw = _backend_maps(TorchMapBackend(source, device=device), small)
    adapted_maps, adapted_raw = _backend_maps(TorchMapBackend(adapted, device=device), small)
    del source_maps, adapted_maps
    source_norm = RobustNormalizer(**calibration["source_calibration"]["normalizer"])
    adapted_norm = RobustNormalizer(**calibration["adapted_calibration"]["normalizer"])
    g2_scores = np.maximum(source_norm.transform(source_raw), adapted_norm.transform(adapted_raw))
    primary_predictions = primary.predict_batch(images, sample_ids=ids)
    dino_predictions = dino.predict_batch(images, sample_ids=ids)
    details = []
    scores = []
    known_values = (
        np.asarray(known_score_overrides, dtype=np.float64)
        if known_score_overrides is not None
        else np.asarray(
            [prediction.metadata["known_score"] for prediction in dino_predictions],
            dtype=np.float64,
        )
    )
    if known_values.shape != (len(images),):
        raise ValueError("known score overrides must align with images")
    for primary_prediction, dino_prediction, g2_score, known_score in zip(
        primary_predictions, dino_predictions, g2_scores, known_values, strict=True
    ):
        detail = fusion_score(
            primary_score=float(primary_prediction.score),
            primary_threshold=float(primary_prediction.threshold),
            g2_score=float(g2_score),
            g2_threshold=float(calibration["g2_image_threshold"]),
            dino_score=float(dino_prediction.score),
            dino_threshold=float(dino_prediction.threshold),
            known_score=float(known_score),
            known_threshold=float(dino_prediction.metadata["known_threshold"]),
        )
        details.append(detail)
        scores.append(detail["general_score"])
    return np.asarray(scores, dtype=np.float64), details


def adapt_episode(
    config: TrainingConfig,
    dataset: TrainingDataset,
    run_dir: Path,
) -> AdaptationArtifacts:
    normals, defects, defect_masks = dataset_arrays(dataset)
    all_images = [*normals, *defects]
    # Every downstream calibration vector is ordered normal-first, defect-second.
    # The manifest itself may be interleaved, so derive IDs from the same grouped
    # views used to build all_images and labels.
    all_ids = [sample.sample_id for sample in dataset.normal] + [
        sample.sample_id for sample in dataset.defect
    ]
    labels = np.asarray([0] * 100 + [1] * 30, dtype=np.uint8)
    images_128 = _resize_images(all_images)
    masks_128 = None
    if defect_masks is not None:
        masks_128 = np.stack(
            [
                *[np.zeros((128, 128), dtype=np.uint8) for _ in normals],
                *_resize_masks(defect_masks),
            ]
        )

    primary, primary_receipt, primary_seconds, primary_path = _fit_primary(
        config, normals, defects, defect_masks, run_dir
    )
    source, adapted, g2_calibration, g2_seconds, adapted_path = _fit_g2(
        config, images_128, masks_128, labels, run_dir
    )
    dino, dino_receipt, dino_seconds, dino_path = _fit_dino(
        config, normals, defects, defect_masks, run_dir
    )
    head, head_receipt, head_seconds, head_path, local_point = _fit_head(
        config, dino, normals, defects, defect_masks, run_dir
    )
    known_scores = _leave_one_out_known_scores(dino)
    general_scores, score_details = _general_scores(
        all_images,
        all_ids,
        source=source,
        adapted=adapted,
        primary=primary,
        dino=dino,
        calibration=g2_calibration,
        device=config.device,
        known_score_overrides=known_scores,
    )
    local_scores = np.concatenate(
        (
            np.asarray(head.calibration_normal_scores, dtype=np.float64),
            np.asarray(head.calibration_defect_scores, dtype=np.float64),
        )
    )
    if head.final_runtime_normal_scores is None or head.final_runtime_defect_scores is None:
        raise RuntimeError("final-head runtime calibration scores are unavailable")
    final_runtime_local_scores = np.concatenate(
        (
            np.asarray(head.final_runtime_normal_scores, dtype=np.float64),
            np.asarray(head.final_runtime_defect_scores, dtype=np.float64),
        )
    )
    local_runtime_transform = fit_runtime_score_transform(
        local_scores,
        final_runtime_local_scores,
        labels,
    )
    aligned_runtime_local_scores = np.asarray(
        [
            transform_runtime_score(value, local_runtime_transform)
            for value in final_runtime_local_scores
        ],
        dtype=np.float64,
    )
    continuous_fusion = fit_five_fold_continuous_fusion(
        labels=labels,
        sample_ids=all_ids,
        branch_scores={
            "general": general_scores,
            "local": local_scores,
            "known": known_scores,
        },
        runtime_transforms={"local": local_runtime_transform},
        refit_branch_scores={
            "general": general_scores,
            "local": aligned_runtime_local_scores,
            "known": known_scores,
        },
        acceptance_minimum_recall=0.95,
        branch_minimum_recall=0.95,
        # Fold fitting keeps an all-defect safety buffer; the final deployed
        # threshold uses the exact hard acceptance floor and is not relaxed for FN.
        fold_fit_minimum_recall=0.98,
        final_fit_minimum_recall=0.95,
    )
    general_threshold = float(
        continuous_fusion["branches"]["general"]["threshold"]
    )
    local_threshold = float(continuous_fusion["branches"]["local"]["threshold"])
    known_threshold = float(continuous_fusion["branches"]["known"]["threshold"])
    legacy_general_threshold = normal_only_threshold(general_scores[:100], 0.05)
    scale_calibration, scale_signatures = calibrate_scale_detector(normals)

    calibration: dict[str, Any] = {
        **g2_calibration,
        "protocol": "support-only-5fold-continuous-fusion-v2",
        "episode_id": config.episode_id,
        "candidate_id": config.candidate_id,
        "adaptation_normal_count": 100,
        "adaptation_known_count": 30,
        "annotation_mode": "image_label" if defect_masks is None else "pixel_mask",
        "a2": {
            "threshold": float(primary.threshold),
            "pixel_threshold": float(primary.pixel_threshold),
            "calibration": primary.calibration,
            "adaptation": primary_receipt,
        },
        "dino": {
            "threshold": float(dino.threshold),
            "pixel_threshold": float(dino.pixel_threshold),
            "known_reference_threshold": float(dino.known_threshold),
            "calibration": dino.calibration,
            "adaptation": dino_receipt,
        },
        "head": {
            "training": head_receipt,
            "legacy_local_target_normal_fpr": 0.10,
            "legacy_local_operating_point": local_point,
            "local_threshold": local_threshold,
            "runtime_score_transform": local_runtime_transform,
        },
        "general": {
            "legacy_target_normal_fpr": 0.05,
            "legacy_general_threshold": legacy_general_threshold,
            "general_threshold": general_threshold,
            "adaptation_scores": [
                {
                    "sample_id": sample_id,
                    "role": "normal" if index < 100 else "defect",
                    "general_score": float(general_scores[index]),
                    "local_calibration_score": float(local_scores[index]),
                    "local_final_runtime_score": float(final_runtime_local_scores[index]),
                    "known_score_leave_one_out": float(known_scores[index]),
                    **score_details[index],
                }
                for index, sample_id in enumerate(all_ids)
            ],
        },
        "known": {
            "known_threshold": known_threshold,
            "reference_threshold": float(dino.known_threshold),
            "support_score_protocol": "relative_knn_leave_one_sample_out",
        },
        "continuous_fusion": continuous_fusion,
        "d44_scale_calibration": {
            str(scale): values for scale, values in scale_calibration.items()
        },
        "d44_scale_signatures": scale_signatures,
        "runtime": {
            "tile_grid": 3,
            "routing": "continuous_nonnegative_linear_fusion",
            "binary_or_disabled": True,
            "backbones_frozen": True,
            "trainable_component": "DinoTiledMLPKnownHead",
            "local_head_supervision": (
                "image_label_weak_auto_tile" if defect_masks is None else "pixel_mask"
            ),
        },
    }
    calibration["payload_sha256"] = payload_hash(calibration)
    calibration_path = run_dir / "calibration" / "calibration.yaml"
    write_yaml(calibration_path, calibration)
    return AdaptationArtifacts(
        primary=primary,
        adapted=adapted,
        dino=dino,
        head=head,
        source=source,
        calibration=calibration,
        model_paths={
            "primary": primary_path,
            "adapted": adapted_path,
            "dino": dino_path,
            "head": head_path,
            "calibration": calibration_path,
        },
        adaptation_seconds={
            "primary": primary_seconds,
            "g2": g2_seconds,
            "dino": dino_seconds,
            "head": head_seconds,
            "total_algorithm": primary_seconds + g2_seconds + dino_seconds + head_seconds,
        },
        input_samples=[normals[0], defects[0]],
        input_ids=[dataset.normal[0].sample_id, dataset.defect[0].sample_id],
    )
