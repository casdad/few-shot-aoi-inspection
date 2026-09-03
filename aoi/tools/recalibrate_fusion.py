from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from aoi.training.calibration import (
    fit_five_fold_continuous_fusion,
    transform_runtime_score,
)
from aoi.training.manifests import payload_hash, sha256_file, write_yaml


def _yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _recalibrate_category(
    *,
    category: str,
    source_bundle_path: Path,
    output_root: Path,
) -> tuple[Path, dict[str, object]]:
    source_bundle = _yaml(source_bundle_path)
    source_calibration_path = Path(source_bundle["calibration"]).resolve(strict=True)
    source_calibration = _yaml(source_calibration_path)
    rows = source_calibration["general"]["adaptation_scores"]
    if len(rows) != 130:
        raise ValueError(f"{category}: expected 130 support score rows, got {len(rows)}")
    labels = np.asarray(
        [0 if row["role"] == "normal" else 1 for row in rows], dtype=np.uint8
    )
    sample_ids = [str(row["sample_id"]) for row in rows]
    general_scores = np.asarray([row["general_score"] for row in rows], dtype=np.float64)
    local_oof_scores = np.asarray(
        [row["local_calibration_score"] for row in rows], dtype=np.float64
    )
    local_runtime_raw = np.asarray(
        [row["local_final_runtime_score"] for row in rows], dtype=np.float64
    )
    known_scores = np.asarray(
        [row["known_score_leave_one_out"] for row in rows], dtype=np.float64
    )
    local_transform = source_calibration["head"]["runtime_score_transform"]
    local_runtime_aligned = np.asarray(
        [transform_runtime_score(value, local_transform) for value in local_runtime_raw],
        dtype=np.float64,
    )
    continuous = fit_five_fold_continuous_fusion(
        labels=labels,
        sample_ids=sample_ids,
        branch_scores={
            "general": general_scores,
            "local": local_oof_scores,
            "known": known_scores,
        },
        runtime_transforms={"local": local_transform},
        refit_branch_scores={
            "general": general_scores,
            "local": local_runtime_aligned,
            "known": known_scores,
        },
        acceptance_minimum_recall=0.95,
        branch_minimum_recall=0.95,
        fold_fit_minimum_recall=0.98,
        final_fit_minimum_recall=0.95,
    )

    category_root = output_root / category
    if category_root.exists():
        raise FileExistsError(f"refusing to overwrite recalibration output: {category_root}")
    (category_root / "calibration").mkdir(parents=True, exist_ok=False)
    calibration = deepcopy(source_calibration)
    calibration["protocol"] = "support-only-5fold-continuous-fusion-v3-stage1"
    calibration["calibration_candidate_id"] = "CALIBRATION-FUSION-V3-STAGE1"
    calibration["derived_from_calibration"] = str(source_calibration_path)
    calibration["derived_from_calibration_sha256"] = sha256_file(source_calibration_path)
    calibration["continuous_fusion"] = continuous
    calibration["general"]["general_threshold"] = float(
        continuous["branches"]["general"]["threshold"]
    )
    calibration["head"]["local_threshold"] = float(
        continuous["branches"]["local"]["threshold"]
    )
    calibration["known"]["known_threshold"] = float(
        continuous["branches"]["known"]["threshold"]
    )
    calibration["runtime"]["routing"] = "continuous_nonnegative_linear_fusion"
    calibration["runtime"]["binary_or_disabled"] = True
    calibration["payload_sha256"] = "pending"
    calibration["payload_sha256"] = payload_hash(
        {key: value for key, value in calibration.items() if key != "payload_sha256"}
    )
    calibration_path = category_root / "calibration" / "calibration.yaml"
    write_yaml(calibration_path, calibration)

    for model_path in source_bundle["models"].values():
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"{category}: referenced model is missing: {model_path}")
    bundle = deepcopy(source_bundle)
    bundle["schema_version"] = 3
    bundle["run_id"] = f"{category}_calibration_fusion_v3_stage1"
    bundle["calibration"] = str(calibration_path.resolve())
    bundle["derived_from_bundle"] = str(source_bundle_path)
    bundle["derived_from_bundle_sha256"] = sha256_file(source_bundle_path)
    bundle["calibration_only_update"] = True
    bundle["backbone_retrained"] = False
    bundle["thresholds"] = {
        "general": float(calibration["general"]["general_threshold"]),
        "local": float(calibration["head"]["local_threshold"]),
        "known": float(calibration["known"]["known_threshold"]),
        "fusion": float(continuous["fusion"]["threshold"]),
    }
    bundle["fusion"] = {
        "schema": continuous["schema"],
        "aggregation": continuous["fusion"]["aggregation"],
        "weights": continuous["fusion"]["weights"],
        "support_oof_metrics": continuous["fusion"]["oof_metrics"],
        "support_refit_metrics_apparent": continuous["fusion"][
            "support_refit_metrics_apparent"
        ],
        "independent_evaluation_used_for_selection": False,
    }
    bundle["bundle_hash"] = "pending"
    bundle["bundle_hash"] = payload_hash(
        {key: value for key, value in bundle.items() if key != "bundle_hash"}
    )
    bundle_path = category_root / "model_bundle.yaml"
    write_yaml(bundle_path, bundle)
    (category_root / "READY.txt").write_text(
        "RECALIBRATED_MODEL_READY\n", encoding="utf-8"
    )
    summary = {
        "category": category,
        "source_bundle": str(source_bundle_path),
        "bundle": str(bundle_path.resolve()),
        "weights": continuous["fusion"]["weights"],
        "general_threshold": bundle["thresholds"]["general"],
        "local_threshold": bundle["thresholds"]["local"],
        "known_threshold": bundle["thresholds"]["known"],
        "fusion_threshold": bundle["thresholds"]["fusion"],
        "support_oof_metrics": continuous["fusion"]["oof_metrics"],
        "support_refit_metrics_apparent": continuous["fusion"][
            "support_refit_metrics_apparent"
        ],
    }
    return bundle_path.resolve(), summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive support-only fusion calibration without retraining backbones"
    )
    parser.add_argument("--input-pointer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    pointer_path = args.input_pointer.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    source_pointer = _yaml(pointer_path)
    models = source_pointer.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"input pointer has no models: {pointer_path}")

    derived_models: dict[str, str] = {}
    summaries = []
    for category, raw_bundle in sorted(models.items()):
        bundle_path, summary = _recalibrate_category(
            category=str(category),
            source_bundle_path=Path(raw_bundle).resolve(strict=True),
            output_root=output_root,
        )
        derived_models[str(category)] = str(bundle_path)
        summaries.append(summary)
        print(
            f"RECALIBRATED_MODEL_READY: {category} "
            f"OOF_R={summary['support_oof_metrics']['recall']:.4f} "
            f"FUSION_T={summary['fusion_threshold']:.6f}"
        )

    pointer = deepcopy(source_pointer)
    pointer["models"] = derived_models
    pointer["calibration_protocol"] = "support-only-5fold-continuous-fusion-v3-stage1"
    pointer["derived_from_pointer"] = str(pointer_path)
    pointer["derived_from_pointer_sha256"] = sha256_file(pointer_path)
    pointer["backbone_retrained"] = False
    pointer["evaluation_images_used_for_selection"] = False
    write_yaml(output_root / "active_models.yaml", pointer)
    write_yaml(
        output_root / "recalibration_summary.yaml",
        {
            "source_pointer": str(pointer_path),
            "source_pointer_sha256": sha256_file(pointer_path),
            "category_count": len(derived_models),
            "selection_scope": "per_category_100_normal_plus_30_defect_support_only",
            "evaluation_images_used_for_selection": False,
            "backbone_retrained": False,
            "categories": summaries,
        },
    )
    print(f"ALL_RECALIBRATED_MODELS_READY: {len(derived_models)}")
    print(f"Pointer: {output_root / 'active_models.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
