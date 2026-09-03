from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from aoi.training.calibration import select_global_category_thresholds
from aoi.training.manifests import payload_hash, sha256_file, write_yaml


def _yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a support-only pooled-FPR Stage-2 calibration attempt"
    )
    parser.add_argument("--input-pointer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    pointer_path = args.input_pointer.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    pointer = _yaml(pointer_path)
    models = pointer.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"input pointer has no models: {pointer_path}")

    loaded: dict[str, tuple[Path, dict, Path, dict]] = {}
    category_scores = {}
    for category, raw_bundle in sorted(models.items()):
        bundle_path = Path(raw_bundle).resolve(strict=True)
        bundle = _yaml(bundle_path)
        calibration_path = Path(bundle["calibration"]).resolve(strict=True)
        calibration = _yaml(calibration_path)
        rows = calibration["continuous_fusion"]["support_refit_predictions"]
        labels = np.asarray([row["label"] for row in rows], dtype=np.uint8)
        scores = np.asarray([row["fusion_score"] for row in rows], dtype=np.float64)
        category_scores[str(category)] = (labels, scores)
        loaded[str(category)] = (bundle_path, bundle, calibration_path, calibration)

    search = select_global_category_thresholds(
        category_scores,
        pooled_minimum_recall=0.95,
        category_minimum_recall=0.80,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    derived_models: dict[str, str] = {}
    summaries = []
    for category in sorted(loaded):
        bundle_path, source_bundle, calibration_path, source_calibration = loaded[category]
        selected = search["categories"][category]
        threshold = float(selected["threshold"])
        category_root = output_root / category
        (category_root / "calibration").mkdir(parents=True, exist_ok=False)

        calibration = deepcopy(source_calibration)
        continuous = calibration["continuous_fusion"]
        continuous["fusion"]["threshold"] = threshold
        continuous["fusion"]["support_refit_metrics_apparent"] = selected["metrics"]
        for row in continuous["support_refit_predictions"]:
            row["fusion_threshold"] = threshold
            row["prediction"] = int(float(row["fusion_score"]) >= threshold)
        calibration["protocol"] = "support-only-global-stage2-fpr-attempt-v4"
        calibration["calibration_candidate_id"] = "CALIBRATION-FUSION-V4-STAGE2-ATTEMPT"
        calibration["derived_from_calibration"] = str(calibration_path)
        calibration["derived_from_calibration_sha256"] = sha256_file(calibration_path)
        calibration["global_stage2_search"] = {
            "method": search["method"],
            "selection_scope": search["selection_scope"],
            "independent_evaluation_used_for_selection": False,
            "pooled_minimum_recall": search["pooled_minimum_recall"],
            "category_minimum_recall": search["category_minimum_recall"],
            "pooled_support_metrics": search["pooled_support_metrics"],
            "selected_category_operating_point": selected,
        }
        calibration["payload_sha256"] = "pending"
        calibration["payload_sha256"] = payload_hash(
            {key: value for key, value in calibration.items() if key != "payload_sha256"}
        )
        derived_calibration_path = category_root / "calibration" / "calibration.yaml"
        write_yaml(derived_calibration_path, calibration)

        bundle = deepcopy(source_bundle)
        bundle["run_id"] = f"{category}_calibration_fusion_v4_stage2_attempt"
        bundle["calibration"] = str(derived_calibration_path.resolve())
        bundle["derived_from_bundle"] = str(bundle_path)
        bundle["derived_from_bundle_sha256"] = sha256_file(bundle_path)
        bundle["calibration_only_update"] = True
        bundle["backbone_retrained"] = False
        bundle["candidate_status"] = "UNVALIDATED_STAGE2_ATTEMPT"
        bundle["thresholds"]["fusion"] = threshold
        bundle["fusion"]["support_refit_metrics_apparent"] = selected["metrics"]
        bundle["bundle_hash"] = "pending"
        bundle["bundle_hash"] = payload_hash(
            {key: value for key, value in bundle.items() if key != "bundle_hash"}
        )
        derived_bundle_path = category_root / "model_bundle.yaml"
        write_yaml(derived_bundle_path, bundle)
        (category_root / "READY.txt").write_text(
            "STAGE2_CANDIDATE_READY_FOR_DIAGNOSTIC_EVALUATION\n", encoding="utf-8"
        )
        derived_models[category] = str(derived_bundle_path.resolve())
        summaries.append(
            {
                "category": category,
                "fusion_threshold": threshold,
                "support_metrics": selected["metrics"],
            }
        )
        print(
            f"STAGE2_CANDIDATE_READY: {category} threshold={threshold:.6f} "
            f"support_recall={selected['metrics']['recall']:.4f} "
            f"support_fpr={selected['metrics']['fpr']:.4f}"
        )

    derived_pointer = deepcopy(pointer)
    derived_pointer["models"] = derived_models
    derived_pointer["calibration_protocol"] = "support-only-global-stage2-fpr-attempt-v4"
    derived_pointer["derived_from_pointer"] = str(pointer_path)
    derived_pointer["derived_from_pointer_sha256"] = sha256_file(pointer_path)
    derived_pointer["candidate_status"] = "UNVALIDATED_STAGE2_ATTEMPT"
    derived_pointer["backbone_retrained"] = False
    derived_pointer["evaluation_images_used_for_selection"] = False
    write_yaml(output_root / "active_models.yaml", derived_pointer)
    write_yaml(
        output_root / "stage2_search_summary.yaml",
        {
            "source_pointer": str(pointer_path),
            "source_pointer_sha256": sha256_file(pointer_path),
            "selection_scope": "support_only",
            "evaluation_images_used_for_selection": False,
            "backbone_retrained": False,
            "pooled_support_metrics": search["pooled_support_metrics"],
            "pooled_minimum_recall": search["pooled_minimum_recall"],
            "category_minimum_recall": search["category_minimum_recall"],
            "categories": summaries,
        },
    )
    print(f"STAGE2_SUPPORT_SEARCH_READY: {len(derived_models)} categories")
    print(f"Support metrics: {search['pooled_support_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
