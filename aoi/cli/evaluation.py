from __future__ import annotations

import hashlib
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from aoi.runtime.config import load_yaml, resolve_device, resolve_path
from aoi.runtime.data import load_dagm_category_test, load_mvtec_category_holdout, read_rgb
from aoi.runtime.metrics import binary_metrics, operating_point_summary, per_defect_type
from aoi.paths import resolve_artifact
from aoi.runtime.runtime import AdaptedRuntime


_DAGM2007_FINAL_CLAIM_COUNTS: tuple[tuple[str, int], ...] = tuple(
    (f"Class{index}", 575 if index <= 6 else 1150)
    for index in range(1, 11)
)
_DAGM2007_FINAL_CLAIM_TOTAL = 8050


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_evidence(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _final_claim_assessment(
    *,
    dataset: str,
    evaluation_role: str,
    fresh_independent_test_attested: bool,
    test_observed_before_freeze: bool,
    prediction_rows: list[dict[str, Any]],
    model_bundle_evidence: dict[str, dict[str, Any]],
) -> tuple[bool, str, dict[str, Any]]:
    expected_counts = dict(_DAGM2007_FINAL_CLAIM_COUNTS)
    observed_counts = dict(
        sorted(Counter(str(row.get("category", "")) for row in prediction_rows).items())
    )
    sample_ids = [str(row.get("sample_id", "")) for row in prediction_rows]
    sample_hashes = [row.get("image_sha256") for row in prediction_rows]
    hashes_present = all(_is_sha256(value) for value in sample_hashes)
    hashes_unique = hashes_present and len(set(sample_hashes)) == len(sample_hashes)
    sample_ids_unique = bool(sample_ids) and len(set(sample_ids)) == len(sample_ids)
    complete_coverage = bool(
        dataset == "dagm2007"
        and len(prediction_rows) == _DAGM2007_FINAL_CLAIM_TOTAL
        and observed_counts == expected_counts
        and sample_ids_unique
    )

    per_model_integrity: dict[str, bool] = {}
    for category, evidence in sorted(model_bundle_evidence.items()):
        runtime_integrity = evidence.get("runtime_integrity")
        per_model_integrity[category] = bool(
            _is_sha256(evidence.get("bundle_hash"))
            and isinstance(runtime_integrity, dict)
            and runtime_integrity.get("bundle_hash_verified") is True
            and runtime_integrity.get("artifact_hashes_verified") is True
            and runtime_integrity.get("legacy_bundle_without_artifact_hashes") is False
        )
    integrity_verified = bool(
        set(per_model_integrity) == set(expected_counts)
        and all(per_model_integrity.values())
    )

    verification = {
        "protocol_id": "dagm2007_official_test_v1",
        "requirements_source": "evaluation.py immutable competition protocol",
        "expected_categories": list(expected_counts),
        "expected_images_per_category": expected_counts,
        "expected_total_images": _DAGM2007_FINAL_CLAIM_TOTAL,
        "observed_images_per_category": observed_counts,
        "observed_total_images": len(prediction_rows),
        "complete_coverage_verified": complete_coverage,
        "sample_ids_unique": sample_ids_unique,
        "prediction_sample_hashes_present": hashes_present,
        "prediction_sample_hashes_unique": hashes_unique,
        "per_model_bundle_and_artifact_integrity": per_model_integrity,
        "bundle_and_artifact_integrity_verified": integrity_verified,
    }
    failures: list[str] = []
    if not fresh_independent_test_attested:
        failures.append("requires a fresh post-freeze independent-test attestation")
    if test_observed_before_freeze:
        failures.append("test data was observed before model and threshold freeze")
    if dataset != "dagm2007":
        failures.append("no immutable final-claim protocol is defined for this dataset")
    if evaluation_role != "independent_test":
        failures.append("evaluation_role must be independent_test")
    if not complete_coverage:
        failures.append("complete DAGM2007 Class1-Class10 / 8050-image coverage is required")
    if not hashes_present:
        failures.append("every prediction must bind a valid sample SHA-256")
    if not integrity_verified:
        failures.append("every model bundle and artifact hash must be verified")

    eligible = not failures
    reason = (
        "fresh post-freeze independent test, complete immutable protocol coverage, "
        "sample hashes, and all model artifacts verified"
        if eligible
        else "; ".join(failures)
    )
    return eligible, reason, verification


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return float(ordered[idx])


def _normalize_dataset_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "dagm": "dagm2007",
        "dagm_2007": "dagm2007",
        "dagm2007": "dagm2007",
        "mvtec": "mvtec_ad",
        "mvtec_ad": "mvtec_ad",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dataset {value!r}; expected dagm2007 or mvtec_ad"
        ) from exc


def _config_boolean(
    config: dict[str, object], key: str, *, default: bool
) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise TypeError(f"{key} must be a YAML boolean")
    return value


def run(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    if "dataset" in config:
        raw_dataset = config["dataset"]
    elif "mvtec_root" in config:
        raw_dataset = "mvtec_ad"
    else:
        raw_dataset = "dagm2007"
    dataset = _normalize_dataset_name(str(raw_dataset))
    device = resolve_device(str(config.get("device", "auto")))
    default_pointer = (
        "outputs/dagm2007/active_models.yaml"
        if dataset == "dagm2007"
        else "outputs/active_models.yaml"
    )
    pointer_path = resolve_path(
        config, str(config.get("active_models_pointer", default_pointer))
    )
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"No {dataset} adapted-model pointer at {pointer_path}. "
            "Run train.py first."
        )
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8")) or {}
    pointer_dataset = _normalize_dataset_name(str(pointer.get("dataset", dataset)))
    if pointer_dataset != dataset:
        raise ValueError(
            f"EVALUATION_DATASET_MISMATCH: config={dataset}, pointer={pointer_dataset}"
        )
    models = pointer.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError(f"active_models.yaml has no models: {pointer_path}")

    if "data_root" in config:
        raw_data_root = config["data_root"]
    elif dataset == "mvtec_ad":
        raw_data_root = config.get("mvtec_root", "data/mvtec_ad")
    else:
        raw_data_root = config.get("dagm_root", "data/dagm2007")
    data_root = resolve_path(config, str(raw_data_root))
    batch_size = int(config.get("batch_size", 4))
    minimum_total = int(config.get("minimum_total_images", 1000))
    output_root = resolve_path(
        config, str(config.get("output_root", "outputs/evaluation"))
    )
    if bool(config.get("timestamp_output", False)):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = output_root / stamp
    output_files = (output_root / "metrics.yaml", output_root / "predictions.yaml")
    if bool(config.get("refuse_overwrite", False)) and any(path.exists() for path in output_files):
        raise FileExistsError(f"refusing to overwrite existing evaluation evidence: {output_root}")

    dataset_label = "DAGM2007" if dataset == "dagm2007" else "MVTec AD"
    evaluation_role = str(
        config.get(
            "evaluation_role",
            "independent_test" if dataset == "dagm2007" else "development_diagnostic",
        )
    )
    print(f"=== AOI {dataset_label} EVALUATION ===")
    print(f"Dataset  : {dataset_label}")
    if dataset == "dagm2007":
        print("Protocol : 100+30 support/calibration from official Train -> full official Test")
        print("Rule     : no Test image may enter training, calibration, or fusion search")
    else:
        print("Protocol : per-category 100+30 support -> remaining development images")
        print("Rule     : historical MVTec holdout is development/diagnostic only")
    print(f"Data root: {data_root}")
    print(f"Models   : {len(models)} categories")
    print(f"Device   : {device}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")

    prediction_rows: list[dict] = []
    latencies: list[float] = []
    category_metrics: dict[str, dict] = {}
    model_bundle_evidence: dict[str, dict[str, Any]] = {}

    for category, raw_bundle in sorted(models.items()):
        bundle_path = resolve_artifact(raw_bundle, base=pointer_path.parent)
        bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {}
        if bundle.get("target_dataset") != dataset or bundle.get("target_object") != category:
            raise ValueError(
                "EVALUATION_TARGET_MISMATCH: "
                f"pointer={category}, bundle dataset={bundle.get('target_dataset')}, "
                f"bundle target={bundle.get('target_object')}"
            )
        bundle_evidence: dict[str, Any] = _file_evidence(bundle_path)
        bundle_evidence["bundle_hash"] = bundle.get("bundle_hash")
        model_bundle_evidence[str(category)] = bundle_evidence
        support_hashes = set(str(v) for v in bundle.get("support_content_hashes", []))
        loader = (
            load_dagm_category_test
            if dataset == "dagm2007"
            else load_mvtec_category_holdout
        )
        samples = loader(data_root, category, excluded_hashes=support_hashes)
        runtime = AdaptedRuntime(bundle_path, device=device)
        runtime_integrity = getattr(runtime, "integrity", None)
        if not isinstance(runtime_integrity, dict):
            raise TypeError("adapted runtime must expose an integrity receipt")
        bundle_evidence["runtime_integrity"] = runtime_integrity
        category_rows: list[dict] = []
        split_label = "official Test" if dataset == "dagm2007" else "development"
        print(f"\n[{category}] {split_label} images: {len(samples)}")

        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = [read_rgb(s.path) for s in batch]
            ids = [s.sample_id for s in batch]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            scored = runtime.predict_batch(images, ids)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_image = elapsed_ms / len(batch)
            latencies.extend([per_image] * len(batch))
            for sample, row in zip(batch, scored, strict=True):
                output = {
                    **row,
                    "sample_id": sample.sample_id,
                    "image_sha256": sample.sha256,
                    "path": str(sample.path),
                    "category": sample.category,
                    "label": sample.label,
                    "defect_type": sample.defect_type,
                    "prediction": int(bool(row["decision"])),
                    "latency_ms": per_image,
                }
                category_rows.append(output)
                prediction_rows.append(output)
            done = min(start + batch_size, len(samples))
            if done % 100 == 0 or done == len(samples):
                print(f"[{category}] {done}/{len(samples)}")

        category_result = binary_metrics(
            [int(r["label"]) for r in category_rows],
            [int(r["prediction"]) for r in category_rows],
        )
        category_result["operating_point"] = operating_point_summary(category_rows)
        continuous = runtime.calibration.get("continuous_fusion")
        category_result["support_calibration"] = (
            {
                "method": continuous["method"],
                "selection_scope": continuous["selection_scope"],
                "independent_evaluation_used_for_selection": continuous[
                    "independent_evaluation_used_for_selection"
                ],
                "oof_metrics": continuous["fusion"]["oof_metrics"],
                "oof_recall_constraint_met": continuous["fusion"][
                    "oof_recall_constraint_met"
                ],
                "candidate_count": continuous["fusion"]["candidate_count"],
            }
            if continuous is not None
            else {"method": "legacy_binary_or", "selection_scope": "historical"}
        )
        category_metrics[category] = category_result

    if len(prediction_rows) < minimum_total:
        raise ValueError(
            f"pooled {dataset} evaluation requires >= {minimum_total} images; "
            f"got {len(prediction_rows)}"
        )

    labels = [int(r["label"]) for r in prediction_rows]
    preds = [int(r["prediction"]) for r in prediction_rows]
    metrics = binary_metrics(labels, preds)
    metrics["latency_ms"] = {
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
    }
    metrics["per_category"] = category_metrics
    metrics["per_defect_type"] = per_defect_type(prediction_rows)
    if dataset == "dagm2007":
        metrics["protocol"] = (
            "DAGM2007 per-class 100+30 support from official Train; full official Test"
        )
    else:
        metrics["protocol"] = (
            "MVTec AD per-category 100+30 few-shot; reused development/diagnostic set"
        )
    metrics["dataset"] = dataset
    metrics["standard_mvtec_benchmark"] = False
    metrics["evaluation_role"] = evaluation_role
    independent_test_attestation = _config_boolean(
        config, "fresh_independent_test_attested", default=False
    )
    test_observed_before_freeze = _config_boolean(
        config, "test_observed_before_freeze", default=False
    )
    claim_eligible, claim_reason, claim_verification = _final_claim_assessment(
        dataset=dataset,
        evaluation_role=evaluation_role,
        fresh_independent_test_attested=independent_test_attestation,
        test_observed_before_freeze=test_observed_before_freeze,
        prediction_rows=prediction_rows,
        model_bundle_evidence=model_bundle_evidence,
    )
    metrics["standard_dagm2007_test_split"] = bool(
        claim_verification["complete_coverage_verified"]
    )
    metrics["fresh_independent_test_attested"] = independent_test_attestation
    metrics["test_observed_before_freeze"] = test_observed_before_freeze
    metrics["eligible_for_final_independent_claim"] = claim_eligible
    metrics["claim_eligibility_reason"] = claim_reason
    metrics["claim_protocol_verification"] = claim_verification
    metrics["selection_scope"] = "per_category_100_normal_plus_30_defect_support_only"
    metrics["evaluation_images_used_for_calibration_or_fusion_search"] = False
    metrics["note"] = (
        "Official DAGM2007 Test is disjoint from the Train-only support/calibration data. "
        "Freeze the model and thresholds before first Test evaluation and do not tune on Test."
        if dataset == "dagm2007"
        else "These MVTec images are development/diagnostic only and are not a fresh "
        "independent or standard unsupervised MVTec benchmark result."
    )
    raw_diagnostics = config.get(
        "fn_diagnostic_categories",
        [] if dataset == "dagm2007" else ["capsule", "pill", "screw"],
    )
    if not isinstance(raw_diagnostics, list):
        raise TypeError("fn_diagnostic_categories must be a YAML list")
    diagnostic_categories = {str(value) for value in raw_diagnostics}
    metrics["fn_diagnostics"] = {
        category: [
            {
                "sample_id": row["sample_id"],
                "path": row["path"],
                "defect_type": row["defect_type"],
                "fusion_score": row["fusion_score"],
                "fusion_threshold": row["fusion_threshold"],
                "general_score": row["general_score"],
                "general_threshold": row["general_threshold"],
                "local_score": row["local_score"],
                "local_threshold": row["local_threshold"],
                "known_score": row["known_score"],
                "known_threshold": row["known_threshold"],
                "branch_margins": row["branch_margins"],
            }
            for row in prediction_rows
            if row["category"] == category
            and int(row["label"]) == 1
            and not bool(row["prediction"])
        ]
        for category in sorted(diagnostic_categories)
    }
    stage1 = bool(float(metrics["recall"]) >= 0.95 and float(metrics["fpr"]) < 0.10)
    stage2 = bool(float(metrics["recall"]) >= 0.95 and float(metrics["fpr"]) < 0.05)
    status_scope = "INDEPENDENT_TEST" if claim_eligible else "DEVELOPMENT_DIAGNOSTIC"
    metrics["acceptance"] = {
        "recall_floor": 0.95,
        "stage1_fpr_strict_upper_bound": 0.10,
        "stage2_fpr_strict_upper_bound": 0.05,
        "stage1_go": stage1,
        "stage2_go": stage2,
        "status": (
            f"GO_STAGE2_{status_scope}"
            if stage2
            else f"GO_STAGE1_{status_scope}"
            if stage1
            else f"NO_GO_{status_scope}_GATE"
        ),
    }

    package_root = Path(__file__).resolve().parents[1]
    metrics["evidence"] = {
        "config": _file_evidence(Path(str(config["_config_path"]))),
        "active_models_pointer": _file_evidence(pointer_path),
        "model_bundles": model_bundle_evidence,
        "implementation": {
            "evaluation": _file_evidence(Path(__file__)),
            "runtime": _file_evidence(package_root / "runtime" / "runtime.py"),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.yaml"
    predictions_path.write_text(
        yaml.safe_dump(prediction_rows, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    metrics["evidence"]["predictions"] = _file_evidence(predictions_path)
    (output_root / "metrics.yaml").write_text(
        yaml.safe_dump(metrics, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print("\n=== POOLED EVALUATION METRICS ===")
    for key in (
        "images", "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1",
        "fpr", "specificity", "balanced_accuracy",
    ):
        print(f"{key:18s}: {metrics[key]}")
    print(f"P95 latency       : {metrics['latency_ms']['p95']:.3f} ms/image")
    print(f"Gate status       : {metrics['acceptance']['status']}")
    print(f"Evaluation role   : {evaluation_role}")
    print(f"Claim note        : {metrics['note']}")
    print(f"Output            : {output_root}")
    return 0
