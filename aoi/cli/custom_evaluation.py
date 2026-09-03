"""Labeled, support-disjoint evaluation for a private 100+30 AOI task."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from aoi.custom_manifest import CustomTestSample, load_custom_test_manifest
from aoi.images import read_rgb
from aoi.paths import PROJECT_ROOT, resolve_artifact
from aoi.runtime.config import resolve_device
from aoi.runtime.metrics import binary_metrics, operating_point_summary, per_defect_type
from aoi.runtime.runtime import AdaptedRuntime
from aoi.training.manifests import sha256_file


@dataclass(frozen=True)
class CustomEvaluation:
    config_path: Path
    pointer_path: Path
    bundle_path: Path
    category: str
    manifest_path: Path
    samples: tuple[CustomTestSample, ...]
    device: str
    batch_size: int
    output_root: Path
    refuse_overwrite: bool
    evaluation_role: str
    fresh_independent_test_attested: bool
    test_observed_before_freeze: bool
    minimum_total_images: int = 1000
    recall_floor: float = 0.95
    stage1_fpr_strict_upper_bound: float = 0.10
    stage2_fpr_strict_upper_bound: float = 0.05


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a YAML mapping")
    return value


def _boolean(payload: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if type(value) is not bool:
        raise TypeError(f"{key} must be a YAML boolean")
    return value


def _acceptance_thresholds(
    payload: Mapping[str, Any],
) -> tuple[float, float, float]:
    if "acceptance_thresholds" not in payload:
        return 0.95, 0.10, 0.05
    raw = payload["acceptance_thresholds"]
    thresholds = _mapping(raw, "acceptance_thresholds")
    allowed = {
        "recall_floor",
        "stage1_fpr_strict_upper_bound",
        "stage2_fpr_strict_upper_bound",
    }
    unknown = sorted(set(thresholds) - allowed)
    if unknown:
        raise ValueError(
            f"acceptance_thresholds contains unsupported fields: {unknown}"
        )

    def probability(key: str, default: float, *, allow_zero: bool) -> float:
        value = thresholds.get(key, default)
        context = f"acceptance_thresholds.{key}"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{context} must be a finite number")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{context} must be finite")
        lower_bound_valid = result >= 0.0 if allow_zero else result > 0.0
        if not lower_bound_valid or result > 1.0:
            interval = "[0, 1]" if allow_zero else "(0, 1]"
            raise ValueError(f"{context} must be in {interval}")
        return result

    recall_floor = probability("recall_floor", 0.95, allow_zero=True)
    stage1_fpr = probability(
        "stage1_fpr_strict_upper_bound", 0.10, allow_zero=False
    )
    stage2_fpr = probability(
        "stage2_fpr_strict_upper_bound", 0.05, allow_zero=False
    )
    if stage2_fpr > stage1_fpr:
        raise ValueError(
            "acceptance_thresholds.stage2_fpr_strict_upper_bound must be <= "
            "acceptance_thresholds.stage1_fpr_strict_upper_bound"
        )
    return recall_floor, stage1_fpr, stage2_fpr


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _file_evidence(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _project_path(value: Any, context: str, *, strict: bool = True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError(f"{context} must be a non-empty path")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    return candidate.resolve(strict=strict)


def prepare_custom_evaluation(config_path: str | Path) -> CustomEvaluation:
    path = Path(config_path).resolve(strict=True)
    payload = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")), "custom test config"
    )
    allowed = {
        "schema_version",
        "active_models_pointer",
        "category",
        "test_manifest",
        "device",
        "batch_size",
        "minimum_total_images",
        "output_root",
        "timestamp_output",
        "refuse_overwrite",
        "evaluation_role",
        "fresh_independent_test_attested",
        "test_observed_before_freeze",
        "acceptance_thresholds",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"custom test config contains unsupported fields: {unknown}")
    if payload.get("schema_version") != 1 or type(payload.get("schema_version")) is not int:
        raise ValueError("custom test config schema_version must be integer 1")
    recall_floor, stage1_fpr, stage2_fpr = _acceptance_thresholds(payload)

    pointer_path = _project_path(
        payload.get("active_models_pointer"), "active_models_pointer"
    )
    pointer = _mapping(
        yaml.safe_load(pointer_path.read_text(encoding="utf-8")),
        "custom active_models pointer",
    )
    if pointer.get("dataset") != "custom":
        raise ValueError(
            f"CUSTOM_EVALUATION_DATASET_MISMATCH: pointer dataset={pointer.get('dataset')!r}"
        )
    models = _mapping(pointer.get("models"), "custom active_models pointer.models")
    raw_category = payload.get("category")
    if raw_category is None:
        if len(models) != 1:
            raise ValueError("category is required when the pointer has multiple models")
        category = str(next(iter(models)))
    elif isinstance(raw_category, str) and raw_category.strip():
        category = raw_category.strip()
    else:
        raise TypeError("category must be a non-empty string")
    if category not in models:
        raise KeyError(
            f"category {category!r} has no private model; available: {sorted(models)}"
        )

    bundle_path = resolve_artifact(models[category], base=pointer_path.parent)
    bundle = _mapping(
        yaml.safe_load(bundle_path.read_text(encoding="utf-8")), "custom model bundle"
    )
    if bundle.get("target_dataset") != "custom" or bundle.get("target_object") != category:
        raise ValueError(
            "CUSTOM_EVALUATION_TARGET_MISMATCH: "
            f"pointer={category}, bundle dataset={bundle.get('target_dataset')}, "
            f"bundle target={bundle.get('target_object')}"
        )

    evaluation_role = str(payload.get("evaluation_role", "independent_test")).strip()
    if evaluation_role not in {"independent_test", "development_diagnostic"}:
        raise ValueError(
            "evaluation_role must be independent_test or development_diagnostic"
        )
    minimum = int(payload.get("minimum_total_images", 1000))
    if minimum < 1:
        raise ValueError("minimum_total_images must be >= 1")
    if evaluation_role == "independent_test" and minimum < 1000:
        raise ValueError(
            "independent_test minimum_total_images must be >= 1000; "
            "use development_diagnostic for a smaller labeled holdout"
        )
    manifest_path = _project_path(payload.get("test_manifest"), "test_manifest")
    raw_support_hashes = bundle.get("support_content_hashes")
    if not isinstance(raw_support_hashes, list) or len(raw_support_hashes) != 130:
        raise ValueError("custom model bundle must record exactly 130 support hashes")
    support_hashes = {str(value).lower() for value in raw_support_hashes}
    if len(support_hashes) != 130 or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in support_hashes
    ):
        raise ValueError("custom model bundle contains invalid or duplicate support hashes")
    support_counts = bundle.get("support_counts")
    if support_counts != {"normal": 100, "defect": 30}:
        raise ValueError("custom model bundle support_counts must be 100 normal + 30 defect")
    samples = load_custom_test_manifest(
        manifest_path,
        minimum_samples=minimum,
        excluded_hashes=support_hashes,
    )

    batch_size = int(payload.get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    refuse_overwrite = payload.get("refuse_overwrite", True)
    timestamp_output = payload.get("timestamp_output", True)
    if type(refuse_overwrite) is not bool or type(timestamp_output) is not bool:
        raise TypeError("refuse_overwrite and timestamp_output must be booleans")
    raw_output = payload.get("output_root")
    output_root = (
        pointer_path.parent / "test_runs"
        if raw_output is None
        else _project_path(raw_output, "output_root", strict=False)
    )
    if timestamp_output:
        output_root /= datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fresh_independent_test_attested = _boolean(
        payload, "fresh_independent_test_attested", default=False
    )
    test_observed_before_freeze = _boolean(
        payload, "test_observed_before_freeze", default=True
    )
    return CustomEvaluation(
        config_path=path,
        pointer_path=pointer_path,
        bundle_path=bundle_path,
        category=category,
        manifest_path=manifest_path,
        samples=samples,
        device=resolve_device(str(payload.get("device", "auto"))),
        batch_size=batch_size,
        output_root=output_root.resolve(),
        refuse_overwrite=refuse_overwrite,
        evaluation_role=evaluation_role,
        fresh_independent_test_attested=fresh_independent_test_attested,
        test_observed_before_freeze=test_observed_before_freeze,
        minimum_total_images=minimum,
        recall_floor=recall_floor,
        stage1_fpr_strict_upper_bound=stage1_fpr,
        stage2_fpr_strict_upper_bound=stage2_fpr,
    )


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _latency(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def run(config_path: str | Path) -> int:
    prepared = prepare_custom_evaluation(config_path)
    output_files = (
        prepared.output_root / "metrics.yaml",
        prepared.output_root / "predictions.yaml",
    )
    if prepared.refuse_overwrite and any(path.exists() for path in output_files):
        raise FileExistsError(
            f"refusing to overwrite private evaluation evidence: {prepared.output_root}"
        )

    print("=== AOI PRIVATE LABELED TEST ===")
    print(f"Category : {prepared.category}")
    print(f"Manifest : {prepared.manifest_path}")
    print(f"Images   : {len(prepared.samples)}")
    print(f"Device   : {prepared.device}")
    runtime = AdaptedRuntime(prepared.bundle_path, device=prepared.device)
    runtime_integrity = getattr(runtime, "integrity", None)
    if not isinstance(runtime_integrity, Mapping):
        raise TypeError("adapted runtime must expose an integrity receipt")
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for start in range(0, len(prepared.samples), prepared.batch_size):
        batch = prepared.samples[start : start + prepared.batch_size]
        images = [read_rgb(sample.path) for sample in batch]
        ids = [sample.sample_id for sample in batch]
        _synchronize(prepared.device)
        tick = time.perf_counter()
        scored = runtime.predict_batch(images, ids)
        _synchronize(prepared.device)
        elapsed_ms = (time.perf_counter() - tick) * 1000.0
        if len(scored) != len(batch):
            raise RuntimeError("runtime returned a different number of predictions")
        per_image = elapsed_ms / len(batch)
        latencies.extend([per_image] * len(batch))
        for sample, result in zip(batch, scored, strict=True):
            rows.append(
                {
                    **result,
                    "sample_id": sample.sample_id,
                    "path": str(sample.path),
                    "image_sha256": sample.sha256,
                    "category": prepared.category,
                    "label": sample.label,
                    "defect_type": sample.defect_type,
                    "prediction": int(bool(result["decision"])),
                    "latency_ms": per_image,
                }
            )
        done = min(start + prepared.batch_size, len(prepared.samples))
        if done % 100 == 0 or done == len(prepared.samples):
            print(f"[{prepared.category}] {done}/{len(prepared.samples)}")

    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    metrics: dict[str, Any] = binary_metrics(labels, predictions)
    metrics["latency_ms"] = _latency(latencies)
    metrics["per_defect_type"] = per_defect_type(rows)
    metrics["operating_point"] = operating_point_summary(rows)
    metrics["protocol"] = (
        "custom private 100+30 support -> labeled 1000+ independent test"
        if prepared.evaluation_role == "independent_test"
        else "custom 100+30 support -> labeled development holdout diagnostic"
    )
    metrics["dataset"] = "custom"
    metrics["category"] = prepared.category
    metrics["evaluation_role"] = prepared.evaluation_role
    metrics["configured_minimum_total_images"] = prepared.minimum_total_images
    sample_ids = [str(row["sample_id"]) for row in rows]
    sample_hashes = [row["image_sha256"] for row in rows]
    hashes_valid = all(_is_sha256(value) for value in sample_hashes)
    hashes_unique = hashes_valid and len(set(sample_hashes)) == len(sample_hashes)
    sample_ids_unique = len(set(sample_ids)) == len(sample_ids)
    integrity_verified = bool(
        runtime_integrity.get("bundle_hash_verified") is True
        and runtime_integrity.get("artifact_hashes_verified") is True
        and runtime_integrity.get("legacy_bundle_without_artifact_hashes") is False
    )
    claim_failures: list[str] = []
    if prepared.evaluation_role != "independent_test":
        claim_failures.append("evaluation_role must be independent_test")
    if not prepared.fresh_independent_test_attested:
        claim_failures.append("fresh post-freeze independent-test attestation is required")
    if prepared.test_observed_before_freeze:
        claim_failures.append("test data was observed before model and threshold freeze")
    if len(rows) < 1000:
        claim_failures.append("at least 1000 independent test images are required")
    if not sample_ids_unique:
        claim_failures.append("test sample IDs must be unique")
    if not hashes_unique:
        claim_failures.append("test image SHA-256 values must be valid and unique")
    if not integrity_verified:
        claim_failures.append("model bundle and artifact hashes must be verified")
    claim_eligible = not claim_failures
    metrics["fresh_independent_test_attested"] = (
        prepared.fresh_independent_test_attested
    )
    metrics["test_observed_before_freeze"] = prepared.test_observed_before_freeze
    metrics["eligible_for_final_independent_claim"] = claim_eligible
    metrics["claim_eligibility_reason"] = (
        "fresh post-freeze private test with 1000+ unique content-bound samples and "
        "verified model artifacts"
        if claim_eligible
        else "; ".join(claim_failures)
    )
    metrics["claim_protocol_verification"] = {
        "protocol_id": "custom_private_100_plus_30_to_1000_plus_v1",
        "observed_total_images": len(rows),
        "minimum_total_images": 1000,
        "sample_ids_unique": sample_ids_unique,
        "prediction_sample_hashes_present": hashes_valid,
        "prediction_sample_hashes_unique": hashes_unique,
        "bundle_and_artifact_integrity_verified": integrity_verified,
    }
    metrics["selection_scope"] = "100_normal_plus_30_defect_support_only"
    metrics["evaluation_images_used_for_calibration_or_fusion_search"] = False
    metrics["test_manifest"] = str(prepared.manifest_path)
    metrics["test_manifest_sha256"] = sha256_file(prepared.manifest_path)
    metrics["test_config"] = str(prepared.config_path)
    metrics["test_config_sha256"] = sha256_file(prepared.config_path)
    metrics["active_models_pointer"] = str(prepared.pointer_path)
    metrics["active_models_pointer_sha256"] = sha256_file(prepared.pointer_path)
    metrics["model_bundle"] = str(prepared.bundle_path)
    metrics["model_bundle_sha256"] = sha256_file(prepared.bundle_path)
    metrics["support_test_content_overlap"] = False
    metrics["label_counts"] = {
        "normal": sum(label == 0 for label in labels),
        "defect": sum(label == 1 for label in labels),
    }
    metrics["note"] = (
        "The loader verified unique test content and no SHA-256 overlap with the "
        "100+30 support. Independent provenance still depends on how the supplied "
        "manifest was collected and frozen."
    )
    stage1 = bool(
        float(metrics["recall"]) >= prepared.recall_floor
        and float(metrics["fpr"]) < prepared.stage1_fpr_strict_upper_bound
    )
    stage2 = bool(
        float(metrics["recall"]) >= prepared.recall_floor
        and float(metrics["fpr"]) < prepared.stage2_fpr_strict_upper_bound
    )
    scope = "INDEPENDENT_TEST" if claim_eligible else "DEVELOPMENT_DIAGNOSTIC"
    metrics["acceptance"] = {
        "recall_floor": prepared.recall_floor,
        "stage1_fpr_strict_upper_bound": prepared.stage1_fpr_strict_upper_bound,
        "stage2_fpr_strict_upper_bound": prepared.stage2_fpr_strict_upper_bound,
        "stage1_go": stage1,
        "stage2_go": stage2,
        "status": (
            f"GO_STAGE2_{scope}"
            if stage2
            else f"GO_STAGE1_{scope}"
            if stage1
            else f"NO_GO_{scope}_GATE"
        ),
    }

    metrics["evidence"] = {
        "config": _file_evidence(prepared.config_path),
        "test_manifest": _file_evidence(prepared.manifest_path),
        "active_models_pointer": _file_evidence(prepared.pointer_path),
        "model_bundle": {
            **_file_evidence(prepared.bundle_path),
            "runtime_integrity": dict(runtime_integrity),
        },
        "implementation": {
            "evaluation": _file_evidence(Path(__file__)),
            "runtime": _file_evidence(
                Path(__file__).resolve().parents[1] / "runtime" / "runtime.py"
            ),
        },
    }

    prepared.output_root.mkdir(parents=True, exist_ok=True)
    output_files[1].write_text(
        yaml.safe_dump(rows, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    metrics["evidence"]["predictions"] = _file_evidence(output_files[1])
    output_files[0].write_text(
        yaml.safe_dump(metrics, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"FPR      : {metrics['fpr']:.4f}")
    print(f"Gate     : {metrics['acceptance']['status']}")
    print(f"Output   : {prepared.output_root}")
    return 0


__all__ = ["CustomEvaluation", "prepare_custom_evaluation", "run"]
