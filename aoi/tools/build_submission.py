"""Build and verify a size-gated, single-task AOI submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickletools
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from aoi.paths import PROJECT_ROOT, resolve_artifact


MAX_SUBMISSION_BYTES = 200_000_000
MAX_UNCOMPRESSED_BYTES = 1_000_000_000
MAX_MEMBER_UNCOMPRESSED_BYTES = 500_000_000
MAX_ARCHIVE_MEMBERS = 10_000
MAX_MEMBER_PATH_LENGTH = 240
MAX_MEMBER_DEPTH = 32
MAX_MEMBER_COMPRESSION_RATIO = 200.0
MAX_TOTAL_COMPRESSION_RATIO = 50.0
MANIFEST_NAME = "SUBMISSION_MANIFEST.yaml"
PACKAGE_CONFIG_NAME = "configs/submission_task.yaml"
PACKAGE_TRAIN_CONFIG_NAME = "configs/train.yaml"
PACKAGE_UI_TRAIN_CONFIG_NAME = "configs/custom_training_base.yaml"
PACKAGE_CUSTOM_TRAIN_NAME = "configs/custom_train.example.yaml"
PACKAGE_CUSTOM_TEST_NAME = "configs/custom_test.example.yaml"
PACKAGE_FEEDBACK_CONFIG_NAME = "configs/feedback.yaml"
PACKAGE_BENCHMARK_RTX2060_NAME = "configs/benchmark_official_rtx2060.yaml"
PACKAGE_BENCHMARK_RTX4070_NAME = "configs/benchmark_internal_rtx4070.yaml"
PACKAGE_BENCHMARK_CPU_NAME = "configs/benchmark_cpu_diagnostic.yaml"
PACKAGE_CONFIG_NAMES = [
    PACKAGE_CONFIG_NAME,
    PACKAGE_TRAIN_CONFIG_NAME,
    PACKAGE_UI_TRAIN_CONFIG_NAME,
    PACKAGE_CUSTOM_TRAIN_NAME,
    PACKAGE_CUSTOM_TEST_NAME,
    PACKAGE_FEEDBACK_CONFIG_NAME,
    PACKAGE_BENCHMARK_RTX2060_NAME,
    PACKAGE_BENCHMARK_RTX4070_NAME,
    PACKAGE_BENCHMARK_CPU_NAME,
]
PACKAGE_README_NAME = "SUBMISSION_README.txt"
PACKAGE_MODEL_GUIDE_NAME = "README_模型使用说明.md"
PACKAGE_NOTICES_NAME = "THIRD_PARTY_NOTICES.md"
PACKAGE_RUNNER_NAME = "submission_runner.py"
PACKAGE_UI_LAUNCHER_NAME = "启动工作台.bat"
ROOT_FILES = {
    "train.py",
    "test.py",
    "test_video.py",
    "feedback_ui.py",
    "requirements.txt",
    "COMPLETE_SYSTEM_DESIGN_CN.md",
    "MODEL_AND_DATA_PROVENANCE.md",
}

# Runtime source code is packaged, but an operator's local images, records,
# revision history, and activation pointers never form competition evidence.
_FEEDBACK_HISTORY_PREFIXES = (
    "outputs/feedback/",
    "feedback/history/",
    "feedback/images/",
    "feedback/revisions/",
    "feedback/candidates/",
)


def _is_feedback_history_arcname(value: str | Path) -> bool:
    normalized = PurePosixPath(str(value).replace("\\", "/")).as_posix().lstrip("./")
    return any(normalized.casefold().startswith(prefix) for prefix in _FEEDBACK_HISTORY_PREFIXES)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(payload: object) -> str:
    encoded = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode(
        "utf-8"
    )
    return _sha256(encoded)


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _project_relative(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            f"submission input is outside project root: {resolved}"
        ) from exc


def _eligible(path: Path) -> bool:
    return path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def _collect_tree(root: Path, *, suffixes: set[str] | None = None) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*")
        if _eligible(path) and (suffixes is None or path.suffix.lower() in suffixes)
    ]


def _load_yaml_mapping(path: Path, context: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a YAML mapping: {path}")
    return payload


def _resolve_packaged_reference(
    value: object,
    *,
    package_root: Path,
    base: Path,
) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        raise ValueError(f"package artifact reference must be relative: {raw}")
    root = package_root.resolve(strict=True)
    attempted: list[Path] = []
    for candidate in (base / raw, root / raw):
        resolved = candidate.resolve()
        attempted.append(resolved)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"package artifact escapes archive root: {raw}") from exc
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"package artifact not found for {raw}; tried: "
        + ", ".join(str(path) for path in attempted)
    )


def _resolve_packaged_dependency(
    value: object,
    *,
    package_root: Path,
    base: Path,
) -> Path:
    raw = Path(str(value))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"package dependency must be a contained relative path: {raw}")
    root = package_root.resolve(strict=True)
    attempted: list[Path] = []
    for candidate in (base / raw, root / raw):
        resolved = candidate.resolve()
        attempted.append(resolved)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"package dependency escapes archive root: {raw}") from exc
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"package dependency not found for {raw}; tried: "
        + ", ".join(str(path) for path in attempted)
    )


def _bundle_references(bundle: Mapping[str, Any]) -> dict[str, object]:
    models = bundle.get("models")
    if not isinstance(models, Mapping) or not models:
        raise TypeError("model bundle models must be a non-empty mapping")
    references: dict[str, object] = {
        "source_model": bundle.get("source_model"),
        "calibration": bundle.get("calibration"),
        "support_receipt": bundle.get("support_receipt"),
    }
    references.update({f"models.{name}": value for name, value in models.items()})
    return references


_PORTABLE_PATH_HINT = re.compile(r"assets[/\\][A-Za-z0-9_.-]+(?:[/\\][A-Za-z0-9_.-]+)*")
MAX_SERIALIZED_METADATA_BYTES = 16_000_000


def _serialized_path_hints(path: Path) -> set[str]:
    """Read portable asset strings from torch pickle metadata without importing torch."""

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                metadata = [
                    info
                    for info in archive.infolist()
                    if info.filename == "data.pkl"
                    or info.filename.endswith("/data.pkl")
                ]
                if (
                    len(metadata) != 1
                    or metadata[0].file_size > MAX_SERIALIZED_METADATA_BYTES
                ):
                    return set()
                payload = archive.read(metadata[0])
        else:
            if path.stat().st_size > MAX_SERIALIZED_METADATA_BYTES:
                return set()
            payload = path.read_bytes()
        values = [
            argument
            for _opcode, argument, _position in pickletools.genops(payload)
            if isinstance(argument, str)
        ]
    except (OSError, ValueError, zipfile.BadZipFile):
        return set()
    return {
        match.group(0).replace("\\", "/")
        for value in values
        for match in [_PORTABLE_PATH_HINT.fullmatch(value)]
        if match is not None
    }


def _bundle_dependency_files(
    bundle_path: Path,
    *,
    package_root: Path,
    allowed_files: set[str] | None = None,
) -> tuple[set[Path], list[dict[str, Any]]]:
    """Resolve the bundle's exact artifact and declared runtime-asset closure."""

    bundle = _load_yaml_mapping(bundle_path, "model bundle")
    root = package_root.resolve(strict=True)
    files = {bundle_path.resolve(strict=True)}
    for name, reference in _bundle_references(bundle).items():
        if reference is None or reference == "":
            raise ValueError(f"model bundle is missing artifact reference for {name}")
        artifact = _resolve_packaged_reference(
            reference,
            package_root=root,
            base=bundle_path.parent,
        )
        files.add(artifact)
        for hint in _serialized_path_hints(artifact):
            hinted = _resolve_packaged_dependency(
                hint,
                package_root=root,
                base=bundle_path.parent,
            )
            files.update(_collect_tree(hinted) if hinted.is_dir() else [hinted])

    raw_assets = bundle.get("assets", {})
    if not isinstance(raw_assets, Mapping):
        raise TypeError("model bundle assets must be a mapping when present")
    asset_receipts: list[dict[str, Any]] = []
    for name, raw_entry in raw_assets.items():
        if not isinstance(raw_entry, Mapping):
            raise TypeError(f"model bundle asset {name!r} must be a mapping")
        reference = raw_entry.get("path")
        if reference is None or reference == "":
            raise ValueError(f"model bundle asset {name!r} is missing path")
        dependency = _resolve_packaged_dependency(
            reference,
            package_root=root,
            base=bundle_path.parent,
        )
        dependency_files = (
            _collect_tree(dependency) if dependency.is_dir() else [dependency]
        )
        if not dependency_files:
            raise ValueError(f"model bundle asset {name!r} contains no package files")
        files.update(dependency_files)
        expected_hash = raw_entry.get("sha256")
        hash_verified = False
        if expected_hash not in {None, ""}:
            if dependency.is_dir():
                raise ValueError(
                    f"model bundle asset {name!r} records a file hash for a directory"
                )
            if not _is_sha256(expected_hash):
                raise ValueError(f"invalid recorded SHA-256 for bundle asset {name!r}")
            actual_hash = _sha256_path(dependency)
            if actual_hash != str(expected_hash).lower():
                raise ValueError(
                    f"bundle asset SHA-256 mismatch for {name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            hash_verified = True
        asset_receipts.append(
            {
                "name": str(name),
                "path": dependency.relative_to(root).as_posix(),
                "kind": "directory" if dependency.is_dir() else "file",
                "files": len(dependency_files),
                "declared_sha256_verified": hash_verified,
            }
        )

    for path in files:
        relative = path.resolve(strict=True).relative_to(root).as_posix()
        if allowed_files is not None and relative not in allowed_files:
            raise ValueError(
                f"bundle dependency is not covered by package manifest: {relative}"
            )
    return files, sorted(asset_receipts, key=lambda row: row["name"])


def _verify_bundle_integrity_at(
    bundle_path: Path,
    *,
    package_root: Path,
    selected_category: str,
    pointer_dataset: str | None,
    allowed_files: set[str] | None = None,
) -> dict[str, Any]:
    bundle = _load_yaml_mapping(bundle_path, "model bundle")
    target = str(bundle.get("target_object", "")).strip()
    if target != selected_category:
        raise ValueError(
            "selected pointer and model bundle target disagree: "
            f"pointer={selected_category!r}, bundle={target!r}"
        )
    target_dataset = str(bundle.get("target_dataset", "")).strip()
    if not target_dataset:
        raise ValueError("model bundle must record target_dataset")
    if pointer_dataset and target_dataset != pointer_dataset:
        raise ValueError(
            "selected pointer and model bundle dataset disagree: "
            f"pointer={pointer_dataset!r}, bundle={target_dataset!r}"
        )

    expected_bundle_hash = bundle.get("bundle_hash")
    if not _is_sha256(expected_bundle_hash):
        raise ValueError("model bundle must contain a valid bundle_hash")
    actual_bundle_hash = _payload_hash(
        {key: value for key, value in bundle.items() if key != "bundle_hash"}
    )
    if actual_bundle_hash != str(expected_bundle_hash).lower():
        raise ValueError(
            "model bundle metadata SHA-256 mismatch: "
            f"expected {expected_bundle_hash}, got {actual_bundle_hash}"
        )

    recorded = bundle.get("artifact_sha256")
    if not isinstance(recorded, Mapping):
        raise TypeError("model bundle artifact_sha256 must be a mapping")
    models = bundle.get("models")
    if not isinstance(models, Mapping) or not models:
        raise TypeError("model bundle models must be a non-empty mapping")
    recorded_models = recorded.get("models")
    if not isinstance(recorded_models, Mapping) or set(recorded_models) != set(models):
        raise ValueError("model bundle artifact hashes must cover every model exactly")

    references: dict[str, tuple[object, object]] = {
        "source_model": (bundle.get("source_model"), recorded.get("source_model")),
        "calibration": (bundle.get("calibration"), recorded.get("calibration")),
        "support_receipt": (
            bundle.get("support_receipt"),
            recorded.get("support_receipt"),
        ),
    }
    references.update(
        {
            f"models.{name}": (value, recorded_models[name])
            for name, value in models.items()
        }
    )

    verified: dict[str, str] = {}
    root = package_root.resolve(strict=True)
    for name, (reference, expected_hash) in references.items():
        if reference is None or reference == "":
            raise ValueError(f"model bundle is missing artifact reference for {name}")
        if not _is_sha256(expected_hash):
            raise ValueError(f"invalid recorded SHA-256 for {name}")
        path = _resolve_packaged_reference(
            reference,
            package_root=root,
            base=bundle_path.parent,
        )
        relative = path.relative_to(root).as_posix()
        if allowed_files is not None and relative not in allowed_files:
            raise ValueError(
                f"bundle artifact is not covered by package manifest: {relative}"
            )
        actual_hash = _sha256_path(path)
        if actual_hash != str(expected_hash).lower():
            raise ValueError(
                f"model artifact SHA-256 mismatch for {name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        verified[name] = actual_hash

    dependency_files, asset_receipts = _bundle_dependency_files(
        bundle_path,
        package_root=root,
        allowed_files=allowed_files,
    )
    # The artifact loop above verifies the files with recorded hashes.  The
    # dependency closure additionally verifies declared runtime assets (for
    # example the local DINOv2 repository) are actually packaged.
    dependency_arcnames = sorted(
        path.resolve(strict=True).relative_to(root).as_posix()
        for path in dependency_files
    )

    return {
        "bundle_hash_verified": True,
        "artifact_hashes_verified": True,
        "verified_artifacts": verified,
        "bundle_dependency_files": dependency_arcnames,
        "declared_runtime_assets_verified": asset_receipts,
        "target_dataset": target_dataset,
        "target_object": target,
    }


def _load_pointer(
    pointer_path: Path, category: str | None
) -> tuple[dict[str, Any], str, Path]:
    payload = _load_yaml_mapping(pointer_path, "active model pointer")
    if not isinstance(payload.get("models"), dict):
        raise TypeError("active model pointer must contain a models mapping")
    models = payload["models"]
    selected = category
    if not selected:
        if len(models) != 1:
            raise ValueError(
                "--category is required when the pointer has multiple models"
            )
        selected = str(next(iter(models)))
    if selected not in models:
        raise KeyError(
            f"category {selected!r} is unavailable; choose one of {sorted(models)}"
        )
    bundle_path = resolve_artifact(models[selected], base=pointer_path.parent)
    return payload, selected, bundle_path


def _package_pointer(
    source: Mapping[str, Any],
    *,
    selected: str,
    bundle_path: Path,
    pointer_path: Path,
) -> dict[str, Any]:
    relative_bundle = Path(os.path.relpath(bundle_path, pointer_path.parent)).as_posix()
    filtered = {
        key: source[key]
        for key in ("schema_version", "dataset", "protocol")
        if key in source
    }
    filtered.setdefault("schema_version", 1)
    filtered["models"] = {selected: relative_bundle}
    filtered["submission_selection"] = {
        "package_scope": "single_category",
        "category": selected,
        "source_pointer_sha256": _sha256_path(pointer_path),
        "pooled_pointer_metadata_removed": True,
    }
    return filtered


def _structured_evidence_categories(payload: object) -> set[str]:
    categories: set[str] = set()
    if isinstance(payload, Mapping):
        per_category = payload.get("per_category")
        if isinstance(per_category, Mapping):
            categories.update(str(value) for value in per_category)
        listed = payload.get("categories")
        if isinstance(listed, (list, tuple, set)):
            categories.update(
                str(value).strip() for value in listed if str(value).strip()
            )
        for key in ("category", "selected_category", "target_object"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                categories.add(value.strip())
    return categories


def _nested_value(payload: Mapping[str, Any], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _evidence_count(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("images", "sample_count", "total_images", "runs", "count"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            if value is not None:
                raise ValueError(
                    f"evidence count {key!r} must be a non-negative integer"
                )
            continue
        return {"field": key, "value": value}
    per_category = payload.get("per_category")
    if isinstance(per_category, Mapping) and len(per_category) == 1:
        row = next(iter(per_category.values()))
        if isinstance(row, Mapping):
            for key in ("images", "sample_count", "total_images", "count"):
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    if value is not None:
                        raise ValueError(
                            f"evidence count {key!r} must be a non-negative integer"
                        )
                    continue
                return {"field": f"per_category.*.{key}", "value": value}
    return None


def _validate_evidence_file(path: Path, selected: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(
            "submission evidence must be structured YAML/JSON with an explicit "
            f"category scope: {path}"
        )
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(
            "submission evidence root must be a mapping with explicit category "
            f"scope: {path}"
        )
    categories = _structured_evidence_categories(payload)
    if not categories:
        raise ValueError(
            f"unscoped evidence is forbidden in a single-category archive: {path}"
        )
    if categories != {selected}:
        raise ValueError(
            "pooled or multi-category evidence is forbidden in a single-category "
            f"archive: {path} records {sorted(categories)}, selected={selected!r}"
        )
    selection_scope = str(payload.get("selection_scope", "")).strip()
    if selection_scope.startswith("per_category_") and not (
        isinstance(payload.get("per_category"), Mapping)
        and set(str(key) for key in payload["per_category"]) == {selected}
    ):
        raise ValueError(
            f"per-category evidence must contain exactly the selected category: {path}"
        )

    role = "unspecified"
    for key in ("evaluation_role", "evidence_role", "claim_role", "report_type"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            role = value.strip()
            break
    if role == "unspecified":
        benchmark_scope = payload.get("benchmark_scope")
        if isinstance(benchmark_scope, str) and benchmark_scope.strip():
            role = benchmark_scope.strip()

    count = _evidence_count(payload)
    claim_value = payload.get("eligible_for_final_independent_claim")
    claim_source = "eligible_for_final_independent_claim"
    if claim_value is None:
        claim_value = _nested_value(payload, "performance", "claim_eligible")
        claim_source = "performance.claim_eligible"
    if claim_value is not None and not isinstance(claim_value, bool):
        raise ValueError(f"evidence claim eligibility must be boolean: {path}")
    claim_recorded = claim_value is not None
    claim_eligible = bool(claim_value) if claim_recorded else False
    if claim_eligible and claim_source == "eligible_for_final_independent_claim":
        if (
            role != "independent_test"
            or payload.get("fresh_independent_test_attested") is not True
            or payload.get("test_observed_before_freeze") is not False
        ):
            raise ValueError(
                "independent-test evidence cannot claim eligibility without a fresh "
                f"post-freeze attestation: {path}"
            )
    return {
        "format": "json" if suffix == ".json" else "yaml",
        "category": selected,
        "role": role,
        "count": count,
        "claim_eligible": claim_eligible,
        "claim_eligibility_recorded": claim_recorded,
        "claim_eligibility_field": claim_source if claim_recorded else None,
    }


def _submission_config(
    pointer_payload: Mapping[str, Any],
    *,
    selected: str,
    pointer_arcname: str,
    bundle_arcname: str,
    evidence_files: list[str],
) -> bytes:
    payload = {
        "schema_version": 2,
        "package_scope": "single_category",
        "dataset": pointer_payload.get("dataset", "unknown"),
        "protocol": pointer_payload.get("protocol", "unknown"),
        "category": selected,
        "active_models_pointer": pointer_arcname,
        "model_bundle": bundle_arcname,
        "datasets_included": False,
        "evidence_scope": "selected_category_only",
        "evidence_files": evidence_files,
        "execution": {
            "mode": "adapt_then_evaluate_or_pretrained_inference",
            "entry_point": PACKAGE_RUNNER_NAME,
            "commands": {
                "adapt": f"python {PACKAGE_RUNNER_NAME} adapt CUSTOM_TRAIN.yaml",
                "evaluate": f"python {PACKAGE_RUNNER_NAME} evaluate CUSTOM_TEST.yaml",
                "infer": f"python {PACKAGE_RUNNER_NAME} infer IMAGE_PATH --device auto",
                "feedback": f"python {PACKAGE_RUNNER_NAME} feedback status --device auto",
                "active_learning_query": (
                    f"python {PACKAGE_RUNNER_NAME} feedback query UNLABELED_DIR "
                    "--budget 20 --device auto"
                ),
            },
            "training_supported": True,
            "evaluation_supported": True,
            "feedback_supported": True,
            "active_learning_query_supported": True,
            "interactive_inference_feedback_enabled": True,
            "interactive_feedback_supported": True,
            "interactive_inference_feedback_default": True,
            "interactive_inference_no_feedback_supported": True,
            "private_evaluation_feedback_frozen": True,
            "feedback_enabled": True,
            "feedback_config": PACKAGE_FEEDBACK_CONFIG_NAME,
            "input": (
                "external 100-normal/30-defect support paths, an external labeled "
                "1000+ test manifest, or one RGB image path"
            ),
        },
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")


def _submission_training_configs(
    bundle_path: Path,
    *,
    selected: str,
) -> dict[str, bytes]:
    bundle = _load_yaml_mapping(bundle_path, "model bundle")
    raw_assets = bundle.get("assets")
    if not isinstance(raw_assets, Mapping):
        raise TypeError("model bundle assets are required for packaged adaptation")
    required = {
        "resnet18_weights",
        "dinov2_repository",
        "dinov2_weights",
        "source_model",
    }
    missing = sorted(required - set(raw_assets))
    if missing:
        raise ValueError(
            f"model bundle lacks source assets required for adaptation: {missing}"
        )
    source_assets: dict[str, str] = {}
    for name in sorted(required):
        entry = raw_assets[name]
        if not isinstance(entry, Mapping) or not str(entry.get("path", "")).strip():
            raise ValueError(f"model bundle adaptation asset {name!r} is invalid")
        source_assets[name] = Path(str(entry["path"])).as_posix()

    base = {
        "dataset": "custom",
        "data_root": "data/NOT_PACKAGED_USE_CUSTOM_MODE",
        "categories": [selected],
        "protocol": {
            "normal_support": 100,
            "defect_support": 30,
            "minimum_holdout_per_category": 0,
            "minimum_pooled_test": 0,
        },
        "run_name": "submission_custom_base",
        "candidate_id": "AOI-SUBMISSION-CUSTOM",
        "device": "auto",
        "seed": 20260808,
        "source_assets": source_assets,
        "head": {"epochs": 60, "learning_rate": 0.01},
        "output_root": "outputs/custom_tasks",
    }
    custom_train = {
        "schema_version": 1,
        "base_training_config": PACKAGE_TRAIN_CONFIG_NAME,
        "task_name": "replace_with_private_task_name",
        "category": "replace_with_private_category",
        "normal_dir": "EXTERNAL_INPUT/support/normal",
        "defect_dir": "EXTERNAL_INPUT/support/defect",
        "test_manifest": "EXTERNAL_INPUT/test_manifest.yaml",
        "minimum_test_images": 1000,
        "device": "auto",
        "seed": 20260808,
        "head": {"epochs": 60, "learning_rate": 0.01},
        "output_root": "outputs/custom_tasks/replace_with_private_task_name",
        "refuse_overwrite": True,
    }
    custom_test = {
        "schema_version": 1,
        "active_models_pointer": (
            "outputs/custom_tasks/replace_with_private_task_name/active_models.yaml"
        ),
        "category": "replace_with_private_category",
        "test_manifest": "EXTERNAL_INPUT/test_manifest.yaml",
        "device": "auto",
        "batch_size": 4,
        "minimum_total_images": 1000,
        "evaluation_role": "independent_test",
        "fresh_independent_test_attested": False,
        "test_observed_before_freeze": True,
        "output_root": (
            "outputs/custom_tasks/replace_with_private_task_name/test_runs"
        ),
        "timestamp_output": True,
        "refuse_overwrite": True,
    }

    def encode(payload: Mapping[str, Any]) -> bytes:
        return yaml.safe_dump(
            dict(payload), sort_keys=False, allow_unicode=True
        ).encode("utf-8")

    return {
        PACKAGE_TRAIN_CONFIG_NAME: encode(base),
        # The desktop UI loads this exact path when it starts a new product.
        # Keep it byte-identical to the package base so GUI and CLI adaptation
        # use the same source assets and immutable 100+30 protocol.
        PACKAGE_UI_TRAIN_CONFIG_NAME: encode(base),
        PACKAGE_CUSTOM_TRAIN_NAME: encode(custom_train),
        PACKAGE_CUSTOM_TEST_NAME: encode(custom_test),
    }


def _submission_benchmark_configs(
    *, selected: str, pointer_arcname: str
) -> dict[str, bytes]:
    """Create package-local P50 benchmark templates for every UI mode."""

    common = {
        "schema_version": 2,
        "active_models_pointer": pointer_arcname,
        "category": selected,
        "image_dir": "EXTERNAL_INPUT/benchmark_images_2500x2500",
        "warmup": 20,
        "runs": 100,
        "timing_scope": (
            "model/runtime inference only; disk decode, 2500x2500 preparation, "
            "model loading, and output encoding excluded"
        ),
    }
    payloads = {
        PACKAGE_BENCHMARK_RTX2060_NAME: {
            **common,
            "benchmark_standard": "official_rtx2060",
            "device": "cuda:0",
            "output": "outputs/benchmark_official_rtx2060_p50_v3.yaml",
            "measurement_note": (
                "Competition requirement: 2500x2500 single-image runtime strictly "
                "below 200 ms on NVIDIA RTX 2060 or lower. The official text does "
                "not define a percentile; this project records synchronized P50."
            ),
        },
        PACKAGE_BENCHMARK_RTX4070_NAME: {
            **common,
            "benchmark_standard": "internal_rtx4070",
            "device": "cuda:0",
            "allow_nonexclusive_measurement": True,
            "output": "outputs/benchmark_internal_rtx4070_p50_v3.yaml",
            "measurement_note": (
                "Internal RTX 4070 diagnostic only; it cannot establish official "
                "RTX 2060 compliance."
            ),
        },
        PACKAGE_BENCHMARK_CPU_NAME: {
            **common,
            "benchmark_standard": "cpu_diagnostic",
            "device": "cpu",
            "allow_nonexclusive_measurement": False,
            "output": "outputs/benchmark_cpu_diagnostic_p50_v3.yaml",
            "measurement_note": (
                "CPU challenge diagnostic only; P50 must be strictly below 2000 ms "
                "and is not eligible for a competition GPU acceptance claim."
            ),
        },
    }
    return {
        name: yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True
        ).encode("utf-8")
        for name, payload in payloads.items()
    }


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_PRIVATE_WINDOWS_HOME_BYTES = re.compile(
    rb"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE
)


def _portable_submission_value(value: Any) -> Any:
    """Replace workstation-specific absolute paths in distributable receipts."""

    if isinstance(value, Mapping):
        return {key: _portable_submission_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_submission_value(item) for item in value]
    if not isinstance(value, str) or not _WINDOWS_ABSOLUTE_PATH.match(value):
        return value

    normalized = value.replace("\\", "/")
    project = PROJECT_ROOT.resolve().as_posix().rstrip("/")
    if normalized.casefold().startswith(project.casefold() + "/"):
        return normalized[len(project) + 1 :]
    return f"EXTERNAL_INPUT/{PureWindowsPath(value).name}"


def _submission_bundle_overrides(bundle_path: Path) -> dict[str, bytes]:
    """Return portable package-only bundle/receipt bytes when paths need redaction."""

    bundle = _load_yaml_mapping(bundle_path, "model bundle")
    receipt_path = _resolve_packaged_reference(
        bundle.get("support_receipt"),
        package_root=PROJECT_ROOT,
        base=bundle_path.parent,
    )
    receipt = _load_yaml_mapping(receipt_path, "support receipt")
    portable_receipt = _portable_submission_value(receipt)
    if portable_receipt == receipt:
        return {}
    portable_receipt["submission_portability"] = {
        "absolute_workstation_paths_removed": True,
        "path_policy": (
            "project-contained paths are project-relative; external paths retain "
            "only their basename"
        ),
    }
    receipt_bytes = yaml.safe_dump(
        portable_receipt, sort_keys=False, allow_unicode=True
    ).encode("utf-8")

    recorded = bundle.get("artifact_sha256")
    if not isinstance(recorded, dict):
        raise TypeError("model bundle artifact_sha256 must be a mutable mapping")
    recorded["support_receipt"] = _sha256(receipt_bytes)
    bundle["bundle_hash"] = _payload_hash(
        {key: value for key, value in bundle.items() if key != "bundle_hash"}
    )
    bundle_bytes = yaml.safe_dump(
        bundle, sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    return {
        _project_relative(receipt_path).as_posix(): receipt_bytes,
        _project_relative(bundle_path).as_posix(): bundle_bytes,
    }


def _submission_feedback_config(
    *, selected: str, pointer_arcname: str
) -> bytes:
    """Create the package-local operator-feedback configuration template."""
    payload = {
        "schema_version": 1,
        "feedback_root": "outputs/feedback",
        "active_models_pointer": pointer_arcname,
        "category": selected,
        "device": "auto",
        "minimum_recall": 0.90,
        "minimum_training_samples": 6,
        "minimum_validation_samples": 4,
        "epochs": 30,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "l2_sp": 0.001,
        "seed": 20260830,
        "query_budget": 20,
        "active_learning": {
            "weights": {
                "uncertainty": 0.55,
                "branch_disagreement": 0.20,
                "rarity": 0.25,
            }
        },
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")


def _submission_readme(
    *, selected: str, pointer_arcname: str, bundle_arcname: str
) -> bytes:
    text = f"""AOI COMPETITION CODE PACKAGE / AOI 竞赛代码包
================================================

Please read README_模型使用说明.md first.
请先阅读 README_模型使用说明.md。

Selected development seed: {selected}
Active pointer: {pointer_arcname}
Model bundle: {bundle_arcname}
Task contract: {PACKAGE_CONFIG_NAME}
Unified entry point: {PACKAGE_RUNNER_NAME}

Quick start / 快速启动:
  python -m venv .venv
  .venv\\Scripts\\python.exe -m pip install -r requirements.txt
  启动工作台.bat
  .venv\\Scripts\\python.exe {PACKAGE_RUNNER_NAME} --help

Integrity verification / 完整性校验:
  python -m aoi.tools.build_submission --verify PATH_TO_ARCHIVE.zip

No public/private dataset or operator feedback history is included.
本包不包含公开/私有数据集，也不包含操作员反馈历史。
"""
    return text.encode("utf-8")


def _submission_model_guide(
    *, selected: str, pointer_arcname: str, bundle_arcname: str
) -> bytes:
    text = f"""# AOI 质量检测模型使用说明

## 1. 提交包定位

本 ZIP 是赛题一“可自学习的 AOI 实时在线 AI 质检”的代码辅助材料，包含完整 Python
源码、桌面工作台、单图/视频入口、100 正常 + 30 缺陷适配入口、1000+ 带标签测试入口、
操作员反馈闭环、P50 性能测试配置和 SHA-256 完整性清单。

- 包内预置的 `{selected}` 是公开开发/演示种子模型，不是赛方隐藏测试成绩。
- 预置指针：`{pointer_arcname}`
- 预置模型包：`{bundle_arcname}`
- 包内不含 DAGM、MVTec、私有测试图像和历史反馈记录。
- 赛方数据到位后，必须先用其 100 张无缺陷 + 30 张有缺陷样本完成新任务适配，
  再对未在冻结前观察的 1000+ 测试集评估。

## 2. 环境

已核验环境：Windows 10、Python 3.10、PyTorch 2.5.1、torchvision 0.20.1。CPU 可用于功能调试；
RTX 2060 验收必须在对应 NVIDIA GPU 上实测。

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\Activate.ps1
```

`torch`/`torchvision` 的 CUDA 构建需与验收机驱动匹配。正式离线评测前应预先安装依赖；
赛题文件没有承诺评测环境可以联网。Windows Python 还需包含 Tcl/Tk，才能启动桌面 UI。
所有命令都应在 ZIP 解压根目录执行，且该目录需可写（用于产生 `outputs/`、报告和反馈记录）。
下文的裸 `python` 命令均假定已执行上述 `Activate.ps1`；如未激活，请将 `python`
替换为 `.\\.venv\\Scripts\\python.exe`。双击启动器会自动优先使用包内 `.venv`。

包内 `tests/` 仅用于可选开发自检，不是运行依赖。如需执行：

```powershell
python -m pip install pytest
python -m pytest -q
```

## 3. 桌面工作台

双击 `启动工作台.bat`，或执行：

```powershell
python feedback_ui.py
```

工作台默认打开“少样本适配”。新产品的建议流程为：

1. 输入产品/任务名称。
2. 选择恰好 100 张正常图和 30 张缺陷图。
3. 通常使用“图像标签”模式；只有需要像素监督时才选 30 张与缺陷图一一对应的掩码。
4. 可选加载 1000+ 带标签评估清单，点击“开始适配”。
5. 适配后使用“在线检测”，对误检/漏检提交操作员反馈；候选修订经独立验证和审核后才可启用。

UI 创建任务所需的基础配置已包含在 `configs/custom_training_base.yaml`。

## 4. 命令行适配和 1000+ 评估

先复制模板，不要直接修改 ZIP 内的原始文件：

```powershell
Copy-Item configs/custom_train.example.yaml CUSTOM_TRAIN.yaml
Copy-Item configs/custom_test.example.yaml CUSTOM_TEST.yaml
```

编辑 `CUSTOM_TRAIN.yaml` 的任务名、类别、外部正常/缺陷目录和评估清单后执行：

```powershell
python {PACKAGE_RUNNER_NAME} adapt CUSTOM_TRAIN.yaml
```

适配成功后，将 `CUSTOM_TEST.yaml` 的 `active_models_pointer`、`category`和 `test_manifest`
改为实际任务值，然后执行：

```powershell
python {PACKAGE_RUNNER_NAME} evaluate CUSTOM_TEST.yaml
```

正式声明要求：`minimum_total_images >= 1000`、模型/阈值已冻结、测试集在冻结前未观察，
且 `fresh_independent_test_attested: true`、`test_observed_before_freeze: false`。程序会检查支持集与测试图的 SHA-256 重复。

### 评估清单最小示例

```yaml
schema_version: 1
image_root: images
samples:
  - id: sample-000001
    image: normal/000001.png
    label: 0
  - id: sample-000002
    image: defect/000002.png
    label: 1
    defect_type: appearance
```

正式清单必须同时含正常和缺陷标签，且图像内容不重复。

## 5. 图片、视频和摄像头

使用预置公开开发模型做单图功能检查：

```powershell
python {PACKAGE_RUNNER_NAME} infer IMAGE_PATH --device auto
python {PACKAGE_RUNNER_NAME} infer IMAGE_PATH --device auto --no-feedback
```

视频文件：

```powershell
python test_video.py VIDEO_PATH --category {selected} --models {pointer_arcname} --device auto --output outputs/video_result.mp4
```

摄像头（`0` 为默认摄像头索引）：

```powershell
python test_video.py 0 --category {selected} --models {pointer_arcname} --device auto --display
```

## 6. 操作员反馈闭环

```powershell
python {PACKAGE_RUNNER_NAME} feedback record IMAGE_PATH --label 1 --roi LEFT TOP RIGHT BOTTOM
python {PACKAGE_RUNNER_NAME} feedback query UNLABELED_DIR --budget 20 --device auto
python {PACKAGE_RUNNER_NAME} feedback propose --device auto
python {PACKAGE_RUNNER_NAME} feedback status --device auto
```

反馈候选不会自动覆盖冻结模型。正式 1000+ 批量评估永久禁用在线反馈修订，避免测试泄漏。

## 7. 2500×2500 性能测试

将下列配置中的 `image_dir` 改为外部测试图目录：

- `configs/benchmark_official_rtx2060.yaml`：RTX 2060 官方硬件线，赛题原文要求运行时间严格小于 200 ms。
- `configs/benchmark_internal_rtx4070.yaml`：RTX 4070 内部诊断，不能代替 2060 证据。
- `configs/benchmark_cpu_diagnostic.yaml`：CPU P50 严格小于 2 秒的挑战目标，不是基础 GPU 验收声明。

```powershell
python test.py benchmark --config configs/benchmark_official_rtx2060.yaml
```

赛题原文未规定延迟百分位；本项目使用 20 次预热 + 100 次同步单图计时的 P50 作为可重复工程协议。

## 8. 输出与证据边界

- 单图输出为 YAML，含预测、分数、阈值和反馈修订状态。
- 视频输出为标注 MP4 及同名 YAML 摘要。
- 1000+ 评估生成 `metrics.yaml`、`predictions.yaml`和配置/代码/模型哈希证据。
- `SUBMISSION_MANIFEST.yaml` 记录 ZIP 中除自身外每个文件的字节数和 SHA-256。
- 公开 DAGM/MVTec 开发结果、RTX 4070 诊断或 CPU 诊断不能改写为赛方隐藏测试或 RTX 2060 官方成绩。

## 9. 完整性校验与已知限制

对原始 ZIP 执行：

```powershell
python -m aoi.tools.build_submission --verify PATH_TO_ARCHIVE.zip
```

校验包括压缩包路径安全、逐文件 SHA-256、单类别指针、模型包哈希和权重哈希。
完整设计见 `COMPLETE_SYSTEM_DESIGN_CN.md`，模型/数据来源及许可状态见 `MODEL_AND_DATA_PROVENANCE.md`。

正式提交前仍需人工闭合：

1. 用真实团队名称替换 ZIP 文件名中的“待填写团队名称”。
2. 在真实 RTX 2060 或赛方认可的更低 GPU 上复测 2500×2500、严格 `<200 ms`。
3. 使用冻结后首次观察的真实私有 1000+ 测试集生成最终成绩。
4. 复核 `MODEL_AND_DATA_PROVENANCE.md` 中尚未闭合的 `source_model.pt` 记忆库来源/许可回执。
"""
    return text.encode("utf-8")


def _submission_ui_launcher() -> bytes:
    return (
        "@echo off\r\n"
        "setlocal\r\n"
        "cd /d \"%~dp0\"\r\n"
        "set AOI_FEEDBACK_UI_RELAUNCHED=1\r\n"
        "if exist \".venv\\Scripts\\python.exe\" (\r\n"
        "  set \"AOI_PYTHON=.venv\\Scripts\\python.exe\"\r\n"
        ") else (\r\n"
        "  where python >nul 2>nul\r\n"
        "  if errorlevel 1 (\r\n"
        "    echo Python was not found. Install Python 3.10 and requirements.txt first.\r\n"
        "    pause\r\n"
        "    exit /b 1\r\n"
        "  )\r\n"
        "  set \"AOI_PYTHON=python\"\r\n"
        ")\r\n"
        "\"%AOI_PYTHON%\" feedback_ui.py\r\n"
        "if errorlevel 1 pause\r\n"
    ).encode("utf-8")


def _submission_third_party_notices() -> bytes:
    return """# 第三方软件、权重与数据声明

1. 本包含 DINOv2 运行时源码和 ViT-S/14 权重。对应的上游 `LICENSE`、
   `MODEL_CARD.md` 和 `COMMIT_SHA.txt` 位于 `assets/dinov2/`，提交方保留其原始声明。
2. `assets/resnet18-f37072fd.pth` 为 torchvision 发布的 ResNet18 ImageNet-1K V1 权重。
   torchvision 代码许可与 ImageNet 数据/权重使用条款不完全等同；正式提交前由团队
   和赛方复核权重再分发范围。
3. `assets/source_model.pt` 的原始记忆库样本与生成回执未在当前工作区中闭合。
   不得将其表述为“来源已完全核验”。详见 `MODEL_AND_DATA_PROVENANCE.md`。
4. 本包不再分发 DAGM2007、MVTec AD、赛方私有数据或操作员反馈图像。
5. Python 运行时依赖及其各自许可以 `requirements.txt` 和安装环境中的上游发行包为准。

本文仅记录当前可核验状态，不替代赛方、数据发布方或模型发布方的法律条款。
""".encode("utf-8")


def _submission_runner() -> bytes:
    """Return the extracted-package adaptation/evaluation/inference wrapper."""

    text = '''#!/usr/bin/env python3
"""Run private adaptation/evaluation or pretrained inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "configs" / "submission_task.yaml"


def _load_runtime(device: str):
    from aoi.paths import resolve_artifact
    from aoi.runtime.config import resolve_device
    from aoi.runtime.runtime import AdaptedRuntime

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8")) or {}
    category = str(contract["category"])
    pointer = (ROOT / str(contract["active_models_pointer"])).resolve(strict=True)
    pointer_payload = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
    models = pointer_payload.get("models", {})
    if set(models) != {category}:
        raise ValueError("packaged pointer does not contain exactly its selected category")
    bundle = resolve_artifact(models[category], base=pointer.parent)
    runtime = AdaptedRuntime(bundle, device=resolve_device(device))
    return contract, category, bundle, runtime


def _feedback_state(runtime, category: str, root: Path, enabled: bool):
    if not enabled:
        return None
    from aoi.feedback import FeedbackStore, apply_active_feedback

    return apply_active_feedback(runtime, FeedbackStore(root), category, enabled=True)


def _infer(image: Path, device: str, *, feedback: bool, feedback_root: Path) -> int:
    from aoi.runtime.data import read_rgb

    _contract, category, _bundle, runtime = _load_runtime(device)
    state = _feedback_state(runtime, category, feedback_root, feedback)
    image = image.resolve(strict=True)
    result = runtime.predict(read_rgb(image), f"{category}/{image.name}")
    print(yaml.safe_dump({
        "category": category,
        "image": str(image),
        "feedback_enabled": bool(feedback),
        "feedback_state": None if state is None else state.receipt(),
        **result,
    }, sort_keys=False, allow_unicode=True))
    return 0


def _feedback_operation(args) -> int:
    from aoi.feedback import FeedbackStore
    from aoi.feedback.active_learning import query_unlabeled_pool
    from aoi.feedback.optimization import _bundle_hash, propose_adaptive_revision
    from aoi.runtime.data import read_rgb

    _contract, category, _bundle, runtime = _load_runtime(args.device)
    root = args.feedback_root.resolve()
    store = FeedbackStore(root)
    run_id = str(getattr(runtime, "bundle", {}).get("run_id", "")).strip()
    if not run_id:
        run_id = Path(getattr(runtime, "bundle_path", _bundle)).parent.name
    bundle_hash = _bundle_hash(runtime)
    if args.feedback_command == "query":
        _feedback_state(runtime, category, root, True)
        result = query_unlabeled_pool(
            runtime, store, category, args.input_dir,
            budget=int(args.budget), model_version=run_id,
        )
        print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
        return 0
    if args.feedback_command == "status":
        active = store.active_revision(category, run_id, bundle_hash, verify=True)
        print(yaml.safe_dump({
            "category": category, "run_id": run_id, "bundle_hash": bundle_hash,
            "feedback_enabled": active is not None, "active_revision": active,
        }, sort_keys=False, allow_unicode=True))
        return 0
    if args.feedback_command == "propose":
        candidate = propose_adaptive_revision(
            runtime, store, category, run_id=run_id, bundle_hash=bundle_hash,
            minimum_recall=float(args.minimum_recall),
        )
        print(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True))
        return 0
    if args.feedback_command == "activate":
        entry = store.activate_revision(
            args.candidate, category=category, run_id=run_id, bundle_hash=bundle_hash
        )
        print(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))
        return 0
    if args.feedback_command == "rollback":
        entry = store.rollback_revision(category, run_id, bundle_hash)
        print(yaml.safe_dump({"restored": entry}, sort_keys=False, allow_unicode=True))
        return 0
    if args.feedback_command == "record":
        image = args.image.resolve(strict=True)
        state = _feedback_state(runtime, category, root, not args.no_feedback)
        rgb = read_rgb(image)
        result = runtime.predict(rgb, f"{category}/{image.name}")
        image_bytes = image.read_bytes()
        shape = getattr(rgb, "shape", None)
        image_size = None if shape is None or len(shape) < 2 else (int(shape[1]), int(shape[0]))
        receipt = store.save_feedback(
            image_bytes=image_bytes,
            original_name=image.name,
            category=category,
            model_version=run_id,
            result=result,
            operator_label=int(args.label),
            defect_type=args.defect_type,
            note=args.note,
            dataset_role=args.role,
            defect_region=args.roi,
            image_size=image_size,
            run_id=run_id,
            bundle_hash=None if state is None else state.bundle_hash,
            adaptive_revision_id=None if state is None else state.revision_id,
            head_version=None if state is None else state.head_version,
            feature_version=None if state is None else state.feature_version,
            threshold_version=None if state is None else state.threshold_version,
        )
        print(yaml.safe_dump(receipt, sort_keys=False, allow_unicode=True))
        return 0
    raise AssertionError(args.feedback_command)


def _run_configured(command: str, config: Path) -> int:
    config = config.resolve(strict=True)
    if command == "adapt":
        from aoi.cli.custom_training import run
    elif command == "evaluate":
        payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError("private evaluation config must be a mapping")
        if payload.get("feedback_enabled") not in {None, False}:
            raise ValueError("private batch evaluation is permanently feedback-frozen")
        if payload.get("feedback") not in (None, False) and payload.get("feedback") != {}:
            raise ValueError("private batch evaluation is permanently feedback-frozen")
        from aoi.cli.custom_evaluation import run
    else:
        raise AssertionError(command)
    return int(run(config))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run this AOI submission package.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="classify one RGB image")
    infer.add_argument("image", type=Path)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--feedback", action="store_true", help="apply approved interactive feedback revision (default)")
    infer.add_argument("--no-feedback", dest="feedback", action="store_false", help="force the frozen bundle")
    infer.set_defaults(feedback=True)
    adapt = subparsers.add_parser("adapt", help="adapt from exactly 100 normal + 30 defect")
    adapt.add_argument("config", type=Path)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a frozen task on 1000+ labels")
    evaluate.add_argument("config", type=Path)
    feedback = subparsers.add_parser("feedback", help="record or manage interactive feedback revisions")
    feedback_subparsers = feedback.add_subparsers(dest="feedback_command", required=True)
    for name in ("status", "propose", "rollback"):
        item = feedback_subparsers.add_parser(name)
        item.add_argument("--device", default="auto")
        item.add_argument("--feedback-root", type=Path, default=ROOT / "outputs" / "feedback")
        if name == "propose":
            item.add_argument("--minimum-recall", type=float, default=0.90)
    query = feedback_subparsers.add_parser("query")
    query.add_argument("input_dir", type=Path)
    query.add_argument("--budget", type=int, default=20)
    query.add_argument("--device", default="auto")
    query.add_argument("--feedback-root", type=Path, default=ROOT / "outputs" / "feedback")
    activate = feedback_subparsers.add_parser("activate")
    activate.add_argument("candidate", help="revision ID or candidate receipt path")
    activate.add_argument("--device", default="auto")
    activate.add_argument("--feedback-root", type=Path, default=ROOT / "outputs" / "feedback")
    record = feedback_subparsers.add_parser("record")
    record.add_argument("image", type=Path)
    record.add_argument("--label", type=int, choices=(0, 1), required=True)
    record.add_argument("--role", choices=("training", "validation"), default="training")
    record.add_argument("--roi", type=int, nargs=4)
    record.add_argument("--defect-type", default="appearance")
    record.add_argument("--note", default="")
    record.add_argument("--device", default="auto")
    record.add_argument("--feedback-root", type=Path, default=ROOT / "outputs" / "feedback")
    record.add_argument("--no-feedback", action="store_true", help="record against the frozen bundle")
    args = parser.parse_args()
    if args.command == "infer":
        return _infer(args.image, args.device, feedback=args.feedback, feedback_root=ROOT / "outputs" / "feedback")
    if args.command == "feedback":
        return _feedback_operation(args)
    if args.command in {"adapt", "evaluate"}:
        return _run_configured(args.command, args.config)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return text.encode("utf-8")


def _receipt(path: str, value: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(value), "sha256": _sha256(value)}


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    if left_resolved == right_resolved:
        return True
    if left_resolved.exists() and right_resolved.exists():
        try:
            if os.path.samefile(left_resolved, right_resolved):
                return True
        except OSError:
            pass
    return False


def _safe_extract_target(root: Path, name: str) -> Path:
    if "\\" in name:
        raise ValueError(f"archive member must use POSIX separators: {name!r}")
    if "\x00" in name or len(name) > MAX_MEMBER_PATH_LENGTH:
        raise ValueError(f"unsafe or overlong archive member path: {name!r}")
    pure = PurePosixPath(name)
    if not name or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if any(":" in part for part in pure.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    raw_parts = name.split("/")
    if len(pure.parts) > MAX_MEMBER_DEPTH or any(
        part in {"", "."} or part.rstrip(" .") != part for part in raw_parts
    ):
        raise ValueError(f"unsafe or overdeep archive member path: {name!r}")
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
    }
    if any(part.split(".", 1)[0].casefold() in reserved for part in raw_parts):
        raise ValueError(f"archive member uses a reserved Windows path: {name!r}")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"archive member escapes extraction root: {name!r}") from exc
    return target


def _extract_archive_safely(
    archive: zipfile.ZipFile,
    extraction_root: Path,
    *,
    maximum_uncompressed_bytes: int,
) -> list[str]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(
            f"submission archive has {len(infos)} members, exceeding "
            f"the {MAX_ARCHIVE_MEMBERS}-member safety limit"
        )
    names = [info.filename for info in infos]
    leaked_feedback_history = [
        name for name in names if _is_feedback_history_arcname(name)
    ]
    if leaked_feedback_history:
        raise ValueError(
            "submission archive contains forbidden operator feedback history: "
            + ", ".join(leaked_feedback_history[:3])
        )
    normalized = [PurePosixPath(name).as_posix().casefold() for name in names]
    if len(names) != len(set(normalized)):
        raise ValueError(
            "submission archive contains duplicate or case-colliding member names"
        )

    targets: dict[str, Path] = {}
    total_uncompressed = 0
    total_compressed = 0
    for info in infos:
        if info.is_dir():
            raise ValueError(
                "submission archive must contain files, not directory entries"
            )
        if info.flag_bits & 0x1:
            raise ValueError(
                f"submission archive contains an encrypted member: {info.filename!r}"
            )
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"submission archive contains a symlink member: {info.filename!r}"
            )
        target = _safe_extract_target(extraction_root, info.filename)
        targets[PurePosixPath(info.filename).as_posix().casefold()] = target
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"archive member {info.filename!r} exceeds the "
                f"{MAX_MEMBER_UNCOMPRESSED_BYTES}-byte member limit"
            )
        total_uncompressed += max(0, info.file_size)
        total_compressed += max(0, info.compress_size)
        if total_uncompressed > maximum_uncompressed_bytes:
            raise ValueError(
                "submission archive uncompressed size exceeds the safety limit"
            )
        if info.file_size and not info.compress_size:
            raise ValueError(
                f"archive member {info.filename!r} has no compressed payload"
            )
        if info.file_size and info.compress_size:
            ratio = info.file_size / info.compress_size
            if ratio > MAX_MEMBER_COMPRESSION_RATIO:
                raise ValueError(
                    f"archive member {info.filename!r} has an unsafe compression ratio "
                    f"({ratio:.1f}:1)"
                )

    # A file cannot also be an ancestor of another member.  Checking this
    # before creating directories prevents extraction-order tricks.
    target_names = set(targets)
    for name in target_names:
        parts = name.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in target_names:
                raise ValueError(
                    f"archive member path collides with a parent: {name!r}"
                )
    if total_uncompressed and total_compressed:
        total_ratio = total_uncompressed / total_compressed
        if total_ratio > MAX_TOTAL_COMPRESSION_RATIO:
            raise ValueError(
                f"archive has an unsafe aggregate compression ratio ({total_ratio:.1f}:1)"
            )

    for info in infos:
        target = targets[PurePosixPath(info.filename).as_posix().casefold()]
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with archive.open(info, "r") as source, target.open("xb") as destination:
                while True:
                    block = source.read(min(1024 * 1024, info.file_size - written))
                    if not block:
                        break
                    destination.write(block)
                    written += len(block)
                    if (
                        written > info.file_size
                        or written > MAX_MEMBER_UNCOMPRESSED_BYTES
                    ):
                        raise ValueError(
                            f"archive member {info.filename!r} exceeded its declared size"
                        )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        if written != info.file_size:
            raise ValueError(
                f"archive member {info.filename!r} ended at {written} bytes, "
                f"expected {info.file_size}"
            )
    return names


def verify_submission_archive(
    archive_path: str | Path,
    *,
    maximum_bytes: int = MAX_SUBMISSION_BYTES,
    maximum_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    """Extract and verify every packaged byte plus the selected model bundle."""

    if maximum_bytes <= 0 or maximum_uncompressed_bytes <= 0:
        raise ValueError("archive safety limits must be positive")
    archive_path = Path(archive_path).resolve(strict=True)
    archive_size = archive_path.stat().st_size
    if archive_size > maximum_bytes:
        raise ValueError(
            f"submission archive is {archive_size} bytes, exceeding the "
            f"{maximum_bytes}-byte verification limit"
        )
    with tempfile.TemporaryDirectory(prefix="aoi_submission_verify_") as raw_root:
        extraction_root = Path(raw_root).resolve()
        with zipfile.ZipFile(archive_path) as archive:
            names = _extract_archive_safely(
                archive,
                extraction_root,
                maximum_uncompressed_bytes=maximum_uncompressed_bytes,
            )

        manifest_path = extraction_root / MANIFEST_NAME
        manifest = _load_yaml_mapping(manifest_path, "submission manifest")
        if manifest.get("schema_version") != 3:
            raise ValueError("submission manifest schema_version must be 3")
        declared_maximum = manifest.get("maximum_archive_bytes")
        if (
            isinstance(declared_maximum, bool)
            or not isinstance(declared_maximum, int)
            or declared_maximum <= 0
            or declared_maximum > maximum_bytes
        ):
            raise ValueError("submission manifest maximum_archive_bytes is invalid")
        if archive_size > declared_maximum:
            raise ValueError(
                f"submission archive is {archive_size} bytes, exceeding its declared "
                f"{declared_maximum}-byte limit"
            )
        receipts = manifest.get("files")
        if not isinstance(receipts, list):
            raise TypeError("submission manifest files must be a list")
        receipt_by_path: dict[str, Mapping[str, Any]] = {}
        for row in receipts:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                raise TypeError("submission manifest file receipt is invalid")
            name = str(row["path"])
            if name in receipt_by_path:
                raise ValueError(f"duplicate package receipt: {name}")
            receipt_by_path[name] = row

        expected_names = set(receipt_by_path) | {MANIFEST_NAME}
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            unexpected = sorted(set(names) - expected_names)
            raise ValueError(
                f"archive/manifest file-set mismatch; missing={missing}, unexpected={unexpected}"
            )
        if any(name == "data" or name.startswith("data/") for name in receipt_by_path):
            raise ValueError("submission archive must not contain dataset files")
        for name, row in receipt_by_path.items():
            path = _safe_extract_target(extraction_root, name)
            if not path.is_file():
                raise FileNotFoundError(f"manifested package file is missing: {name}")
            expected_hash = row.get("sha256")
            if not _is_sha256(expected_hash):
                raise ValueError(f"invalid package SHA-256 receipt: {name}")
            if path.stat().st_size != int(row.get("bytes", -1)):
                raise ValueError(f"package byte-count mismatch: {name}")
            actual_hash = _sha256_path(path)
            if actual_hash != str(expected_hash).lower():
                raise ValueError(
                    f"package SHA-256 mismatch for {name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            if (
                path.suffix.casefold()
                in {".yaml", ".yml", ".json", ".txt", ".md"}
                and _PRIVATE_WINDOWS_HOME_BYTES.search(path.read_bytes())
            ):
                raise ValueError(
                    f"submission text contains a private Windows home path: {name}"
                )

        selected = str(manifest.get("selected_category", "")).strip()
        pointer_arcname = str(manifest.get("selected_pointer", ""))
        bundle_arcname = str(manifest.get("model_bundle", ""))
        if not selected or pointer_arcname not in receipt_by_path:
            raise ValueError("submission manifest is missing its selected pointer")
        if bundle_arcname not in receipt_by_path:
            raise ValueError("submission manifest is missing its selected model bundle")

        pointer_path = _safe_extract_target(extraction_root, pointer_arcname)
        pointer = _load_yaml_mapping(pointer_path, "packaged active model pointer")
        models = pointer.get("models")
        if not isinstance(models, Mapping) or set(models) != {selected}:
            raise ValueError(
                "packaged pointer must contain exactly the selected category"
            )
        selection = pointer.get("submission_selection")
        if not isinstance(selection, Mapping) or selection.get("category") != selected:
            raise ValueError("packaged pointer submission_selection is inconsistent")
        resolved_bundle = _resolve_packaged_reference(
            models[selected],
            package_root=extraction_root,
            base=pointer_path.parent,
        )
        expected_bundle = _safe_extract_target(extraction_root, bundle_arcname)
        if resolved_bundle != expected_bundle:
            raise ValueError(
                "packaged pointer does not resolve to the manifested model bundle"
            )

        configuration = manifest.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("submission manifest configuration must be a mapping")
        generated_configs = configuration.get("generated_files")
        if generated_configs != PACKAGE_CONFIG_NAMES:
            raise ValueError(
                "submission must contain its exact generated task/adaptation configs"
            )
        packaged_configs = sorted(
            name for name in receipt_by_path if name.startswith("configs/")
        )
        if packaged_configs != sorted(PACKAGE_CONFIG_NAMES):
            raise ValueError(
                "pooled or repository configs leaked into submission archive"
            )
        config_path = _safe_extract_target(extraction_root, PACKAGE_CONFIG_NAME)
        config = _load_yaml_mapping(config_path, "submission task config")
        expected_config = {
            "package_scope": "single_category",
            "category": selected,
            "active_models_pointer": pointer_arcname,
            "model_bundle": bundle_arcname,
            "evidence_scope": "selected_category_only",
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise ValueError(
                    f"submission task config mismatch for {key}: "
                    f"expected {expected!r}, got {config.get(key)!r}"
                )
        if (
            config.get("schema_version") != 2
            or config.get("datasets_included") is not False
        ):
            raise ValueError(
                "submission task config must declare schema 2 and no datasets"
            )
        execution = config.get("execution")
        expected_execution = {
            "mode": "adapt_then_evaluate_or_pretrained_inference",
            "entry_point": PACKAGE_RUNNER_NAME,
            "training_supported": True,
            "evaluation_supported": True,
            "feedback_supported": True,
            "active_learning_query_supported": True,
            "interactive_inference_feedback_enabled": True,
            "interactive_feedback_supported": True,
            "interactive_inference_feedback_default": True,
            "interactive_inference_no_feedback_supported": True,
            "private_evaluation_feedback_frozen": True,
            "feedback_enabled": True,
            "feedback_config": PACKAGE_FEEDBACK_CONFIG_NAME,
        }
        if not isinstance(execution, Mapping):
            raise ValueError("submission task config is missing its execution contract")
        for key, expected in expected_execution.items():
            if execution.get(key) != expected:
                raise ValueError(
                    f"submission execution contract mismatch for {key}: "
                    f"expected {expected!r}, got {execution.get(key)!r}"
                )
        runner_path = _safe_extract_target(extraction_root, PACKAGE_RUNNER_NAME)
        if PACKAGE_RUNNER_NAME not in receipt_by_path or not runner_path.is_file():
            raise ValueError(
                "submission package is missing its runnable inference entry point"
            )
        if runner_path.read_bytes() != _submission_runner():
            raise ValueError(
                "submission execution entry point does not match its contract"
            )
        expected_training_configs = _submission_training_configs(
            expected_bundle,
            selected=selected,
        )
        for name, expected_bytes in expected_training_configs.items():
            path = _safe_extract_target(extraction_root, name)
            if path.read_bytes() != expected_bytes:
                raise ValueError(
                    f"submission adaptation config does not match its contract: {name}"
                )
        expected_benchmark_configs = _submission_benchmark_configs(
            selected=selected,
            pointer_arcname=pointer_arcname,
        )
        for name, expected_bytes in expected_benchmark_configs.items():
            path = _safe_extract_target(extraction_root, name)
            if path.read_bytes() != expected_bytes:
                raise ValueError(
                    f"submission benchmark config does not match its contract: {name}"
                )
        feedback_config = _safe_extract_target(
            extraction_root, PACKAGE_FEEDBACK_CONFIG_NAME
        )
        expected_feedback_config = _submission_feedback_config(
            selected=selected,
            pointer_arcname=pointer_arcname,
        )
        if feedback_config.read_bytes() != expected_feedback_config:
            raise ValueError(
                "submission feedback config does not match its contract"
            )

        evidence = manifest.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TypeError("submission manifest evidence must be a mapping")
        if evidence.get("scope") != "selected_category_only":
            raise ValueError("submission evidence scope must be selected_category_only")
        if (
            evidence.get("pooled_or_multi_category_machine_evidence_included")
            is not False
        ):
            raise ValueError("submission manifest does not exclude pooled evidence")
        evidence_files = evidence.get("included_files")
        if not isinstance(evidence_files, list):
            raise TypeError(
                "submission manifest evidence included_files must be a list"
            )
        evidence_records = evidence.get("records", [])
        if not isinstance(evidence_records, list):
            raise TypeError("submission manifest evidence records must be a list")
        records_by_path: dict[str, Mapping[str, Any]] = {}
        for row in evidence_records:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                raise TypeError("submission manifest evidence record is invalid")
            record_path = str(row["path"])
            if record_path in records_by_path:
                raise ValueError(f"duplicate submission evidence record: {record_path}")
            records_by_path[record_path] = row
        if set(records_by_path) != {str(name) for name in evidence_files}:
            raise ValueError("submission evidence records do not match included files")
        for name in evidence_files:
            if name not in receipt_by_path:
                raise ValueError(
                    f"manifested evidence is absent from package files: {name}"
                )
            evidence_metadata = _validate_evidence_file(
                _safe_extract_target(extraction_root, str(name)), selected
            )
            recorded_metadata = {
                key: records_by_path[str(name)].get(key)
                for key in evidence_metadata
                if key != "path"
            }
            if recorded_metadata != evidence_metadata:
                raise ValueError(f"submission evidence metadata mismatch: {name}")

        bundle_receipt = _verify_bundle_integrity_at(
            expected_bundle,
            package_root=extraction_root,
            selected_category=selected,
            pointer_dataset=(
                None if pointer.get("dataset") is None else str(pointer["dataset"])
            ),
            allowed_files=set(receipt_by_path),
        )
        runtime = manifest.get("runtime")
        if not isinstance(runtime, Mapping):
            raise TypeError("submission manifest runtime contract must be a mapping")
        if (
            runtime.get("entry_point") != PACKAGE_RUNNER_NAME
            or runtime.get("training_supported") is not True
            or runtime.get("evaluation_supported") is not True
            or runtime.get("feedback_supported") is not True
            or runtime.get("active_learning_query_supported") is not True
            or runtime.get("interactive_inference_feedback_enabled") is not True
            or runtime.get("interactive_feedback_supported") is not True
            or runtime.get("interactive_inference_feedback_default") is not True
            or runtime.get("interactive_inference_no_feedback_supported") is not True
            or runtime.get("private_evaluation_feedback_frozen") is not True
            or runtime.get("feedback_config") != PACKAGE_FEEDBACK_CONFIG_NAME
            or runtime.get("feedback_history_included") is not False
        ):
            raise ValueError("submission runtime contract is inconsistent")
        expected_dependencies = bundle_receipt.get("bundle_dependency_files")
        if runtime.get("bundle_dependency_files") != expected_dependencies:
            raise ValueError("submission bundle dependency manifest is inconsistent")
        if runtime.get("declared_assets") != bundle_receipt.get(
            "declared_runtime_assets_verified"
        ):
            raise ValueError("submission declared runtime assets are inconsistent")
        bundle_parent = PurePosixPath(bundle_arcname).parent.as_posix()
        packaged_under_bundle_parent = {
            name
            for name in receipt_by_path
            if name == bundle_parent or name.startswith(bundle_parent + "/")
        }
        if not packaged_under_bundle_parent.issubset(set(expected_dependencies or ())):
            raise ValueError(
                "submission contains files from the selected bundle directory "
                "that are not exact bundle dependencies"
            )
        return {
            "manifest_schema_version": 3,
            "selected_category": selected,
            "files_verified": len(receipt_by_path),
            "evidence_files_verified": len(evidence_files),
            "package_sha256_verified": True,
            **bundle_receipt,
        }


def build_submission(
    *,
    pointer: str | Path,
    category: str | None,
    output: str | Path,
    evidence: tuple[str | Path, ...] = (),
    maximum_bytes: int = MAX_SUBMISSION_BYTES,
) -> dict[str, Any]:
    if maximum_bytes <= 0 or maximum_bytes > MAX_SUBMISSION_BYTES:
        raise ValueError(f"maximum_bytes must be between 1 and {MAX_SUBMISSION_BYTES}")
    pointer_path = Path(pointer)
    if not pointer_path.is_absolute():
        pointer_path = PROJECT_ROOT / pointer_path
    pointer_path = pointer_path.resolve(strict=True)
    source_pointer, selected, bundle_path = _load_pointer(pointer_path, category)
    _project_relative(bundle_path)
    pointer_dataset = (
        None
        if source_pointer.get("dataset") is None
        else str(source_pointer["dataset"])
    )
    bundle_verification = _verify_bundle_integrity_at(
        bundle_path,
        package_root=PROJECT_ROOT,
        selected_category=selected,
        pointer_dataset=pointer_dataset,
    )
    bundle_dependency_files, declared_assets = _bundle_dependency_files(
        bundle_path,
        package_root=PROJECT_ROOT,
    )

    paths: set[Path] = set()
    for name in ROOT_FILES:
        candidate = PROJECT_ROOT / name
        if candidate.is_file():
            paths.add(candidate)
    paths.update(_collect_tree(PROJECT_ROOT / "aoi", suffixes={".py"}))
    paths.update(_collect_tree(PROJECT_ROOT / "tests", suffixes={".py"}))
    paths.update(bundle_dependency_files)
    portable_bundle_overrides = _submission_bundle_overrides(bundle_path)
    if portable_bundle_overrides:
        paths = {
            path
            for path in paths
            if _project_relative(path).as_posix() not in portable_bundle_overrides
        }

    evidence_paths: set[Path] = set()
    evidence_metadata_by_path: dict[Path, dict[str, Any]] = {}
    for raw in evidence:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve(strict=True)
        _project_relative(candidate)
        if _is_feedback_history_arcname(_project_relative(candidate)):
            raise ValueError(
                "operator feedback history/images/revisions cannot be packaged as submission evidence"
            )
        selected_paths = _collect_tree(candidate) if candidate.is_dir() else [candidate]
        for path in selected_paths:
            if not _eligible(path):
                continue
            if _is_feedback_history_arcname(_project_relative(path)):
                raise ValueError(
                    "operator feedback history/images/revisions cannot be packaged as submission evidence"
                )
            evidence_metadata_by_path[path] = _validate_evidence_file(path, selected)
            evidence_paths.add(path)
            paths.add(path)

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()
    protected_inputs = {
        pointer_path,
        bundle_path,
        *bundle_dependency_files,
        *evidence_paths,
    }
    for protected in protected_inputs:
        if _paths_overlap(output_path, protected):
            raise ValueError(
                f"output path overlaps protected submission input: {protected}"
            )
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"submission output must be a file path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
    for protected in protected_inputs:
        if _paths_overlap(temporary, protected):
            raise ValueError(
                f"temporary output path overlaps protected submission input: {protected}"
            )
    pointer_arcname = _project_relative(pointer_path).as_posix()
    bundle_arcname = _project_relative(bundle_path).as_posix()
    pointer_payload = _package_pointer(
        source_pointer,
        selected=selected,
        bundle_path=bundle_path,
        pointer_path=pointer_path,
    )
    pointer_bytes = yaml.safe_dump(
        pointer_payload, sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    evidence_arcnames = sorted(
        _project_relative(path).as_posix() for path in evidence_paths
    )
    evidence_records = sorted(
        (
            {
                "path": _project_relative(path).as_posix(),
                **metadata,
            }
            for path, metadata in evidence_metadata_by_path.items()
        ),
        key=lambda row: row["path"],
    )
    config_bytes = _submission_config(
        pointer_payload,
        selected=selected,
        pointer_arcname=pointer_arcname,
        bundle_arcname=bundle_arcname,
        evidence_files=evidence_arcnames,
    )
    readme_bytes = _submission_readme(
        selected=selected,
        pointer_arcname=pointer_arcname,
        bundle_arcname=bundle_arcname,
    )
    training_config_bytes = _submission_training_configs(
        bundle_path,
        selected=selected,
    )
    feedback_config_bytes = _submission_feedback_config(
        selected=selected,
        pointer_arcname=pointer_arcname,
    )
    benchmark_config_bytes = _submission_benchmark_configs(
        selected=selected,
        pointer_arcname=pointer_arcname,
    )
    generated = {
        PACKAGE_CONFIG_NAME: config_bytes,
        **training_config_bytes,
        PACKAGE_FEEDBACK_CONFIG_NAME: feedback_config_bytes,
        **benchmark_config_bytes,
        PACKAGE_README_NAME: readme_bytes,
        PACKAGE_MODEL_GUIDE_NAME: _submission_model_guide(
            selected=selected,
            pointer_arcname=pointer_arcname,
            bundle_arcname=bundle_arcname,
        ),
        PACKAGE_NOTICES_NAME: _submission_third_party_notices(),
        PACKAGE_RUNNER_NAME: _submission_runner(),
        PACKAGE_UI_LAUNCHER_NAME: _submission_ui_launcher(),
        **portable_bundle_overrides,
        pointer_arcname: pointer_bytes,
    }
    source_arcnames = {_project_relative(path).as_posix() for path in paths}
    leaked_feedback_history = sorted(
        name for name in source_arcnames if _is_feedback_history_arcname(name)
    )
    if leaked_feedback_history:
        raise ValueError(
            "submission source set contains operator feedback history: "
            + ", ".join(leaked_feedback_history[:3])
        )
    collisions = sorted(set(generated) & (source_arcnames - {pointer_arcname}))
    if collisions:
        raise ValueError(
            f"generated package files collide with source files: {collisions}"
        )

    file_receipts: list[dict[str, Any]] = []
    verification: dict[str, Any]
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(paths, key=lambda value: str(value).casefold()):
                arcname = _project_relative(path).as_posix()
                if arcname == pointer_arcname or arcname.startswith("configs/"):
                    continue
                archive.write(path, arcname)
                file_receipts.append(
                    {
                        "path": arcname,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_path(path),
                    }
                )
            for arcname, value in generated.items():
                archive.writestr(arcname, value)
                file_receipts.append(_receipt(arcname, value))

            manifest = {
                "schema_version": 3,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "package_scope": "single_category",
                "selected_category": selected,
                "selected_pointer": pointer_arcname,
                "model_bundle": bundle_arcname,
                "maximum_archive_bytes": maximum_bytes,
                "datasets_excluded": True,
                "history_and_unselected_models_excluded": True,
                "runtime": {
                    "mode": "adapt_then_evaluate_or_pretrained_inference",
                    "entry_point": PACKAGE_RUNNER_NAME,
                    "commands": {
                        "adapt": (
                            f"python {PACKAGE_RUNNER_NAME} adapt CUSTOM_TRAIN.yaml"
                        ),
                        "evaluate": (
                            f"python {PACKAGE_RUNNER_NAME} evaluate CUSTOM_TEST.yaml"
                        ),
                        "infer": (
                            f"python {PACKAGE_RUNNER_NAME} infer IMAGE_PATH --device auto"
                        ),
                        "feedback": (
                            f"python {PACKAGE_RUNNER_NAME} feedback status --device auto"
                        ),
                        "active_learning_query": (
                            f"python {PACKAGE_RUNNER_NAME} feedback query UNLABELED_DIR "
                            "--budget 20 --device auto"
                        ),
                    },
                    "training_supported": True,
                    "evaluation_supported": True,
                    "feedback_supported": True,
                    "active_learning_query_supported": True,
                    "interactive_inference_feedback_enabled": True,
                    "interactive_feedback_supported": True,
                    "interactive_inference_feedback_default": True,
                    "interactive_inference_no_feedback_supported": True,
                    "private_evaluation_feedback_frozen": True,
                    "feedback_config": PACKAGE_FEEDBACK_CONFIG_NAME,
                    "feedback_history_included": False,
                    "datasets_included": False,
                    "bundle_dependency_files": bundle_verification[
                        "bundle_dependency_files"
                    ],
                    "declared_assets": declared_assets,
                },
                "configuration": {
                    "scope": "single_category",
                    "repository_configs_included": False,
                    "pooled_configs_included": False,
                    "generated_files": PACKAGE_CONFIG_NAMES,
                    "reason": (
                        "Repository configs target pooled, historical, or example tasks; "
                        "the package contains generated single-task, private adaptation "
                        "and private evaluation configs bound to packaged source assets."
                    ),
                },
                "evidence": {
                    "scope": "selected_category_only",
                    "selected_category": selected,
                    "included_files": evidence_arcnames,
                    "records": evidence_records,
                    "pooled_or_multi_category_machine_evidence_included": False,
                    "boundary": (
                        "Only evidence whose structured category scope equals the selected "
                        "category is allowed. Pooled 8050-image metrics and predictions are "
                        "not reproducible by this archive and are excluded."
                    ),
                    "project_document_note": (
                        "Project-level documents may describe external development results; "
                        "those statements are not package evidence."
                    ),
                },
                "verification": {
                    "manifest_hash_algorithm": "sha256",
                    "manifest_covers_all_files_except_itself": True,
                    "post_build_extract_and_verify": True,
                    "bundle_hash_and_recorded_artifact_hashes_required": True,
                },
                "files": sorted(file_receipts, key=lambda row: row["path"]),
            }
            archive.writestr(
                MANIFEST_NAME,
                yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).encode(
                    "utf-8"
                ),
            )
        archive_bytes = temporary.stat().st_size
        if archive_bytes > maximum_bytes:
            raise ValueError(
                f"submission archive is {archive_bytes} bytes, exceeding "
                f"the {maximum_bytes}-byte limit"
            )
        verification = verify_submission_archive(
            temporary,
            maximum_bytes=maximum_bytes,
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output_path),
        "category": selected,
        "files": len(file_receipts) + 1,
        "archive_bytes": output_path.stat().st_size,
        "maximum_bytes": maximum_bytes,
        "under_size_limit": output_path.stat().st_size <= maximum_bytes,
        "sha256": _sha256_path(output_path),
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify one hash-manifested AOI category/task archive."
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="verify an existing archive instead of building one",
    )
    parser.add_argument(
        "--pointer",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dagm2007" / "active_models.yaml",
    )
    parser.add_argument("--category")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "submission" / "aoi_submission.zip",
    )
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--max-bytes", type=int, default=MAX_SUBMISSION_BYTES)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_submission_archive(
            args.verify,
            maximum_bytes=args.max_bytes,
        )
    else:
        result = build_submission(
            pointer=args.pointer,
            category=args.category,
            output=args.output,
            evidence=tuple(args.evidence),
            maximum_bytes=args.max_bytes,
        )
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
