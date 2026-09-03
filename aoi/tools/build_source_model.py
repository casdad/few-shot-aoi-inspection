"""Build a lineage-receipted ResNet18 PatchCore source model.

This command is intentionally separate from competition adaptation. Every
input image must be covered either by a DAGM ``Labels.txt`` manifest or by an
explicit hash inventory and human attestation. The receipt reports the
assurance each mode actually provides and refuses to overwrite outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
import cv2
import numpy as np
import PIL
import torchvision

from aoi.competition.baselines.deep_patchcore import DeepPatchCoreLite
from aoi.paths import PROJECT_ROOT
from aoi.runtime.data import read_rgb


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RESNET18_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class _ImageSelection:
    paths: tuple[Path, ...]
    evidence: dict[str, Any]
    status: str
    normal_only_selection_verified: bool
    normal_only_attested: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _supported_images(root: Path, *, recursive: bool) -> dict[str, Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    images: dict[str, Path] = {}
    folded_paths: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if candidate.is_symlink():
            raise ValueError(f"source image symlinks are not allowed: {candidate}")
        relative = candidate.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in folded_paths:
            raise ValueError(f"duplicate case-insensitive source image path: {relative}")
        folded_paths.add(folded)
        images[relative] = candidate.resolve(strict=True)
    return images


def _verify_unique_content(paths: list[Path]) -> dict[Path, str]:
    hashes = {path: _sha256(path) for path in paths}
    seen: dict[str, Path] = {}
    for path, digest in hashes.items():
        previous = seen.get(digest)
        if previous is not None:
            raise ValueError(
                "source evidence contains duplicate image content: "
                f"{previous} and {path}"
            )
        seen[digest] = path
    return hashes


def _select_dagm_normals(
    root: Path,
    manifest_value: str | Path,
    *,
    maximum_images: int,
) -> _ImageSelection:
    manifest = _resolve_file(manifest_value, label="dagm_manifest")
    expected_manifest = (root / "Label" / "Labels.txt").resolve()
    if os.path.normcase(str(manifest)) != os.path.normcase(str(expected_manifest)):
        raise ValueError(
            "dagm_manifest must be normal_dir/Label/Labels.txt so its rows bind "
            "the supplied DAGM split"
        )

    manifest_bytes = manifest.read_bytes()
    try:
        lines = [
            line.strip()
            for line in manifest_bytes.decode("utf-8-sig").splitlines()
            if line.strip()
        ]
    except UnicodeDecodeError as exc:
        raise ValueError(f"DAGM manifest is not UTF-8 text: {manifest}") from exc
    if len(lines) < 2:
        raise ValueError(f"DAGM manifest has no samples: {manifest}")
    try:
        int(lines[0])
    except ValueError as exc:
        raise ValueError(
            f"invalid DAGM class header in {manifest}: {lines[0]!r}"
        ) from exc

    rows: list[tuple[str, int, Path]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        columns = line.split()
        if len(columns) < 3:
            raise ValueError(f"invalid DAGM row at {manifest}:{line_number}: {line!r}")
        record_id, raw_label, image_name = columns[:3]
        if Path(image_name).name != image_name:
            raise ValueError(f"unsafe DAGM image name at {manifest}:{line_number}")
        if Path(image_name).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"unsupported DAGM image extension at {manifest}:{line_number}"
            )
        try:
            label = int(raw_label)
        except ValueError as exc:
            raise ValueError(
                f"invalid DAGM label at {manifest}:{line_number}: {raw_label!r}"
            ) from exc
        if label not in {0, 1}:
            raise ValueError(
                f"DAGM label must be 0 or 1 at {manifest}:{line_number}"
            )
        folded_id = record_id.casefold()
        folded_name = image_name.casefold()
        if folded_id in seen_ids:
            raise ValueError(f"duplicate DAGM record id in {manifest}: {record_id}")
        if folded_name in seen_names:
            raise ValueError(f"duplicate DAGM image row in {manifest}: {image_name}")
        seen_ids.add(folded_id)
        seen_names.add(folded_name)
        image_path = (root / image_name).resolve(strict=True)
        if image_path.parent != root:
            raise ValueError(f"DAGM image resolves outside normal_dir: {image_name}")
        rows.append((image_name, label, image_path))

    actual_images = _supported_images(root, recursive=False)
    actual_names = {name.casefold() for name in actual_images}
    if actual_names != seen_names:
        missing = sorted(seen_names - actual_names)
        unlisted = sorted(actual_names - seen_names)
        raise ValueError(
            "DAGM manifest/image mismatch: "
            f"missing={missing[:5]}, unlisted={unlisted[:5]}"
        )

    all_paths = [path for _, _, path in rows]
    all_hashes = _verify_unique_content(all_paths)
    normal_rows = sorted(
        ((name, path) for name, label, path in rows if label == 0),
        key=lambda item: item[0].casefold(),
    )
    selected = normal_rows[:maximum_images]
    if not selected:
        raise ValueError("DAGM manifest contains no label=0 images")
    return _ImageSelection(
        paths=tuple(path for _, path in selected),
        evidence={
            "mode": "dagm_labels_manifest",
            "path": str(manifest),
            "sha256": _sha256_bytes(manifest_bytes),
            "manifest_row_count": len(rows),
            "manifest_normal_count": sum(label == 0 for _, label, _ in rows),
            "manifest_defect_count": sum(label == 1 for _, label, _ in rows),
            "selected_labels": [0],
            "manifest_inventory_exact_match": True,
            "normal_assurance": "verified_against_bound_manifest_labels_only",
            "semantic_image_content_reviewed": False,
            "all_manifest_image_hashes": [
                {
                    "path": name,
                    "label": label,
                    "sha256": all_hashes[path],
                }
                for name, label, path in rows
            ],
        },
        status="lineage_receipted_manifest_normal_labels_verified",
        normal_only_selection_verified=True,
        normal_only_attested=False,
    )


def _safe_inventory_path(root: Path, raw_path: Any, *, row_number: int) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"inventory image row {row_number} has no path")
    relative_text = raw_path.strip().replace("\\", "/")
    relative = Path(*relative_text.split("/"))
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"unsafe inventory image path at row {row_number}: {raw_path!r}")
    image_path = (root / relative).resolve(strict=True)
    try:
        canonical_relative = image_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"inventory image resolves outside normal_dir at row {row_number}: {raw_path!r}"
        ) from exc
    if canonical_relative.casefold() != relative_text.casefold():
        raise ValueError(
            f"inventory image path is not canonical at row {row_number}: {raw_path!r}"
        )
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported inventory image extension at row {row_number}")
    if image_path.is_symlink():
        raise ValueError(f"source image symlinks are not allowed: {image_path}")
    return canonical_relative, image_path


def _require_attestation_text(attestation: dict[str, Any], field: str) -> str:
    value = attestation.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"generic inventory attestation requires non-empty {field}")
    return value.strip()


def _select_attested_inventory(
    root: Path,
    inventory_value: str | Path,
    *,
    maximum_images: int,
) -> _ImageSelection:
    inventory = _resolve_file(inventory_value, label="inventory_attestation")
    inventory_bytes = inventory.read_bytes()
    try:
        payload = yaml.safe_load(inventory_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid generic inventory YAML: {inventory}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("generic inventory must be a schema_version: 1 mapping")
    attestation = payload.get("attestation")
    if not isinstance(attestation, dict):
        raise ValueError("generic inventory requires an attestation mapping")
    claims = attestation.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("generic inventory attestation requires a claims mapping")
    if claims.get("all_images_are_normal") is not True:
        raise ValueError("generic inventory must attest all_images_are_normal: true")
    if claims.get("private_task_data_included") is not False:
        raise ValueError(
            "generic inventory must attest private_task_data_included: false"
        )
    attested_by = _require_attestation_text(attestation, "attested_by")
    attested_at = _require_attestation_text(attestation, "attested_at")
    basis = _require_attestation_text(attestation, "basis")

    image_rows = payload.get("images")
    if not isinstance(image_rows, list) or not image_rows:
        raise ValueError("generic inventory requires a non-empty images list")
    rows: list[tuple[str, Path, str]] = []
    seen_paths: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for row_number, row in enumerate(image_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"inventory image row {row_number} must be a mapping")
        if row.get("role") != "normal":
            raise ValueError(f"inventory image row {row_number} must have role: normal")
        relative, image_path = _safe_inventory_path(
            root, row.get("path"), row_number=row_number
        )
        folded = relative.casefold()
        if folded in seen_paths:
            raise ValueError(f"duplicate inventory image path: {relative}")
        seen_paths.add(folded)
        expected_hash = row.get("sha256")
        if not isinstance(expected_hash, str) or SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise ValueError(f"invalid SHA-256 in inventory image row {row_number}")
        expected_hash = expected_hash.lower()
        actual_hash = _sha256(image_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"inventory SHA-256 mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        previous = seen_hashes.get(actual_hash)
        if previous is not None:
            raise ValueError(
                f"source evidence contains duplicate image content: {previous} and {relative}"
            )
        seen_hashes[actual_hash] = relative
        rows.append((relative, image_path, actual_hash))

    actual_images = _supported_images(root, recursive=True)
    actual_names = {name.casefold() for name in actual_images}
    if actual_names != seen_paths:
        missing = sorted(seen_paths - actual_names)
        unlisted = sorted(actual_names - seen_paths)
        raise ValueError(
            "generic inventory/image mismatch: "
            f"missing={missing[:5]}, unlisted={unlisted[:5]}"
        )
    selected = sorted(rows, key=lambda item: item[0].casefold())[:maximum_images]
    return _ImageSelection(
        paths=tuple(path for _, path, _ in selected),
        evidence={
            "mode": "generic_inventory_attestation",
            "path": str(inventory),
            "sha256": _sha256_bytes(inventory_bytes),
            "inventory_image_count": len(rows),
            "inventory_exact_match": True,
            "normal_assurance": "human_attested_not_independently_verified",
            "attestation": {
                "attested_by": attested_by,
                "attested_at": attested_at,
                "basis": basis,
                "claims": {
                    "all_images_are_normal": True,
                    "private_task_data_included": False,
                },
            },
        },
        status="lineage_receipted_normal_only_attested_not_verified",
        normal_only_selection_verified=False,
        normal_only_attested=True,
    )


def _load_local_resnet(model: DeepPatchCoreLite, weights: Path) -> None:
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("torchvision ResNet18 weight file must contain a state dict")
    target = {}
    prefixes = {
        "conv1.": "stem.0.",
        "bn1.": "stem.1.",
        "layer1.": "layer1.",
        "layer2.": "layer2.",
        "layer3.": "layer3.",
    }
    for key, value in payload.items():
        for source_prefix, target_prefix in prefixes.items():
            if key.startswith(source_prefix):
                target[target_prefix + key[len(source_prefix) :]] = value
                break
    result = model.extractor.load_state_dict(target, strict=False)
    unexpected = [key for key in result.unexpected_keys if key not in {"mean", "std"}]
    missing = [key for key in result.missing_keys if key not in {"mean", "std"}]
    if unexpected or missing:
        raise ValueError(
            f"local ResNet18 mapping mismatch: missing={missing}, unexpected={unexpected}"
        )


def build_source_model(
    *,
    normal_dir: str | Path,
    output: str | Path,
    receipt: str | Path,
    weights: str | Path = PROJECT_ROOT / "assets" / "resnet18-f37072fd.pth",
    maximum_images: int = 256,
    source_dataset: str | None = None,
    source_url: str | None = None,
    license_evidence: str | Path | None = None,
    dagm_manifest: str | Path | None = None,
    inventory_attestation: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(normal_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"normal_dir must be a directory: {root}")
    dataset_name = str(source_dataset or "").strip()
    dataset_url = str(source_url or "").strip()
    if not dataset_name or not dataset_url:
        raise ValueError(
            "source_dataset and source_url are required so the declared source "
            "identity is explicit"
        )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()
    receipt_path = Path(receipt)
    if not receipt_path.is_absolute():
        receipt_path = PROJECT_ROOT / receipt_path
    receipt_path = receipt_path.resolve()
    if os.path.normcase(str(output_path)) == os.path.normcase(str(receipt_path)):
        raise ValueError("output and receipt must be different paths")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite source-model output or receipt")
    weight_path = Path(weights)
    if not weight_path.is_absolute():
        weight_path = PROJECT_ROOT / weight_path
    weight_path = weight_path.resolve(strict=True)
    actual_weight_hash = _sha256(weight_path)
    if actual_weight_hash != RESNET18_SHA256:
        raise ValueError(
            "ResNet18 weight SHA-256 mismatch: "
            f"expected {RESNET18_SHA256}, got {actual_weight_hash}"
        )
    license_path = None
    license_hash = None
    if license_evidence is not None:
        license_path = Path(license_evidence)
        if not license_path.is_absolute():
            license_path = PROJECT_ROOT / license_path
        license_path = license_path.resolve(strict=True)
        license_hash = _sha256(license_path)
    else:
        raise ValueError(
            "license_evidence is required; provide the dataset license or authorization receipt"
        )
    if maximum_images <= 0:
        raise ValueError("maximum_images must be a positive explicit limit")
    if (dagm_manifest is None) == (inventory_attestation is None):
        raise ValueError(
            "provide exactly one evidence mode: dagm_manifest or inventory_attestation"
        )
    selection = (
        _select_dagm_normals(root, dagm_manifest, maximum_images=maximum_images)
        if dagm_manifest is not None
        else _select_attested_inventory(
            root, inventory_attestation, maximum_images=maximum_images
        )
    )
    paths = list(selection.paths)
    image_hashes_by_path = _verify_unique_content(paths)
    image_hashes = [image_hashes_by_path[path] for path in paths]

    model = DeepPatchCoreLite(
        input_size=(128, 128),
        max_memory=512,
        projection_dim=64,
        seed=20260945,
        pretrained=False,
        device="cpu",
    )
    _load_local_resnet(model, weight_path)
    loaded_images = []
    for path, expected_hash in zip(paths, image_hashes, strict=True):
        image = read_rgb(path)
        current_hash = _sha256(path)
        if current_hash != expected_hash:
            raise RuntimeError(
                f"source image changed while building the model: {path}"
            )
        loaded_images.append(image)
    model.fit(loaded_images, batch_size=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_token = f"{os.getpid()}.{secrets.token_hex(8)}"
    temporary_output = output_path.with_name(
        f".{output_path.name}.{temporary_token}.partial"
    )
    temporary_receipt = receipt_path.with_name(
        f".{receipt_path.name}.{temporary_token}.partial"
    )
    receipt_payload: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": selection.status,
        "source_dataset": dataset_name,
        "source_dataset_url": dataset_url,
        "license_evidence": (
            None
            if license_path is None
            else {"path": str(license_path), "sha256": license_hash}
        ),
        "source_identity_assurance": "declared_not_independently_verified",
        "license_assurance": "file_hash_bound_not_legal_review",
        "source_directory": str(root),
        "source_image_count": len(paths),
        "source_image_limit": maximum_images,
        "source_images": [
            {"path": path.relative_to(root).as_posix(), "sha256": digest}
            for path, digest in zip(paths, image_hashes, strict=True)
        ],
        "resnet18_weights": {
            "path": str(weight_path),
            "sha256": actual_weight_hash,
            "expected_sha256": RESNET18_SHA256,
        },
        "output_model": str(output_path),
        "output_sha256": None,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "command": list(sys.argv),
        "python": sys.version,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
        "pillow": str(PIL.__version__),
        "platform": platform.platform(),
        "git_commit": None,
        "selection_rule": (
            f"{selection.evidence['mode']}; lexicographic relative-path order; "
            f"maximum_images={maximum_images}"
        ),
        "evidence": selection.evidence,
        # A manifest can verify row labels, but neither labels nor a human
        # attestation proves the private/public origin of the underlying files.
        "normal_only_selection_verified": selection.normal_only_selection_verified,
        "normal_only_attested": selection.normal_only_attested,
        "private_task_data_included": None,
        "defect_images_included": None,
        "manifest_labeled_defect_images_selected": (
            False if selection.evidence["mode"] == "dagm_labels_manifest" else None
        ),
        "attested_private_task_data_included": (
            selection.evidence.get("attestation", {})
            .get("claims", {})
            .get("private_task_data_included")
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    reserved_output = False
    reserved_receipt = False
    committed_output = False
    committed_receipt = False
    try:
        model.save(temporary_output)
        receipt_payload["output_sha256"] = _sha256(temporary_output)
        temporary_receipt.write_text(
            yaml.safe_dump(receipt_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # Reserve both destinations immediately before commit. This keeps a
        # concurrent invocation from being silently overwritten by os.replace.
        for destination, flag_name in (
            (output_path, "reserved_output"),
            (receipt_path, "reserved_receipt"),
        ):
            try:
                descriptor = os.open(
                    destination,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite source-model output or receipt: {destination}"
                ) from exc
            else:
                os.close(descriptor)
                if flag_name == "reserved_output":
                    reserved_output = True
                else:
                    reserved_receipt = True
        os.replace(temporary_output, output_path)
        committed_output = True
        os.replace(temporary_receipt, receipt_path)
        committed_receipt = True
    except Exception:
        temporary_receipt.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)
        if committed_output or reserved_output:
            output_path.unlink(missing_ok=True)
        if committed_receipt or reserved_receipt:
            receipt_path.unlink(missing_ok=True)
        raise
    return receipt_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normal_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "assets" / "resnet18-f37072fd.pth")
    parser.add_argument("--maximum-images", type=int, default=256)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument(
        "--dagm-manifest",
        type=Path,
        help="bound normal_dir/Label/Labels.txt; only label=0 rows are selected",
    )
    evidence.add_argument(
        "--inventory-attestation",
        type=Path,
        help="schema-versioned YAML hash inventory with human normal-only attestation",
    )
    args = parser.parse_args()
    result = build_source_model(
        normal_dir=args.normal_dir,
        output=args.output,
        receipt=args.receipt,
        weights=args.weights,
        maximum_images=args.maximum_images,
        source_dataset=args.source_dataset,
        source_url=args.source_url,
        license_evidence=args.license_evidence,
        dagm_manifest=args.dagm_manifest,
        inventory_attestation=args.inventory_attestation,
    )
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
