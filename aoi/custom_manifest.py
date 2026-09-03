"""Strict, auditable label manifests for private AOI test sets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aoi.training.dataset import IMAGE_EXTENSIONS
from aoi.training.manifests import sha256_file


@dataclass(frozen=True)
class CustomTestSample:
    sample_id: str
    path: Path
    label: int
    defect_type: str
    sha256: str


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a YAML mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{context} keys must be strings")
    return value


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unsupported fields: {unknown}")


def _required_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value.strip()


def _label(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be 0/1 or normal/defect")
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "0": 0,
            "normal": 0,
            "good": 0,
            "ok": 0,
            "1": 1,
            "defect": 1,
            "anomaly": 1,
            "ng": 1,
        }
        if normalized in aliases:
            return aliases[normalized]
    raise ValueError(f"{context} must be 0/1 or normal/defect")


def _resolved_image(image_root: Path, value: Any, context: str) -> Path:
    raw = Path(_required_text(value, context))
    if raw.is_absolute():
        raise ValueError(f"{context} must be relative to image_root")
    candidate = (image_root / raw).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes image_root: {raw}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{context} does not exist: {candidate}")
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"{context} is not a supported image: {candidate}")
    return candidate


def load_custom_test_manifest(
    path: str | Path,
    *,
    minimum_samples: int = 1000,
    excluded_hashes: set[str] | None = None,
) -> tuple[CustomTestSample, ...]:
    """Load a labeled test manifest and reject duplicates or support leakage."""

    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int):
        raise TypeError("minimum_samples must be an integer")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be >= 1")
    manifest_path = Path(path).resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError(f"custom test manifest is not a file: {manifest_path}")
    payload = _mapping(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8")),
        "custom test manifest",
    )
    _reject_unknown(
        payload,
        {"schema_version", "image_root", "samples"},
        "custom test manifest",
    )
    if payload.get("schema_version") != 1 or type(payload.get("schema_version")) is not int:
        raise ValueError("custom test manifest schema_version must be integer 1")

    raw_root = Path(
        _required_text(payload.get("image_root"), "custom test manifest.image_root")
    )
    image_root = raw_root if raw_root.is_absolute() else manifest_path.parent / raw_root
    image_root = image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise ValueError(f"custom test image_root is not a directory: {image_root}")

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise TypeError("custom test manifest.samples must be a YAML list")
    if len(raw_samples) < minimum_samples:
        raise ValueError(
            f"custom labeled test requires >= {minimum_samples} images; "
            f"found {len(raw_samples)}"
        )

    support_hashes = excluded_hashes or set()
    ids: set[str] = set()
    paths: set[Path] = set()
    hashes: set[str] = set()
    samples: list[CustomTestSample] = []
    for index, raw_sample in enumerate(raw_samples):
        context = f"custom test manifest.samples[{index}]"
        sample = _mapping(raw_sample, context)
        _reject_unknown(
            sample,
            {"id", "image", "label", "defect_type", "sha256"},
            context,
        )
        sample_id = _required_text(
            sample.get("id", f"custom-test-{index + 1:06d}"),
            f"{context}.id",
        )
        if sample_id in ids:
            raise ValueError(f"duplicate custom test sample id: {sample_id!r}")
        image_path = _resolved_image(image_root, sample.get("image"), f"{context}.image")
        if image_path in paths:
            raise ValueError(f"duplicate custom test image path: {image_path}")
        label = _label(sample.get("label"), f"{context}.label")
        raw_defect_type = sample.get("defect_type")
        if label == 0:
            if raw_defect_type is not None and raw_defect_type not in {
                "",
                "normal",
                "good",
            }:
                raise ValueError(f"{context}.defect_type must be normal or omitted for label 0")
            defect_type = "normal"
        else:
            defect_type = (
                "defect"
                if raw_defect_type is None
                else _required_text(raw_defect_type, f"{context}.defect_type")
            )

        digest = sha256_file(image_path)
        expected_digest = sample.get("sha256")
        if expected_digest is not None:
            expected = _required_text(expected_digest, f"{context}.sha256").lower()
            if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
                raise ValueError(f"{context}.sha256 must be a 64-character hexadecimal digest")
            if expected != digest:
                raise ValueError(f"{context}.sha256 does not match image content")
        if digest in hashes:
            raise ValueError(f"duplicate custom test image content: {image_path}")
        if digest in support_hashes:
            raise PermissionError(
                f"CUSTOM_TEST_SUPPORT_LEAKAGE: test image matches 100+30 support: {image_path}"
            )

        ids.add(sample_id)
        paths.add(image_path)
        hashes.add(digest)
        samples.append(
            CustomTestSample(
                sample_id=sample_id,
                path=image_path,
                label=label,
                defect_type=defect_type,
                sha256=digest,
            )
        )

    labels = {sample.label for sample in samples}
    if labels != {0, 1}:
        raise ValueError("custom labeled test must contain both normal and defect samples")
    return tuple(samples)


__all__ = ["CustomTestSample", "load_custom_test_manifest"]
