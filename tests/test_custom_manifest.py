from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aoi.custom_manifest import load_custom_test_manifest
from aoi.training.manifests import sha256_file


def _manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    path = tmp_path / "test_manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "image_root": "images", "samples": rows},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_custom_manifest_loads_labels_types_and_optional_digest(
    tmp_path: Path,
) -> None:
    normal = tmp_path / "images" / "normal.png"
    defect = tmp_path / "images" / "defect.png"
    normal.parent.mkdir()
    normal.write_bytes(b"normal-image")
    defect.write_bytes(b"defect-image")
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": "normal-1",
                "image": "normal.png",
                "label": "normal",
                "sha256": sha256_file(normal),
            },
            {
                "image": "defect.png",
                "label": 1,
                "defect_type": "scratch",
            },
        ],
    )

    samples = load_custom_test_manifest(manifest, minimum_samples=2)

    assert [sample.sample_id for sample in samples] == [
        "normal-1",
        "custom-test-000002",
    ]
    assert [sample.label for sample in samples] == [0, 1]
    assert [sample.defect_type for sample in samples] == ["normal", "scratch"]
    assert samples[0].sha256 == sha256_file(normal)


def test_custom_manifest_rejects_support_leakage_and_duplicate_content(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    normal = image_root / "normal.png"
    duplicate = image_root / "duplicate.png"
    defect = image_root / "defect.png"
    normal.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    defect.write_bytes(b"different")

    leaking = _manifest(
        tmp_path,
        [
            {"image": "normal.png", "label": 0},
            {"image": "defect.png", "label": 1},
        ],
    )
    with pytest.raises(PermissionError, match="SUPPORT_LEAKAGE"):
        load_custom_test_manifest(
            leaking,
            minimum_samples=2,
            excluded_hashes={sha256_file(normal)},
        )

    duplicate_manifest = _manifest(
        tmp_path,
        [
            {"image": "normal.png", "label": 0},
            {"image": "duplicate.png", "label": 1},
        ],
    )
    with pytest.raises(ValueError, match="duplicate.*content"):
        load_custom_test_manifest(duplicate_manifest, minimum_samples=2)


def test_custom_manifest_enforces_count_both_labels_and_root_boundary(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "normal.png").write_bytes(b"normal")
    (tmp_path / "outside.png").write_bytes(b"outside")
    one = _manifest(
        tmp_path,
        [{"image": "normal.png", "label": "normal"}],
    )
    with pytest.raises(ValueError, match=">= 2"):
        load_custom_test_manifest(one, minimum_samples=2)
    with pytest.raises(ValueError, match="both normal and defect"):
        load_custom_test_manifest(one, minimum_samples=1)

    escaped = _manifest(
        tmp_path,
        [
            {"image": "normal.png", "label": 0},
            {"image": "../outside.png", "label": 1},
        ],
    )
    with pytest.raises(ValueError, match="escapes image_root"):
        load_custom_test_manifest(escaped, minimum_samples=2)
