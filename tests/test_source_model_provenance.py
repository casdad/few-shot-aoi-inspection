from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

import aoi.tools.build_source_model as source_builder
from aoi.tools.build_source_model import build_source_model


def _fixture_dir(tmp_path: Path) -> Path:
    root = tmp_path / "public_normals"
    root.mkdir()
    (root / "normal.png").write_bytes(b"not-an-image")
    return root


def _license(tmp_path: Path) -> Path:
    path = tmp_path / "license.txt"
    path.write_text("authorization", encoding="utf-8")
    return path


class _FakeModel:
    fitted_images: list[np.ndarray] = []

    def __init__(self, *args, **kwargs) -> None:
        self.images: list[np.ndarray] = []

    def fit(self, images: list[np.ndarray], *, batch_size: int) -> None:
        self.images = images
        type(self).fitted_images = images

    def save(self, path: Path) -> None:
        path.write_bytes(b"fake-source-model")


def _patch_training(monkeypatch: pytest.MonkeyPatch, weight: Path) -> None:
    monkeypatch.setattr(source_builder, "DeepPatchCoreLite", _FakeModel)
    monkeypatch.setattr(source_builder, "_load_local_resnet", lambda model, path: None)
    monkeypatch.setattr(
        source_builder,
        "read_rgb",
        lambda path: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        source_builder,
        "RESNET18_SHA256",
        hashlib.sha256(weight.read_bytes()).hexdigest(),
    )


def _dagm_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "dagm_train"
    label_root = root / "Label"
    label_root.mkdir(parents=True)
    (root / "normal.png").write_bytes(b"normal-image")
    (root / "defect.png").write_bytes(b"defect-image")
    (label_root / "Labels.txt").write_text(
        "1\nnormal-id 0 normal.png 0 0\ndefect-id 1 defect.png 0 0\n",
        encoding="utf-8",
    )
    return root


def _generic_inventory(root: Path, *, all_normal: bool = True) -> Path:
    image_rows = []
    for path in sorted(root.rglob("*.png")):
        image_rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": "normal",
            }
        )
    inventory = root.parent / "inventory.yaml"
    inventory.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "images": image_rows,
                "attestation": {
                    "attested_by": "qa@example.invalid",
                    "attested_at": "2026-08-21T00:00:00Z",
                    "basis": "manual review and source register",
                    "claims": {
                        "all_images_are_normal": all_normal,
                        "private_task_data_included": False,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return inventory


def test_source_builder_requires_declared_dataset_and_license(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_dataset and source_url"):
        build_source_model(
            normal_dir=_fixture_dir(tmp_path),
            output=tmp_path / "source.pt",
            receipt=tmp_path / "receipt.yaml",
        )


def test_source_builder_rejects_same_output_and_receipt_path(tmp_path: Path) -> None:
    same = tmp_path / "same"
    with pytest.raises(ValueError, match="different paths"):
        build_source_model(
            normal_dir=_fixture_dir(tmp_path),
            output=same,
            receipt=same,
            source_dataset="public",
            source_url="https://example.invalid/data",
            license_evidence=tmp_path / "license.txt",
        )


def test_source_builder_verifies_pinned_resnet_weight_before_training(
    tmp_path: Path,
) -> None:
    license_path = tmp_path / "license.txt"
    license_path.write_text("authorization", encoding="utf-8")
    wrong_weights = tmp_path / "wrong.pth"
    wrong_weights.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="ResNet18 weight SHA-256 mismatch"):
        build_source_model(
            normal_dir=_fixture_dir(tmp_path),
            output=tmp_path / "source.pt",
            receipt=tmp_path / "receipt.yaml",
            weights=wrong_weights,
            source_dataset="public",
            source_url="https://example.invalid/data",
            license_evidence=license_path,
        )


def test_source_builder_requires_exactly_one_selection_evidence_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dagm_fixture(tmp_path)
    inventory_root = tmp_path / "inventory_normals"
    inventory_root.mkdir()
    (inventory_root / "normal.png").write_bytes(b"inventory-normal")
    inventory = _generic_inventory(inventory_root)
    weight = tmp_path / "resnet.pth"
    weight.write_bytes(b"pinned-weight")
    _patch_training(monkeypatch, weight)
    common = {
        "normal_dir": root,
        "output": tmp_path / "source.pt",
        "receipt": tmp_path / "receipt.yaml",
        "weights": weight,
        "source_dataset": "DAGM2007",
        "source_url": "https://example.invalid/dagm",
        "license_evidence": _license(tmp_path),
    }

    with pytest.raises(ValueError, match="exactly one evidence mode"):
        build_source_model(**common)
    with pytest.raises(ValueError, match="exactly one evidence mode"):
        build_source_model(
            **common,
            dagm_manifest=root / "Label" / "Labels.txt",
            inventory_attestation=inventory,
        )


def test_source_builder_dagm_manifest_selects_only_label_zero_and_binds_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dagm_fixture(tmp_path)
    weight = tmp_path / "resnet.pth"
    weight.write_bytes(b"pinned-weight")
    _patch_training(monkeypatch, weight)

    result = build_source_model(
        normal_dir=root,
        output=tmp_path / "source.pt",
        receipt=tmp_path / "receipt.yaml",
        weights=weight,
        source_dataset="DAGM2007",
        source_url="https://example.invalid/dagm",
        license_evidence=_license(tmp_path),
        dagm_manifest=root / "Label" / "Labels.txt",
    )

    assert len(_FakeModel.fitted_images) == 1
    assert result["status"] == "lineage_receipted_manifest_normal_labels_verified"
    assert result["normal_only_selection_verified"] is True
    assert result["normal_only_attested"] is False
    assert result["private_task_data_included"] is None
    assert result["defect_images_included"] is None
    assert result["manifest_labeled_defect_images_selected"] is False
    assert [row["path"] for row in result["source_images"]] == ["normal.png"]
    assert result["evidence"]["manifest_defect_count"] == 1
    assert result["evidence"]["sha256"] == hashlib.sha256(
        (root / "Label" / "Labels.txt").read_bytes()
    ).hexdigest()
    assert yaml.safe_load((tmp_path / "receipt.yaml").read_text(encoding="utf-8")) == result


def test_source_builder_dagm_manifest_rejects_unlisted_image(tmp_path: Path) -> None:
    root = _dagm_fixture(tmp_path)
    (root / "unlisted.png").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="manifest/image mismatch"):
        source_builder._select_dagm_normals(
            root,
            root / "Label" / "Labels.txt",
            maximum_images=10,
        )


def test_source_builder_generic_inventory_is_attested_not_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "normals"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested" / "normal.png").write_bytes(b"normal-image")
    inventory = _generic_inventory(root)
    weight = tmp_path / "resnet.pth"
    weight.write_bytes(b"pinned-weight")
    _patch_training(monkeypatch, weight)

    result = build_source_model(
        normal_dir=root,
        output=tmp_path / "source.pt",
        receipt=tmp_path / "receipt.yaml",
        weights=weight,
        source_dataset="internal-register",
        source_url="https://example.invalid/register",
        license_evidence=_license(tmp_path),
        inventory_attestation=inventory,
    )

    assert result["status"] == "lineage_receipted_normal_only_attested_not_verified"
    assert result["normal_only_selection_verified"] is False
    assert result["normal_only_attested"] is True
    assert result["private_task_data_included"] is None
    assert result["attested_private_task_data_included"] is False
    assert result["evidence"]["normal_assurance"] == (
        "human_attested_not_independently_verified"
    )


def test_source_builder_generic_inventory_rejects_tampered_hash(tmp_path: Path) -> None:
    root = tmp_path / "normals"
    root.mkdir()
    (root / "normal.png").write_bytes(b"normal-image")
    inventory = _generic_inventory(root)
    (root / "normal.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        source_builder._select_attested_inventory(root, inventory, maximum_images=10)


def test_source_builder_generic_inventory_requires_normal_only_attestation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normals"
    root.mkdir()
    (root / "normal.png").write_bytes(b"normal-image")
    inventory = _generic_inventory(root, all_normal=False)
    with pytest.raises(ValueError, match="all_images_are_normal: true"):
        source_builder._select_attested_inventory(root, inventory, maximum_images=10)


def test_source_builder_commit_failure_rolls_back_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dagm_fixture(tmp_path)
    weight = tmp_path / "resnet.pth"
    weight.write_bytes(b"pinned-weight")
    _patch_training(monkeypatch, weight)
    output = tmp_path / "source.pt"
    receipt = tmp_path / "receipt.yaml"
    original_replace = source_builder.os.replace

    def fail_receipt_replace(source: Path, destination: Path) -> None:
        if destination == receipt:
            raise OSError("simulated commit failure")
        original_replace(source, destination)

    monkeypatch.setattr(source_builder.os, "replace", fail_receipt_replace)
    with pytest.raises(OSError, match="simulated commit failure"):
        build_source_model(
            normal_dir=root,
            output=output,
            receipt=receipt,
            weights=weight,
            source_dataset="DAGM2007",
            source_url="https://example.invalid/dagm",
            license_evidence=_license(tmp_path),
            dagm_manifest=root / "Label" / "Labels.txt",
        )

    assert not output.exists()
    assert not receipt.exists()
    assert not list(tmp_path.glob("*.partial"))
