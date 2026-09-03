from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aoi.paths as artifact_paths  # noqa: E402
from aoi.images import read_rgb  # noqa: E402
from aoi.runtime.config import load_yaml, resolve_path  # noqa: E402
from aoi.runtime.data import load_dagm_category_test  # noqa: E402
from aoi.training.config import load_config  # noqa: E402
from aoi.training.dagm import (  # noqa: E402
    build_episode,
    list_categories,
    read_split_records,
)


def _write_split(
    class_root: Path,
    split: str,
    rows: list[tuple[str, int, bytes]],
) -> None:
    split_root = class_root / split
    label_root = split_root / "Label"
    label_root.mkdir(parents=True)
    manifest = [class_root.name.removeprefix("Class")]
    for record_id, label, payload in rows:
        image_name = f"{record_id}.PNG"
        (split_root / image_name).write_bytes(payload)
        mask_name = "0"
        if label:
            mask_name = f"{record_id}_label.PNG"
            (label_root / mask_name).write_bytes(b"mask-" + payload)
        manifest.append(f"{record_id}\t{label}\t{image_name}\t0\t{mask_name}")
    (label_root / "Labels.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def test_dagm_categories_use_numeric_class_order(tmp_path: Path) -> None:
    for name in ("Class10", "Class2", "Class1"):
        class_root = tmp_path / name
        _write_split(class_root, "Train", [("0001", 0, f"{name}-train".encode())])
        _write_split(class_root, "Test", [("0001", 0, f"{name}-test".encode())])

    assert list_categories(tmp_path) == ["Class1", "Class2", "Class10"]


def test_dagm_episode_uses_train_support_and_full_test_holdout(tmp_path: Path) -> None:
    class_root = tmp_path / "Class1"
    _write_split(
        class_root,
        "Train",
        [
            ("1001", 0, b"train-normal-1"),
            ("1002", 0, b"train-normal-2"),
            ("1003", 0, b"train-normal-3"),
            ("1010", 1, b"train-defect-1"),
            ("1011", 1, b"train-defect-2"),
        ],
    )
    _write_split(
        class_root,
        "Test",
        [
            ("0001", 0, b"test-normal-1"),
            ("0002", 0, b"test-normal-2"),
            ("0010", 1, b"test-defect-1"),
            ("0011", 1, b"test-defect-2"),
        ],
    )

    episode = build_episode(
        tmp_path,
        "Class1",
        normal_count=2,
        defect_count=1,
        seed=7,
        minimum_holdout=4,
    )

    assert len(episode.dataset.normal) == 2
    assert len(episode.dataset.defect) == 1
    assert all("Train" in sample.image_path.parts for sample in episode.dataset.samples)
    assert episode.independent_test_count == 4
    assert episode.test_normal_count == 2
    assert episode.test_defect_count == 2

    test_rows = load_dagm_category_test(
        tmp_path,
        "Class1",
        excluded_hashes={sample.sha256 for sample in episode.dataset.samples},
    )
    assert len(test_rows) == 4
    assert [row.label for row in test_rows] == [0, 0, 1, 1]
    assert all("Test" in row.path.parts for row in test_rows)


def test_dagm_test_loader_rejects_support_content_leakage(tmp_path: Path) -> None:
    class_root = tmp_path / "Class1"
    _write_split(class_root, "Test", [("0001", 0, b"shared-content")])
    row = read_split_records(tmp_path, "Class1", "Test")[0]

    digest = hashlib.sha256(row.image_path.read_bytes()).hexdigest()
    with pytest.raises(PermissionError, match="leakage"):
        load_dagm_category_test(tmp_path, "Class1", excluded_hashes={digest})


def test_default_configs_resolve_dagm_from_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_training_assets,
) -> None:
    monkeypatch.chdir(tmp_path)
    training = load_config(ROOT / "configs" / "train.yaml")
    evaluation = load_yaml(ROOT / "configs" / "test.yaml")

    assert training.dataset == "dagm2007"
    assert training.data_root == (ROOT / "data" / "dagm2007").resolve()
    assert training.output_root == (ROOT / "outputs" / "dagm2007").resolve()
    assert resolve_path(evaluation, str(evaluation["data_root"])) == (
        ROOT / "data" / "dagm2007"
    ).resolve()
    assert resolve_path(evaluation, str(evaluation["active_models_pointer"])) == (
        ROOT / "outputs" / "dagm2007" / "active_models.yaml"
    ).resolve()


def test_historical_config_without_dataset_remains_mvtec(
    require_training_assets,
) -> None:
    config = load_config(
        ROOT / "configs" / "train_calibration_fusion_v2.yaml"
    )
    assert config.dataset == "mvtec_ad"
    assert config.data_root == (ROOT / "data" / "mvtec_ad").resolve()


def test_unicode_image_path_is_decoded(tmp_path: Path) -> None:
    directory = tmp_path / "中文目录"
    directory.mkdir()
    path = directory / "样本.PNG"
    bgr = np.zeros((5, 7, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    path.write_bytes(encoded.tobytes())

    rgb = read_rgb(path)

    assert rgb.shape == (5, 7, 3)
    assert np.all(rgb[:, :, 0] == 255)


def test_legacy_output_path_maps_to_unified_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "outputs" / "dagm2007" / "model_bundle.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("schema_version: 3\n", encoding="utf-8")
    legacy = (
        Path(tmp_path.anchor)
        / "obsolete"
        / "AOI_train"
        / "outputs"
        / "dagm2007"
        / "model_bundle.yaml"
    )
    monkeypatch.setattr(artifact_paths, "PROJECT_ROOT", tmp_path)

    assert artifact_paths.resolve_artifact(legacy) == target.resolve()
