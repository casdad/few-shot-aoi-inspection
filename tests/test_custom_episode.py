from __future__ import annotations

from pathlib import Path

import pytest

from aoi.training.custom import build_custom_episode


def _support_tree(root: Path) -> tuple[Path, Path, Path]:
    normal = root / "normal"
    defect = root / "defect"
    masks = root / "masks"
    normal.mkdir()
    defect.mkdir()
    masks.mkdir()
    for index in range(100):
        (normal / f"normal-{index:03d}.png").write_bytes(f"normal-{index}".encode())
    for index in range(30):
        (defect / f"defect-{index:03d}.png").write_bytes(f"defect-{index}".encode())
        (masks / f"defect-{index:03d}_mask.png").write_bytes(f"mask-{index}".encode())
    return normal, defect, masks


def test_custom_episode_requires_exact_support_and_tracks_holdout(tmp_path: Path) -> None:
    normal, defect, masks = _support_tree(tmp_path)
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for index in range(3):
        (test_dir / f"test-{index}.png").write_bytes(f"test-{index}".encode())
    episode = build_custom_episode(
        normal_dir=normal, defect_dir=defect, mask_dir=masks, test_dir=test_dir
    )
    assert len(episode.dataset.normal) == 100
    assert len(episode.dataset.defect) == 30
    assert episode.independent_test_count == 3
    assert episode.annotation_mode == "pixel_mask"
    assert all(sample.mask_path is not None for sample in episode.dataset.defect)


def test_custom_episode_accepts_image_labels_without_masks(tmp_path: Path) -> None:
    normal, defect, _masks = _support_tree(tmp_path)
    episode = build_custom_episode(normal_dir=normal, defect_dir=defect)
    assert len(episode.dataset.normal) == 100
    assert len(episode.dataset.defect) == 30
    assert episode.annotation_mode == "image_label"
    assert all(sample.mask_path is None for sample in episode.dataset.defect)


def test_custom_episode_rejects_support_test_hash_overlap(tmp_path: Path) -> None:
    normal, defect, masks = _support_tree(tmp_path)
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "leak.png").write_bytes((normal / "normal-000.png").read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        build_custom_episode(
            normal_dir=normal, defect_dir=defect, mask_dir=masks, test_dir=test_dir
        )
