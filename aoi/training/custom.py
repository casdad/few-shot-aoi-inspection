from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset import TrainingDataset, load_support_dataset
from .manifests import sha256_file


@dataclass(frozen=True)
class CustomEpisode:
    category: str
    dataset: TrainingDataset
    independent_test_count: int
    test_normal_count: int
    test_defect_count: int
    defect_types: tuple[str, ...]
    test_root: Path | None
    annotation_mode: str
    test_manifest: Path | None = None


def build_custom_episode(
    *, normal_dir: str | Path, defect_dir: str | Path, mask_dir: str | Path | None = None,
    test_dir: str | Path | None = None, category: str = "custom",
) -> CustomEpisode:
    resolved_mask_dir = None if mask_dir is None else Path(mask_dir).resolve(strict=True)
    dataset = load_support_dataset(
        Path(normal_dir).resolve(strict=True),
        Path(defect_dir).resolve(strict=True),
        resolved_mask_dir,
    )
    support_hashes = {sample.sha256 for sample in dataset.samples}
    test_root = None if test_dir is None else Path(test_dir).resolve(strict=True)
    test_paths = []
    if test_root is not None:
        from .dataset import IMAGE_EXTENSIONS

        test_paths = sorted(
            path for path in test_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        overlap = [path for path in test_paths if sha256_file(path) in support_hashes]
        if overlap:
            raise ValueError(f"custom test/support content overlap detected: {overlap[0]}")
    return CustomEpisode(
        category=category,
        dataset=dataset,
        independent_test_count=len(test_paths),
        test_normal_count=0,
        test_defect_count=0,
        defect_types=tuple(sorted({sample.defect_type for sample in dataset.defect})),
        test_root=test_root,
        annotation_mode="image_label" if resolved_mask_dir is None else "pixel_mask",
    )
