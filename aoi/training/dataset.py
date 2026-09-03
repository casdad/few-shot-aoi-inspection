from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from aoi.images import read_mask, read_rgb

from .manifests import sha256_file

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    role: str
    image_path: Path
    mask_path: Path | None
    defect_type: str
    sha256: str


@dataclass(frozen=True)
class TrainingDataset:
    samples: tuple[Sample, ...]

    @property
    def normal(self) -> tuple[Sample, ...]:
        return tuple(s for s in self.samples if s.role == "normal")

    @property
    def defect(self) -> tuple[Sample, ...]:
        return tuple(s for s in self.samples if s.role == "defect")


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(p.resolve() for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _match_mask(image: Path, masks: list[Path]) -> Path:
    by_stem = {p.stem.casefold(): p for p in masks}
    candidates = [image.stem, f"{image.stem}_mask", f"{image.stem}_label"]
    for stem in candidates:
        match = by_stem.get(stem.casefold())
        if match is not None:
            return match
    raise FileNotFoundError(f"no mask found for defect image: {image.name}")


def load_support_dataset(
    normal_dir: Path,
    defect_dir: Path,
    mask_dir: Path | None = None,
) -> TrainingDataset:
    normals = _images(normal_dir)
    defects = _images(defect_dir)
    masks = [] if mask_dir is None else _images(mask_dir)
    if len(normals) != 100 or len(defects) != 30:
        raise ValueError(f"original scheme requires exactly 100 normal + 30 defect; got {len(normals)}+{len(defects)}")
    if mask_dir is not None and len(masks) < 30:
        raise ValueError(f"30 defect masks are required; found {len(masks)}")
    samples: list[Sample] = []
    seen_hashes: set[str] = set()
    for index, path in enumerate(normals, 1):
        digest = sha256_file(path)
        if digest in seen_hashes:
            raise ValueError(f"duplicate image content in support set: {path}")
        seen_hashes.add(digest)
        samples.append(Sample(f"normal-{index:03d}", "normal", path, None, "normal", digest))
    for index, path in enumerate(defects, 1):
        digest = sha256_file(path)
        if digest in seen_hashes:
            raise ValueError(f"duplicate image content in support set: {path}")
        seen_hashes.add(digest)
        mask = None if mask_dir is None else _match_mask(path, masks)
        samples.append(Sample(f"defect-{index:03d}", "defect", path, mask, path.parent.name, digest))
    return TrainingDataset(tuple(samples))


def dataset_arrays(
    dataset: TrainingDataset,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray] | None]:
    normals = [read_rgb(sample.image_path) for sample in dataset.normal]
    defects = [read_rgb(sample.image_path) for sample in dataset.defect]
    mask_paths = [sample.mask_path for sample in dataset.defect]
    if not any(path is not None for path in mask_paths):
        return normals, defects, None
    if not all(path is not None for path in mask_paths):
        raise RuntimeError("defect masks must be supplied for either all or none of the 30 images")
    masks = [
        read_mask(path, image.shape[:2])
        for path, image in zip(mask_paths, defects, strict=True)
        if path is not None
    ]
    return normals, defects, masks
