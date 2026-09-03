from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .dataset import IMAGE_EXTENSIONS, Sample, TrainingDataset
from .manifests import sha256_file


@dataclass(frozen=True)
class MVTecEpisode:
    category: str
    dataset: TrainingDataset
    independent_test_count: int
    test_normal_count: int
    test_defect_count: int
    defect_types: tuple[str, ...]


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p.resolve() for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _rank(path: Path, root: Path, seed: int, category: str) -> str:
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        # ``_images`` resolves paths, so a category exposed through a Windows
        # junction can point outside the logical dataset root.  Rebuild the
        # same category-relative identity without depending on the physical
        # storage location.
        category_root = (root / category).resolve()
        try:
            rel_path = Path(category) / path.relative_to(category_root)
        except ValueError as exc:
            raise ValueError(
                f"MVTec image is outside category root {category_root}: {path}"
            ) from exc
    rel = rel_path.as_posix()
    return hashlib.sha256(f"{seed}|{category}|{rel}".encode("utf-8")).hexdigest()


def list_categories(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec AD root not found: {root}")
    result = []
    for p in sorted(v for v in root.iterdir() if v.is_dir()):
        if (p / "train" / "good").is_dir() and (p / "test").is_dir():
            result.append(p.name)
    if not result:
        raise ValueError(
            f"No MVTec categories found under {root}. Expected root/category/train/good and root/category/test."
        )
    return result


def _mask_for(root: Path, category: str, defect_type: str, image: Path) -> Path:
    gt = root / category / "ground_truth" / defect_type
    candidates = [
        gt / f"{image.stem}_mask.png",
        gt / f"{image.stem}.png",
        gt / f"{image.stem}_mask.tif",
        gt / f"{image.stem}.tif",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    # Fallback for non-PNG variants.
    for p in _images(gt):
        stem = p.stem.casefold()
        if stem in {image.stem.casefold(), f"{image.stem}_mask".casefold()}:
            return p
    raise FileNotFoundError(
        f"MVTec mask not found for {category}/{defect_type}/{image.name} under {gt}"
    )


def _stratified_defects(
    root: Path,
    category: str,
    count: int,
    seed: int,
) -> list[tuple[Path, str, Path]]:
    test_root = root / category / "test"
    groups: dict[str, list[Path]] = {}
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir() and p.name.casefold() != "good"):
        values = _images(defect_dir)
        values.sort(key=lambda p: _rank(p, root, seed, category))
        if values:
            groups[defect_dir.name] = values
    total = sum(len(v) for v in groups.values())
    if total < count:
        raise ValueError(f"{category}: requires {count} defect support images, found {total}")
    selected: list[tuple[Path, str, Path]] = []
    offsets = {name: 0 for name in groups}
    names = sorted(groups)
    while len(selected) < count:
        progressed = False
        for name in names:
            i = offsets[name]
            if i >= len(groups[name]):
                continue
            image = groups[name][i]
            offsets[name] = i + 1
            selected.append((image, name, _mask_for(root, category, name, image)))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(f"{category}: failed to select {count} defect samples")
    return selected


def build_episode(
    root: Path,
    category: str,
    *,
    normal_count: int,
    defect_count: int,
    seed: int,
    minimum_holdout: int,
) -> MVTecEpisode:
    category_root = root / category
    normals = _images(category_root / "train" / "good")
    if len(normals) < normal_count:
        raise ValueError(f"{category}: requires {normal_count} train/good images, found {len(normals)}")
    normals.sort(key=lambda p: _rank(p, root, seed, category))
    support_normals = normals[:normal_count]
    support_defects = _stratified_defects(root, category, defect_count, seed)

    support_defect_paths = {item[0].resolve() for item in support_defects}
    test_good = _images(category_root / "test" / "good")
    all_test_defects: list[tuple[Path, str]] = []
    for defect_dir in sorted(p for p in (category_root / "test").iterdir() if p.is_dir() and p.name.casefold() != "good"):
        for p in _images(defect_dir):
            all_test_defects.append((p, defect_dir.name))
    remaining_defects = [(p, t) for p, t in all_test_defects if p.resolve() not in support_defect_paths]
    holdout = len(test_good) + len(remaining_defects)
    if holdout < minimum_holdout:
        raise ValueError(f"{category}: independent holdout has {holdout}, requires >= {minimum_holdout}")

    samples: list[Sample] = []
    seen: set[str] = set()
    for i, path in enumerate(support_normals, 1):
        digest = sha256_file(path)
        if digest in seen:
            raise ValueError(f"{category}: duplicate support content: {path}")
        seen.add(digest)
        samples.append(Sample(f"{category}-normal-{i:03d}", "normal", path, None, "normal", digest))
    for i, (path, defect_type, mask) in enumerate(support_defects, 1):
        digest = sha256_file(path)
        if digest in seen:
            raise ValueError(f"{category}: duplicate support content: {path}")
        seen.add(digest)
        samples.append(Sample(f"{category}-defect-{i:03d}", "defect", path, mask, defect_type, digest))

    return MVTecEpisode(
        category=category,
        dataset=TrainingDataset(tuple(samples)),
        independent_test_count=holdout,
        test_normal_count=len(test_good),
        test_defect_count=len(remaining_defects),
        defect_types=tuple(sorted({t for _, t in all_test_defects})),
    )


def build_eligible_episodes(
    root: Path,
    *,
    categories: tuple[str, ...] | None,
    normal_count: int,
    defect_count: int,
    seed: int,
    minimum_holdout: int,
) -> tuple[list[MVTecEpisode], list[dict[str, str]]]:
    available = list_categories(root)
    requested = available if categories is None else list(categories)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown MVTec categories: {unknown}")

    episodes: list[MVTecEpisode] = []
    skipped: list[dict[str, str]] = []
    for category in requested:
        try:
            episodes.append(
                build_episode(
                    root,
                    category,
                    normal_count=normal_count,
                    defect_count=defect_count,
                    seed=seed,
                    minimum_holdout=minimum_holdout,
                )
            )
        except (ValueError, FileNotFoundError) as exc:
            if categories is not None:
                raise
            skipped.append({"category": category, "reason": str(exc)})
    if not episodes:
        raise ValueError("No MVTec category can satisfy the configured 100+30 episode protocol")
    return episodes, skipped
