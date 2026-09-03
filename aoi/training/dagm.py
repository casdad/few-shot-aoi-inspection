from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import IMAGE_EXTENSIONS, Sample, TrainingDataset
from .manifests import sha256_file


@dataclass(frozen=True)
class DAGMRecord:
    record_id: str
    image_path: Path
    label: int
    mask_path: Path | None


@dataclass(frozen=True)
class DAGMEpisode:
    category: str
    dataset: TrainingDataset
    independent_test_count: int
    test_normal_count: int
    test_defect_count: int
    defect_types: tuple[str, ...]


def _category_key(name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"class(\d+)", name, flags=re.IGNORECASE)
    return (0, int(match.group(1))) if match else (1, name.casefold())


def list_categories(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"DAGM2007 root not found: {root}")
    categories = [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "Train" / "Label" / "Labels.txt").is_file()
        and (path / "Test" / "Label" / "Labels.txt").is_file()
    ]
    categories.sort(key=_category_key)
    if not categories:
        raise ValueError(
            f"No DAGM2007 classes found under {root}. Expected "
            "root/ClassN/{Train,Test}/Label/Labels.txt."
        )
    return categories


def read_split_records(root: Path, category: str, split: str) -> list[DAGMRecord]:
    if split not in {"Train", "Test"}:
        raise ValueError(f"DAGM2007 split must be Train or Test, got {split!r}")
    split_root = root / category / split
    manifest = split_root / "Label" / "Labels.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"DAGM2007 manifest missing: {manifest}")

    lines = [line.strip() for line in manifest.read_text(encoding="utf-8-sig").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 2:
        raise ValueError(f"DAGM2007 manifest has no samples: {manifest}")
    try:
        int(lines[0])
    except ValueError as exc:
        raise ValueError(f"invalid DAGM2007 class header in {manifest}: {lines[0]!r}") from exc

    records: list[DAGMRecord] = []
    seen_names: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        columns = line.split()
        if len(columns) < 3:
            raise ValueError(
                f"invalid DAGM2007 row at {manifest}:{line_number}: {line!r}"
            )
        record_id, raw_label, image_name = columns[:3]
        if Path(image_name).name != image_name:
            raise ValueError(f"unsafe DAGM2007 image name at {manifest}:{line_number}")
        try:
            label = int(raw_label)
        except ValueError as exc:
            raise ValueError(
                f"invalid DAGM2007 label at {manifest}:{line_number}: {raw_label!r}"
            ) from exc
        if label not in {0, 1}:
            raise ValueError(
                f"DAGM2007 label must be 0 or 1 at {manifest}:{line_number}"
            )

        folded_name = image_name.casefold()
        if folded_name in seen_names:
            raise ValueError(f"duplicate DAGM2007 image row in {manifest}: {image_name}")
        seen_names.add(folded_name)
        image_path = (split_root / image_name).resolve(strict=True)

        mask_path: Path | None = None
        if label == 1:
            mask_name = columns[4] if len(columns) >= 5 else ""
            if not mask_name or mask_name == "0":
                mask_name = f"{Path(image_name).stem}_label.PNG"
            if Path(mask_name).name != mask_name:
                raise ValueError(f"unsafe DAGM2007 mask name at {manifest}:{line_number}")
            mask_path = (split_root / "Label" / mask_name).resolve(strict=True)

        records.append(
            DAGMRecord(
                record_id=record_id,
                image_path=image_path,
                label=label,
                mask_path=mask_path,
            )
        )

    actual_images = {
        path.name.casefold()
        for path in split_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if actual_images != seen_names:
        missing = sorted(seen_names - actual_images)
        unlisted = sorted(actual_images - seen_names)
        raise ValueError(
            f"DAGM2007 manifest/image mismatch for {category}/{split}: "
            f"missing={missing[:5]}, unlisted={unlisted[:5]}"
        )
    return records


def _rank(record: DAGMRecord, seed: int, category: str) -> str:
    identity = f"{seed}|dagm2007|{category}|Train|{record.image_path.name}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_episode(
    root: Path,
    category: str,
    *,
    normal_count: int,
    defect_count: int,
    seed: int,
    minimum_holdout: int,
) -> DAGMEpisode:
    train_records = read_split_records(root, category, "Train")
    test_records = read_split_records(root, category, "Test")
    normals = sorted(
        (record for record in train_records if record.label == 0),
        key=lambda record: _rank(record, seed, category),
    )
    defects = sorted(
        (record for record in train_records if record.label == 1),
        key=lambda record: _rank(record, seed, category),
    )
    if len(normals) < normal_count:
        raise ValueError(
            f"{category}: requires {normal_count} DAGM Train normals, found {len(normals)}"
        )
    if len(defects) < defect_count:
        raise ValueError(
            f"{category}: requires {defect_count} DAGM Train defects, found {len(defects)}"
        )
    if len(test_records) < minimum_holdout:
        raise ValueError(
            f"{category}: DAGM Test has {len(test_records)}, requires >= {minimum_holdout}"
        )

    samples: list[Sample] = []
    seen_hashes: set[str] = set()
    for record in normals[:normal_count]:
        digest = sha256_file(record.image_path)
        if digest in seen_hashes:
            raise ValueError(f"{category}: duplicate support content: {record.image_path}")
        seen_hashes.add(digest)
        samples.append(
            Sample(
                sample_id=f"{category}-normal-{record.record_id}",
                role="normal",
                image_path=record.image_path,
                mask_path=None,
                defect_type="normal",
                sha256=digest,
            )
        )
    for record in defects[:defect_count]:
        if record.mask_path is None:
            raise ValueError(f"{category}: defect mask missing: {record.image_path}")
        digest = sha256_file(record.image_path)
        if digest in seen_hashes:
            raise ValueError(f"{category}: duplicate support content: {record.image_path}")
        seen_hashes.add(digest)
        samples.append(
            Sample(
                sample_id=f"{category}-defect-{record.record_id}",
                role="defect",
                image_path=record.image_path,
                mask_path=record.mask_path,
                defect_type="defect",
                sha256=digest,
            )
        )

    test_normal_count = sum(record.label == 0 for record in test_records)
    test_defect_count = len(test_records) - test_normal_count
    return DAGMEpisode(
        category=category,
        dataset=TrainingDataset(tuple(samples)),
        independent_test_count=len(test_records),
        test_normal_count=test_normal_count,
        test_defect_count=test_defect_count,
        defect_types=("defect",),
    )


def build_eligible_episodes(
    root: Path,
    *,
    categories: tuple[str, ...] | None,
    normal_count: int,
    defect_count: int,
    seed: int,
    minimum_holdout: int,
) -> tuple[list[DAGMEpisode], list[dict[str, str]]]:
    available = list_categories(root)
    requested = available if categories is None else list(categories)
    unknown = sorted(set(requested) - set(available), key=_category_key)
    if unknown:
        raise ValueError(f"Unknown DAGM2007 classes: {unknown}")

    episodes: list[DAGMEpisode] = []
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
        raise ValueError("No DAGM2007 class can satisfy the configured support protocol")
    return episodes, skipped
