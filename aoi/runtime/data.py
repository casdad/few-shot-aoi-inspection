from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from aoi.images import read_rgb as read_rgb

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    path: Path
    label: int
    defect_type: str
    category: str
    sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        p.resolve() for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_mvtec_category_holdout(
    root: Path,
    category: str,
    *,
    excluded_hashes: set[str],
) -> list[EvalSample]:
    category_root = root / category
    test_root = category_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"MVTec test directory missing: {test_root}")

    rows: list[EvalSample] = []
    seen: set[str] = set()
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        label = 0 if defect_dir.name.casefold() == "good" else 1
        for path in image_files(defect_dir):
            digest = sha256_file(path)
            if digest in excluded_hashes:
                # This is one of the 30 labeled defect support images. It is intentionally excluded.
                continue
            if digest in seen:
                raise ValueError(f"duplicate MVTec holdout content in {category}: {path}")
            seen.add(digest)
            rows.append(
                EvalSample(
                    sample_id=f"{category}/{defect_dir.name}/{path.name}",
                    path=path,
                    label=label,
                    defect_type=defect_dir.name,
                    category=category,
                    sha256=digest,
                )
            )
    if not rows:
        raise ValueError(f"no independent MVTec holdout images remain for {category}")
    leaked = [row for row in rows if row.sha256 in excluded_hashes]
    if leaked:
        raise PermissionError(f"support/test content leakage detected for {category}: {len(leaked)}")
    return rows


def load_mvtec_standard_test(root: Path) -> list[EvalSample]:
    rows: list[EvalSample] = []
    for category in sorted(p.name for p in root.iterdir() if p.is_dir()):
        category_root = root / category
        if not (category_root / "test").is_dir():
            continue
        rows.extend(load_mvtec_category_holdout(root, category, excluded_hashes=set()))
    if not rows:
        raise ValueError(f"no MVTec test images found under {root}")
    return rows


def _dagm_manifest_rows(
    root: Path,
    category: str,
    split: str,
) -> list[tuple[str, Path, int]]:
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

    rows: list[tuple[str, Path, int]] = []
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
        if label == 1:
            mask_name = columns[4] if len(columns) >= 5 else ""
            if not mask_name or mask_name == "0":
                mask_name = f"{Path(image_name).stem}_label.PNG"
            if Path(mask_name).name != mask_name:
                raise ValueError(f"unsafe DAGM2007 mask name at {manifest}:{line_number}")
            (split_root / "Label" / mask_name).resolve(strict=True)
        rows.append((record_id, image_path, label))

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
    return rows


def load_dagm_category_test(
    root: Path,
    category: str,
    *,
    excluded_hashes: set[str],
) -> list[EvalSample]:
    rows: list[EvalSample] = []
    seen: set[str] = set()
    for record_id, path, label in _dagm_manifest_rows(root, category, "Test"):
        digest = sha256_file(path)
        if digest in excluded_hashes:
            raise PermissionError(
                f"DAGM2007 Train-support/Test leakage detected for {category}: {path}"
            )
        if digest in seen:
            raise ValueError(f"duplicate DAGM2007 Test content in {category}: {path}")
        seen.add(digest)
        defect_type = "defect" if label else "normal"
        rows.append(
            EvalSample(
                sample_id=f"{category}/Test/{record_id}",
                path=path,
                label=label,
                defect_type=defect_type,
                category=category,
                sha256=digest,
            )
        )
    if not rows:
        raise ValueError(f"no DAGM2007 Test images found for {category}")
    return rows


def load_dagm_standard_test(root: Path) -> list[EvalSample]:
    rows: list[EvalSample] = []
    for category_root in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = category_root / "Test" / "Label" / "Labels.txt"
        if manifest.is_file():
            rows.extend(
                load_dagm_category_test(
                    root,
                    category_root.name,
                    excluded_hashes=set(),
                )
            )
    if not rows:
        raise ValueError(f"no DAGM2007 Test images found under {root}")
    return rows
