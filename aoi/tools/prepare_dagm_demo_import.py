"""Build a UI-ready DAGM2007 demonstration import package.

The 100 normal and 30 defect support images are taken from the immutable
support receipts of the active DAGM models.  Evaluation data consists of the
unused official Train images plus every official Test image, so every class
contains at least 1,000 labeled evaluation images without support leakage.
"""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import yaml

from aoi.custom_manifest import load_custom_test_manifest
from aoi.training.dagm import DAGMRecord, list_categories, read_split_records
from aoi.training.manifests import sha256_file, write_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "dagm2007"
DEFAULT_POINTER = PROJECT_ROOT / "outputs" / "dagm2007" / "active_models.yaml"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "DAGM演示导入包"
EXPECTED_CATEGORIES = tuple(f"Class{index}" for index in range(1, 11))


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a YAML mapping")
    return value


def _load_yaml(path: Path, context: str) -> dict[str, Any]:
    return _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")) or {},
        context,
    )


def _resolve_child(base: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty path")
    raw = Path(value.strip())
    return (raw if raw.is_absolute() else base / raw).resolve(strict=True)


def _record_index(records: list[DAGMRecord]) -> dict[str, DAGMRecord]:
    return {record.image_path.name.casefold(): record for record in records}


def _receipt_records(
    receipt: dict[str, Any],
    train_records: list[DAGMRecord],
    role: str,
    expected_count: int,
) -> list[tuple[DAGMRecord, str]]:
    raw_rows = receipt.get(role)
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_count:
        raise ValueError(
            f"support receipt requires exactly {expected_count} {role} rows"
        )
    by_name = _record_index(train_records)
    expected_label = 0 if role == "normal" else 1
    selected: list[tuple[DAGMRecord, str]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, f"support receipt {role}[{index}]")
        raw_path = row.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise TypeError(f"support receipt {role}[{index}].path is invalid")
        record = by_name.get(Path(raw_path).name.casefold())
        if record is None or record.label != expected_label:
            raise ValueError(
                f"support receipt {role}[{index}] is not a matching Train sample"
            )
        digest = sha256_file(record.image_path)
        recorded_digest = str(row.get("sha256", "")).lower()
        if digest != recorded_digest:
            raise ValueError(f"support receipt hash mismatch: {record.image_path}")
        if digest in seen:
            raise ValueError(f"duplicate support content: {record.image_path}")
        if expected_label == 1 and record.mask_path is None:
            raise ValueError(f"defect support mask is missing: {record.image_path}")
        seen.add(digest)
        selected.append((record, digest))
    return selected


def _materialize(source: Path, destination: Path, *, copy_files: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def _readme() -> str:
    return """# DAGM 十类演示导入包

此目录用于 AOI 质量检测工作台的手动导入演示，不是生产默认数据，也不能作为新的独立竞赛测试证据。

每个 `ClassN` 的导入步骤：

1. 在工作台点击“新建产品”，建议命名为 `DAGM_ClassN`。
2. 标注模式选择“像素掩码”时，依次选择：
   - `01_正常样本_100`
   - `02_缺陷样本_30`
   - `03_像素掩码_30`
3. 如只演示图像级闭环，可选择“图像标签”，只导入前两个目录。
4. “独立评估”选择 `05_评估清单_1000plus.yaml`。
5. 不要勾选“本测试集为模型冻结后首次评测”。这些是公开演示数据，不能据此声明新的独立测试成绩。

数据边界：

- 100+30 适配集严格复用现有冻结 DAGM 模型的 `support_receipt.yaml`，没有重新随机抽样。
- 评估目录由未进入适配集的官方 Train 图片和全部官方 Test 图片组成。
- 清单包含 SHA-256，生成时和导入时都会拒绝适配/评估内容重叠。
- `04_评估图片_1000plus` 下按正常/缺陷分开，文件名前缀标明原始 Train 或 Test 来源。
- `06_评估缺陷掩码` 仅用于人工核对，不是工作台评估 YAML 的必填项。
"""


def _prepare_category(
    *,
    source_root: Path,
    pointer_root: Path,
    bundle_value: Any,
    category: str,
    category_destination: Path,
    copy_files: bool,
) -> dict[str, Any]:
    bundle_path = _resolve_child(pointer_root, bundle_value, f"{category} bundle")
    bundle = _load_yaml(bundle_path, f"{category} model bundle")
    receipt_path = _resolve_child(
        bundle_path.parent,
        bundle.get("support_receipt"),
        f"{category} support receipt",
    )
    receipt = _load_yaml(receipt_path, f"{category} support receipt")
    if receipt.get("category") != category:
        raise ValueError(f"support receipt category mismatch: {receipt_path}")
    if receipt.get("normal_count") != 100 or receipt.get("defect_count") != 30:
        raise ValueError(f"{category} support receipt is not the 100+30 protocol")

    train_records = read_split_records(source_root, category, "Train")
    test_records = read_split_records(source_root, category, "Test")
    normal_support = _receipt_records(receipt, train_records, "normal", 100)
    defect_support = _receipt_records(receipt, train_records, "defect", 30)
    support = normal_support + defect_support
    support_hashes = {digest for _, digest in support}
    if len(support_hashes) != 130:
        raise ValueError(f"{category} support set has duplicate content")

    normal_dir = category_destination / "01_正常样本_100"
    defect_dir = category_destination / "02_缺陷样本_30"
    mask_dir = category_destination / "03_像素掩码_30"
    evaluation_root = category_destination / "04_评估图片_1000plus"
    evaluation_masks = category_destination / "06_评估缺陷掩码"
    materialization = {"hardlink": 0, "copy": 0, "copy_fallback": 0}

    for record, _ in normal_support:
        mode = _materialize(
            record.image_path,
            normal_dir / record.image_path.name,
            copy_files=copy_files,
        )
        materialization[mode] += 1
    for record, _ in defect_support:
        mode = _materialize(
            record.image_path,
            defect_dir / record.image_path.name,
            copy_files=copy_files,
        )
        materialization[mode] += 1
        assert record.mask_path is not None
        mode = _materialize(
            record.mask_path,
            mask_dir / record.mask_path.name,
            copy_files=copy_files,
        )
        materialization[mode] += 1

    evaluation_rows: list[dict[str, Any]] = []
    evaluation_hashes: set[str] = set()
    evaluation_normal = 0
    evaluation_defect = 0
    unused_train = 0
    official_test = 0
    for split, records in (("Train", train_records), ("Test", test_records)):
        for record in records:
            digest = sha256_file(record.image_path)
            if digest in support_hashes:
                if split != "Train":
                    raise PermissionError(
                        f"{category}: official Test overlaps support: {record.image_path}"
                    )
                continue
            if digest in evaluation_hashes:
                raise ValueError(
                    f"{category}: duplicate evaluation content: {record.image_path}"
                )
            evaluation_hashes.add(digest)
            label_dir = "normal" if record.label == 0 else "defect"
            output_name = f"{split}_{record.image_path.name}"
            relative_image = Path(label_dir) / output_name
            mode = _materialize(
                record.image_path,
                evaluation_root / relative_image,
                copy_files=copy_files,
            )
            materialization[mode] += 1
            row: dict[str, Any] = {
                "id": f"{category}-{split}-{record.record_id}",
                "image": relative_image.as_posix(),
                "label": record.label,
                "sha256": digest,
            }
            if record.label == 1:
                row["defect_type"] = "defect"
                evaluation_defect += 1
                if record.mask_path is not None:
                    mode = _materialize(
                        record.mask_path,
                        evaluation_masks / f"{split}_{record.mask_path.name}",
                        copy_files=copy_files,
                    )
                    materialization[mode] += 1
            else:
                evaluation_normal += 1
            evaluation_rows.append(row)
            if split == "Train":
                unused_train += 1
            else:
                official_test += 1

    if len(evaluation_rows) < 1000:
        raise ValueError(
            f"{category}: expected 1000+ evaluation images, got {len(evaluation_rows)}"
        )
    manifest_path = category_destination / "05_评估清单_1000plus.yaml"
    write_yaml(
        manifest_path,
        {
            "schema_version": 1,
            "image_root": "04_评估图片_1000plus",
            "samples": evaluation_rows,
        },
    )
    loaded = load_custom_test_manifest(
        manifest_path,
        minimum_samples=1000,
        excluded_hashes=support_hashes,
    )
    if len(loaded) != len(evaluation_rows):
        raise RuntimeError(f"{category}: generated evaluation manifest count changed")

    summary = {
        "category": category,
        "support_selection": "frozen_model_support_receipt",
        "support_receipt": str(receipt_path),
        "support": {"normal": 100, "defect": 30, "masks": 30},
        "evaluation": {
            "total": len(evaluation_rows),
            "normal": evaluation_normal,
            "defect": evaluation_defect,
            "unused_official_train": unused_train,
            "official_test": official_test,
            "fresh_independent_test": False,
        },
        "support_evaluation_hash_overlap": 0,
        "materialization": materialization,
    }
    write_yaml(category_destination / "划分摘要.yaml", summary)
    return summary


def prepare_package(
    source_root: Path,
    pointer_path: Path,
    destination: Path,
    *,
    copy_files: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    pointer_path = pointer_path.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    categories = tuple(list_categories(source_root))
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            f"complete DAGM2007 Class1-Class10 required; found {categories}"
        )
    pointer = _load_yaml(pointer_path, "active DAGM model pointer")
    if pointer.get("dataset") != "dagm2007":
        raise ValueError("active model pointer is not DAGM2007")
    models = _mapping(pointer.get("models"), "active DAGM models")
    if tuple(sorted(models, key=lambda name: int(name[5:]))) != EXPECTED_CATEGORIES:
        raise ValueError("active model pointer must contain exactly Class1-Class10")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        summaries = []
        for category in EXPECTED_CATEGORIES:
            summaries.append(
                _prepare_category(
                    source_root=source_root,
                    pointer_root=pointer_path.parent,
                    bundle_value=models[category],
                    category=category,
                    category_destination=staging / category,
                    copy_files=copy_files,
                )
            )
        (staging / "导入说明.md").write_text(_readme(), encoding="utf-8")
        result = {
            "schema_version": 1,
            "purpose": "DAGM2007 UI manual import demonstration",
            "source_root": str(source_root),
            "active_models_pointer": str(pointer_path),
            "support_protocol": "per class 100 normal + 30 defect",
            "evaluation_policy": (
                "unused official Train plus all official Test; public demo only"
            ),
            "fresh_independent_test": False,
            "classes": summaries,
            "totals": {
                "classes": len(summaries),
                "support_images": sum(
                    row["support"]["normal"] + row["support"]["defect"]
                    for row in summaries
                ),
                "support_masks": sum(row["support"]["masks"] for row in summaries),
                "evaluation_images": sum(row["evaluation"]["total"] for row in summaries),
                "evaluation_masks": sum(row["evaluation"]["defect"] for row in summaries),
            },
        }
        write_yaml(staging / "总划分摘要.yaml", result)
        staging.rename(destination)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Class1-Class10 100+30 and 1000+ UI import folders."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy image bytes instead of using same-volume hard links",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = prepare_package(
        args.source_root,
        args.pointer,
        args.destination,
        copy_files=args.copy,
    )
    print(yaml.safe_dump(result["totals"], sort_keys=False, allow_unicode=True))
    print(f"destination: {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
