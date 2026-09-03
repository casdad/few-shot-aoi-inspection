"""Build a DAGM-style manual-import package from eligible MVTec AD classes.

The source dataset is never modified.  Each eligible category receives an
auditable 100-normal + 30-defect support split and the remaining official test
images as a disjoint holdout.  Files are hard-linked by default (with a copy
fallback) so the package does not unnecessarily duplicate the dataset bytes.
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
from aoi.training.dataset import IMAGE_EXTENSIONS, Sample
from aoi.training.manifests import sha256_file, write_yaml
from aoi.training.mvtec import MVTecEpisode, build_eligible_episodes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "MVTec泛化导入包"
DEFAULT_NORMAL_COUNT = 100
DEFAULT_DEFECT_COUNT = 30
DEFAULT_SEED = 20260808


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _mask_for(
    source_root: Path,
    category: str,
    defect_type: str,
    image_path: Path,
) -> Path:
    mask_root = source_root / category / "ground_truth" / defect_type
    candidates = (
        mask_root / f"{image_path.stem}_mask.png",
        mask_root / f"{image_path.stem}.png",
        mask_root / f"{image_path.stem}_mask.tif",
        mask_root / f"{image_path.stem}.tif",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    expected_stems = {
        image_path.stem.casefold(),
        f"{image_path.stem}_mask".casefold(),
    }
    for candidate in _images(mask_root):
        if candidate.stem.casefold() in expected_stems:
            return candidate
    raise FileNotFoundError(
        f"MVTec mask not found for {category}/{defect_type}/{image_path.name}"
    )


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


def _support_name(index: int, sample: Sample) -> str:
    if sample.role == "normal":
        return f"{index:03d}__{sample.image_path.name}"
    return f"{index:03d}__{sample.defect_type}__{sample.image_path.name}"


def _readme(
    *,
    categories: list[str],
    skipped: list[dict[str, str]],
    pooled_holdout: int,
    seed: int,
) -> str:
    skipped_text = "无"
    if skipped:
        skipped_text = "、".join(
            f"`{row['category']}`（{row['reason']}）" for row in skipped
        )
    category_text = "、".join(f"`{category}`" for category in categories)
    return f"""# MVTec 十四类泛化演示导入包

此目录按 DAGM 演示包的编号结构整理，用于 AOI 质量检测工作台的逐类自学习/泛化演示。原始 `data/mvtec_ad` 未被移动或删除。

## 类别和划分

- 可用类别（{len(categories)} 类）：{category_text}
- 未纳入：{skipped_text}
- 每类适配集：100 张正常 + 30 张缺陷 + 30 张配对掩码
- 随机种子：`{seed}`
- 14 类留出评估合计：{pooled_holdout} 张

## 每类导入方法

1. 在工作台点击“新建产品”，建议产品名使用 `MVTec_类别名`。
2. 选择“像素掩码”模式，依次导入：
   - `01_正常样本_100`
   - `02_缺陷样本_30`
   - `03_像素掩码_30`
3. 如只演示图像级闭环，可选“图像标签”，只导入前两个目录。
4. `04_评估图片_留出集`、`05_评估清单_留出集.yaml` 和 `06_评估缺陷掩码` 用于核对该类的未参与适配数据。

## 重要评估边界

- 工作台的“最终独立评估”强制单个清单至少 1000 张。MVTec 的 {pooled_holdout} 张是 14 类汇总，不是任一单类的 1000+，因此不应将单类留出清单冒充为最终独立测试。
- 这是“同一算法/参数流程，按类别分别建模”的流程泛化，不是用一个产品模型混合检测 14 类。
- 30 张缺陷支持图来自 MVTec 官方 `test`，所以这是竞赛式 100+30 自定义协议，不是标准的“官方 test 完全未见” MVTec 评测。
- 数据来自公开 MVTec AD，只能作为开发/演示泛化证据，不是新的私有独立竞赛成绩。

## 文件和安全

- 默认使用 NTFS 硬链接节省空间；若硬链接不可用，自动回退为复制。
- 可删除整个导入包而不会删除原数据，但不要就地编辑硬链接图片内容，因为它与原图共享同一份数据字节。
- 每类 `划分摘要.yaml` 记录数量、缺陷类型和哈希隔离结果；根目录 `泛化评估索引.yaml` 列出所有单类留出清单。
"""


def _prepare_category(
    *,
    source_root: Path,
    episode: MVTecEpisode,
    category_destination: Path,
    seed: int,
    copy_files: bool,
) -> dict[str, Any]:
    category = episode.category
    normal_dir = category_destination / "01_正常样本_100"
    defect_dir = category_destination / "02_缺陷样本_30"
    mask_dir = category_destination / "03_像素掩码_30"
    evaluation_root = category_destination / "04_评估图片_留出集"
    evaluation_masks = category_destination / "06_评估缺陷掩码"
    materialization = {"hardlink": 0, "copy": 0, "copy_fallback": 0}

    support_hashes = {sample.sha256 for sample in episode.dataset.samples}
    if len(support_hashes) != len(episode.dataset.samples):
        raise ValueError(f"{category}: support set contains duplicate content")
    support_defect_paths = {
        sample.image_path.resolve() for sample in episode.dataset.defect
    }

    for index, sample in enumerate(episode.dataset.normal, 1):
        mode = _materialize(
            sample.image_path,
            normal_dir / _support_name(index, sample),
            copy_files=copy_files,
        )
        materialization[mode] += 1
    for index, sample in enumerate(episode.dataset.defect, 1):
        if sample.mask_path is None:
            raise ValueError(f"{category}: support mask is missing: {sample.image_path}")
        output_name = _support_name(index, sample)
        mode = _materialize(
            sample.image_path,
            defect_dir / output_name,
            copy_files=copy_files,
        )
        materialization[mode] += 1
        mode = _materialize(
            sample.mask_path,
            mask_dir / output_name,
            copy_files=copy_files,
        )
        materialization[mode] += 1

    evaluation_rows: list[dict[str, Any]] = []
    evaluation_hashes: set[str] = set()
    evaluation_normal = 0
    evaluation_defect = 0
    test_root = source_root / category / "test"
    test_groups = [
        directory
        for directory in sorted(path for path in test_root.iterdir() if path.is_dir())
    ]
    for group in test_groups:
        is_normal = group.name.casefold() == "good"
        for image_path in _images(group):
            if image_path in support_defect_paths:
                continue
            digest = sha256_file(image_path)
            if digest in support_hashes:
                raise PermissionError(
                    f"{category}: holdout content overlaps 100+30 support: {image_path}"
                )
            if digest in evaluation_hashes:
                raise ValueError(
                    f"{category}: duplicate holdout image content: {image_path}"
                )
            evaluation_hashes.add(digest)
            if is_normal:
                label = 0
                defect_type = "normal"
                output_name = f"good__{image_path.name}"
                label_dir = "normal"
                evaluation_normal += 1
            else:
                label = 1
                defect_type = group.name
                output_name = f"{defect_type}__{image_path.name}"
                label_dir = "defect"
                evaluation_defect += 1
            relative_image = Path(label_dir) / output_name
            mode = _materialize(
                image_path,
                evaluation_root / relative_image,
                copy_files=copy_files,
            )
            materialization[mode] += 1
            row: dict[str, Any] = {
                "id": f"{category}-holdout-{len(evaluation_rows) + 1:04d}",
                "image": relative_image.as_posix(),
                "label": label,
                "sha256": digest,
            }
            if label == 1:
                row["defect_type"] = defect_type
                mask_path = _mask_for(
                    source_root,
                    category,
                    defect_type,
                    image_path,
                )
                mode = _materialize(
                    mask_path,
                    evaluation_masks / output_name,
                    copy_files=copy_files,
                )
                materialization[mode] += 1
            evaluation_rows.append(row)

    if evaluation_normal != episode.test_normal_count:
        raise RuntimeError(
            f"{category}: expected {episode.test_normal_count} holdout normals, "
            f"got {evaluation_normal}"
        )
    if evaluation_defect != episode.test_defect_count:
        raise RuntimeError(
            f"{category}: expected {episode.test_defect_count} holdout defects, "
            f"got {evaluation_defect}"
        )
    if len(evaluation_rows) != episode.independent_test_count:
        raise RuntimeError(
            f"{category}: expected {episode.independent_test_count} holdout images, "
            f"got {len(evaluation_rows)}"
        )

    manifest_name = "05_评估清单_留出集.yaml"
    manifest_path = category_destination / manifest_name
    write_yaml(
        manifest_path,
        {
            "schema_version": 1,
            "image_root": "04_评估图片_留出集",
            "samples": evaluation_rows,
        },
    )
    loaded = load_custom_test_manifest(
        manifest_path,
        minimum_samples=1,
        excluded_hashes=support_hashes,
    )
    if len(loaded) != len(evaluation_rows):
        raise RuntimeError(f"{category}: generated holdout manifest count changed")

    summary = {
        "category": category,
        "protocol": "deterministic per-category 100 normal + 30 defect",
        "seed": seed,
        "support": {
            "normal": len(episode.dataset.normal),
            "defect": len(episode.dataset.defect),
            "masks": sum(
                sample.mask_path is not None for sample in episode.dataset.defect
            ),
            "defect_types": list(episode.defect_types),
        },
        "evaluation": {
            "manifest": manifest_name,
            "total": len(evaluation_rows),
            "normal": evaluation_normal,
            "defect": evaluation_defect,
            "masks": evaluation_defect,
            "source": "remaining official MVTec test after defect support selection",
            "ui_1000_plus_independent_evaluation_eligible": len(evaluation_rows)
            >= 1000,
        },
        "support_evaluation_hash_overlap": len(support_hashes & evaluation_hashes),
        "materialization": materialization,
    }
    write_yaml(category_destination / "划分摘要.yaml", summary)
    return summary


def prepare_package(
    source_root: Path,
    destination: Path,
    *,
    normal_count: int = DEFAULT_NORMAL_COUNT,
    defect_count: int = DEFAULT_DEFECT_COUNT,
    seed: int = DEFAULT_SEED,
    minimum_holdout: int = 1,
    copy_files: bool = False,
) -> dict[str, Any]:
    if normal_count != DEFAULT_NORMAL_COUNT or defect_count != DEFAULT_DEFECT_COUNT:
        raise ValueError(
            "this UI import package requires exactly 100 normal + 30 defect images "
            "per category"
        )
    source_root = source_root.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    episodes, skipped = build_eligible_episodes(
        source_root,
        categories=None,
        normal_count=normal_count,
        defect_count=defect_count,
        seed=seed,
        minimum_holdout=minimum_holdout,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        summaries = []
        for episode in episodes:
            summaries.append(
                _prepare_category(
                    source_root=source_root,
                    episode=episode,
                    category_destination=staging / episode.category,
                    seed=seed,
                    copy_files=copy_files,
                )
            )
        pooled_holdout = sum(row["evaluation"]["total"] for row in summaries)
        index = {
            "schema_version": 1,
            "purpose": "per-category MVTec generalization holdout index",
            "direct_ui_1000_plus_manifest": False,
            "reason": (
                "the 1000+ requirement is met only after pooling categories; "
                "each category requires its own adapted model"
            ),
            "pooled_images": pooled_holdout,
            "categories": [
                {
                    "category": row["category"],
                    "manifest": (
                        f"{row['category']}/"
                        f"{row['evaluation']['manifest']}"
                    ),
                    "total": row["evaluation"]["total"],
                    "normal": row["evaluation"]["normal"],
                    "defect": row["evaluation"]["defect"],
                }
                for row in summaries
            ],
        }
        write_yaml(staging / "泛化评估索引.yaml", index)
        (staging / "导入说明.md").write_text(
            _readme(
                categories=[episode.category for episode in episodes],
                skipped=skipped,
                pooled_holdout=pooled_holdout,
                seed=seed,
            ),
            encoding="utf-8",
        )
        result = {
            "schema_version": 1,
            "purpose": "MVTec AD DAGM-style UI manual import and generalization demo",
            "source_root": str(source_root),
            "source_modified": False,
            "support_protocol": {
                "normal_per_category": normal_count,
                "defect_per_category": defect_count,
                "seed": seed,
            },
            "evaluation_policy": (
                "remaining official MVTec test after per-category support selection"
            ),
            "fresh_independent_test": False,
            "classes": summaries,
            "skipped": skipped,
            "totals": {
                "eligible_categories": len(summaries),
                "skipped_categories": len(skipped),
                "support_images": sum(
                    row["support"]["normal"] + row["support"]["defect"]
                    for row in summaries
                ),
                "support_masks": sum(row["support"]["masks"] for row in summaries),
                "evaluation_images": pooled_holdout,
                "evaluation_masks": sum(
                    row["evaluation"]["masks"] for row in summaries
                ),
                "support_evaluation_hash_overlap": sum(
                    row["support_evaluation_hash_overlap"] for row in summaries
                ),
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
        description="Prepare eligible MVTec categories as DAGM-style import folders."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--normal-count", type=int, default=DEFAULT_NORMAL_COUNT)
    parser.add_argument("--defect-count", type=int, default=DEFAULT_DEFECT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--minimum-holdout", type=int, default=1)
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
        args.destination,
        normal_count=args.normal_count,
        defect_count=args.defect_count,
        seed=args.seed,
        minimum_holdout=args.minimum_holdout,
        copy_files=args.copy,
    )
    print(yaml.safe_dump(result["totals"], sort_keys=False, allow_unicode=True))
    print(f"destination: {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
