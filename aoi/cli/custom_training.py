"""Non-interactive 100-normal + 30-defect private-task training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from aoi.custom_manifest import load_custom_test_manifest
from aoi.paths import PROJECT_ROOT
from aoi.training.config import TrainingConfig, load_config, resolve_device
from aoi.training.custom import CustomEpisode, build_custom_episode
from aoi.training.pipeline import train_custom_episode


def _project_path(value: Any, context: str, *, strict: bool = True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError(f"{context} must be a non-empty path")
    path = Path(value)
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    return candidate.resolve(strict=strict)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a YAML mapping")
    return value


def _task_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    name = value.strip()
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{context} must be a single safe folder name")
    return name


def prepare_custom_training(
    config_path: str | Path,
) -> tuple[TrainingConfig, CustomEpisode, bool]:
    """Validate the custom wrapper config without starting model training."""

    path = Path(config_path).resolve(strict=True)
    payload = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")), "custom training config"
    )
    allowed = {
        "schema_version",
        "base_training_config",
        "task_name",
        "category",
        "normal_dir",
        "defect_dir",
        "mask_dir",
        "test_manifest",
        "minimum_test_images",
        "output_root",
        "refuse_overwrite",
        "device",
        "seed",
        "head",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"custom training config contains unsupported fields: {unknown}")
    if payload.get("schema_version") != 1 or type(payload.get("schema_version")) is not int:
        raise ValueError("custom training config schema_version must be integer 1")

    task_name = _task_name(payload.get("task_name"), "task_name")
    category = _task_name(payload.get("category", task_name), "category")
    base_path = _project_path(
        payload.get("base_training_config", "configs/custom_training_base.yaml"),
        "base_training_config",
    )
    base = load_config(base_path)
    normal_dir = _project_path(payload.get("normal_dir"), "normal_dir")
    defect_dir = _project_path(payload.get("defect_dir"), "defect_dir")
    raw_mask = payload.get("mask_dir")
    mask_dir = None if raw_mask in {None, ""} else _project_path(raw_mask, "mask_dir")
    episode = build_custom_episode(
        normal_dir=normal_dir,
        defect_dir=defect_dir,
        mask_dir=mask_dir,
        category=category,
    )

    raw_manifest = payload.get("test_manifest")
    if raw_manifest not in {None, ""}:
        minimum_test_images = int(payload.get("minimum_test_images", 1000))
        if minimum_test_images < 1000:
            raise ValueError("minimum_test_images must be >= 1000")
        manifest_path = _project_path(raw_manifest, "test_manifest")
        support_hashes = {sample.sha256 for sample in episode.dataset.samples}
        test_samples = load_custom_test_manifest(
            manifest_path,
            minimum_samples=minimum_test_images,
            excluded_hashes=support_hashes,
        )
        episode = replace(
            episode,
            independent_test_count=len(test_samples),
            test_normal_count=sum(sample.label == 0 for sample in test_samples),
            test_defect_count=sum(sample.label == 1 for sample in test_samples),
            test_manifest=manifest_path,
        )

    raw_output = payload.get("output_root", f"outputs/custom_tasks/{task_name}")
    output_root = _project_path(raw_output, "output_root", strict=False)
    refuse_overwrite = payload.get("refuse_overwrite", True)
    if type(refuse_overwrite) is not bool:
        raise TypeError("refuse_overwrite must be a boolean")

    head = _mapping(payload.get("head", {}), "head")
    unknown_head = sorted(set(head) - {"epochs", "learning_rate"})
    if unknown_head:
        raise ValueError(f"head contains unsupported fields: {unknown_head}")
    training = replace(
        base,
        config_path=path,
        dataset="custom",
        data_root=normal_dir,
        run_name=task_name,
        candidate_id=f"AOI-CUSTOM-{task_name}",
        output_root=output_root,
        device=resolve_device(str(payload.get("device", base.device))),
        seed=int(payload.get("seed", base.seed)),
        head_epochs=int(head.get("epochs", base.head_epochs)),
        head_learning_rate=float(
            head.get("learning_rate", base.head_learning_rate)
        ),
        support_normal_count=100,
        support_defect_count=30,
        categories=(category,),
        minimum_holdout_per_category=0,
        minimum_pooled_test=0,
    )
    return training, episode, refuse_overwrite


def run(config_path: str | Path) -> int:
    training, episode, refuse_overwrite = prepare_custom_training(config_path)
    pointer_path = training.output_root / "active_models.yaml"
    if refuse_overwrite and pointer_path.exists():
        raise FileExistsError(
            f"refusing to replace existing private-task pointer: {pointer_path}"
        )
    print("=== AOI PRIVATE 100+30 TRAINING ===")
    print(f"Task     : {training.run_name}")
    print(f"Category : {episode.category}")
    print("Support  : 100 normal + 30 labeled defect")
    print(f"Annotation: {episode.annotation_mode}")
    print(f"Holdout  : {episode.independent_test_count} labeled images")
    print(f"Device   : {training.device}")
    result = train_custom_episode(training, episode)
    print(f"Pointer  : {result}")
    return 0


__all__ = ["prepare_custom_training", "run"]
