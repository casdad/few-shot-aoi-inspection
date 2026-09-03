from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class SourceAsset:
    path: Path


@dataclass(frozen=True)
class TrainingConfig:
    config_path: Path
    project_root: Path
    dataset: str
    data_root: Path
    run_name: str
    candidate_id: str
    output_root: Path
    device: str
    seed: int
    source_assets: dict[str, SourceAsset]
    head_epochs: int
    head_learning_rate: float
    support_normal_count: int
    support_defect_count: int
    categories: tuple[str, ...] | None
    minimum_holdout_per_category: int
    minimum_pooled_test: int

    @property
    def episode_id(self) -> str:
        return self.run_name

    @property
    def mvtec_root(self) -> Path:
        """Backward-compatible alias for historical MVTec-only callers."""
        return self.data_root

    def for_category(self, category: str) -> "TrainingConfig":
        prefix = (
            "mvtec" if self.dataset == "mvtec_ad"
            else "custom" if self.dataset == "custom"
            else "dagm2007"
        )
        candidate_dataset = (
            "MVTEC" if self.dataset == "mvtec_ad"
            else "CUSTOM" if self.dataset == "custom"
            else "DAGM2007"
        )
        return replace(
            self,
            run_name=f"{prefix}_{category}_100_30",
            candidate_id=f"AOI-{candidate_dataset}-{category}",
        )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_device(value: str) -> str:
    value = str(value).strip().lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return value


def normalize_dataset_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "dagm": "dagm2007",
        "dagm_2007": "dagm2007",
        "dagm2007": "dagm2007",
        "mvtec": "mvtec_ad",
        "mvtec_ad": "mvtec_ad",
        "custom": "custom",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dataset {value!r}; expected custom, dagm2007 or mvtec_ad"
        ) from exc


def load_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).resolve(strict=True)
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_root = config_path.parent.parent
    assets = payload.get("source_assets", {})
    required_assets = {
        "resnet18_weights",
        "dinov2_repository",
        "dinov2_weights",
        "source_model",
    }
    missing = sorted(required_assets - set(assets))
    if missing:
        raise ValueError(f"missing source_assets entries: {missing}")
    source_assets = {
        name: SourceAsset(_resolve(project_root, str(assets[name]))) for name in required_assets
    }
    for name, item in source_assets.items():
        if not item.path.exists():
            raise FileNotFoundError(f"source asset missing: {name}: {item.path}")

    protocol = payload.get("protocol", {}) or {}
    raw_dataset = payload.get("dataset")
    if raw_dataset is None:
        raw_dataset = "mvtec_ad" if "mvtec_root" in payload else "dagm2007"
    dataset = normalize_dataset_name(str(raw_dataset))
    if "data_root" in payload:
        raw_data_root = payload["data_root"]
    elif dataset == "mvtec_ad":
        raw_data_root = payload.get("mvtec_root", "data/mvtec_ad")
    else:
        raw_data_root = payload.get("dagm_root", "data/dagm2007")

    raw_categories = payload.get("categories", "auto")
    if isinstance(raw_categories, str) and raw_categories.strip().lower() == "auto":
        categories = None
    elif isinstance(raw_categories, list):
        categories = tuple(str(v).strip() for v in raw_categories if str(v).strip())
        if not categories:
            raise ValueError("categories list is empty")
    else:
        raise TypeError("categories must be 'auto' or a YAML list")

    return TrainingConfig(
        config_path=config_path,
        project_root=project_root,
        dataset=dataset,
        data_root=_resolve(project_root, str(raw_data_root)),
        run_name=str(payload.get("run_name", dataset)),
        candidate_id=str(payload.get("candidate_id", f"AOI-{dataset.upper()}")),
        output_root=_resolve(project_root, str(payload.get("output_root", "outputs"))),
        device=resolve_device(str(payload.get("device", "auto"))),
        seed=int(payload.get("seed", 20260808)),
        source_assets=source_assets,
        head_epochs=int(payload.get("head", {}).get("epochs", 60)),
        head_learning_rate=float(payload.get("head", {}).get("learning_rate", 0.01)),
        support_normal_count=int(protocol.get("normal_support", 100)),
        support_defect_count=int(protocol.get("defect_support", 30)),
        categories=categories,
        minimum_holdout_per_category=int(protocol.get("minimum_holdout_per_category", 1)),
        minimum_pooled_test=int(protocol.get("minimum_pooled_test", 1000)),
    )
