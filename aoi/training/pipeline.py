from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from aoi.competition.baselines import DeepPatchCoreLite
from aoi.competition.models.anomaly_detectors import DINOv2SpatialPrototype, MultiScalePatchMemory
from aoi.competition.models.known_defect_heads import DinoTiledMLPKnownHead

from .adaptation import adapt_episode
from .config import TrainingConfig, load_config
from .custom import CustomEpisode
from .dagm import DAGMEpisode, build_eligible_episodes as build_dagm_episodes
from .dataset import TrainingDataset
from .manifests import payload_hash, sha256_file, write_yaml
from .mvtec import MVTecEpisode, build_eligible_episodes as build_mvtec_episodes
from .runtime_scoring import score_images


Episode = DAGMEpisode | MVTecEpisode | CustomEpisode


def _run_id(config: TrainingConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.run_name}_{stamp}"


def _environment(config: TrainingConfig) -> dict[str, object]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        # ``torch.__version__`` is a ``TorchVersion`` subclass in recent
        # releases, which PyYAML's safe dumper cannot serialize directly.
        "torch": str(torch.__version__),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": config.device,
        "gpu": gpu,
    }


def _support_receipt(
    dataset: TrainingDataset,
    category: str,
    dataset_name: str,
) -> dict[str, object]:
    annotation_mode = (
        "pixel_mask" if all(s.mask_path is not None for s in dataset.defect)
        else "image_label"
    )
    return {
        "dataset": dataset_name,
        "category": category,
        "annotation_mode": annotation_mode,
        "normal_count": len(dataset.normal),
        "defect_count": len(dataset.defect),
        "normal": [
            {"id": s.sample_id, "path": str(s.image_path), "sha256": s.sha256}
            for s in dataset.normal
        ],
        "defect": [
            {
                "id": s.sample_id,
                "path": str(s.image_path),
                "mask": None if s.mask_path is None else str(s.mask_path),
                "defect_type": s.defect_type,
                "sha256": s.sha256,
            }
            for s in dataset.defect
        ],
    }


def _reload_smoke(config: TrainingConfig, run_dir: Path, artifacts) -> dict[str, object]:
    primary = MultiScalePatchMemory.load(run_dir / "models" / "primary.pt", device=config.device)
    adapted = DeepPatchCoreLite.load(run_dir / "models" / "adapted.pt", map_location=config.device)
    dino = DINOv2SpatialPrototype.load(run_dir / "models" / "dino.pt", device=config.device)
    head = DinoTiledMLPKnownHead.load(run_dir / "models" / "head.pt", device=config.device)
    rows = score_images(
        images=artifacts.input_samples,
        ids=artifacts.input_ids,
        source=artifacts.source,
        adapted=adapted,
        primary=primary,
        dino=dino,
        head=head,
        calibration=artifacts.calibration,
        device=config.device,
    )
    if len(rows) != 2:
        raise RuntimeError("reload smoke did not return two predictions")
    return {"status": "PASS", "predictions": rows}


def _protocol_name(dataset: str) -> str:
    if dataset == "dagm2007":
        return "dagm2007-official-train-test-fewshot-100+30"
    if dataset == "mvtec_ad":
        return "mvtec-fewshot-competition-100+30"
    return "custom-private-fewshot-100+30"


def _benchmark_note(dataset: str) -> str:
    if dataset == "dagm2007":
        return (
            "DAGM2007 100+30 support and calibration use only the official Train split; "
            "evaluation uses the disjoint official Test split."
        )
    if dataset == "custom":
        return (
            "Private-task 100 normal + 30 labeled defects are used only for adaptation; "
            "pixel masks are optional and the selected annotation mode is recorded; the "
            "optional custom test directory is held out and checked for hash overlap."
        )
    return (
        "This is not the standard unsupervised MVTec AD benchmark: 30 labeled defect "
        "test images are reserved as few-shot support and excluded from holdout evaluation."
    )


def _train_episode(config: TrainingConfig, episode: Episode) -> Path:
    dataset = episode.dataset
    category_config = config.for_category(episode.category)
    run_dir = category_config.output_root / _run_id(category_config)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for name in ("models", "calibration", "reports"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    dataset_label = (
        "DAGM2007" if config.dataset == "dagm2007"
        else "MVTec AD" if config.dataset == "mvtec_ad"
        else "CUSTOM PRIVATE TASK"
    )
    print(f"\n=== {dataset_label} EPISODE: {episode.category} ===")
    support_source = (
        "official Train" if config.dataset == "dagm2007"
        else "train/good + test defects" if config.dataset == "mvtec_ad"
        else "operator-selected private support"
    )
    holdout_source = (
        "official Test" if config.dataset == "dagm2007"
        else "remaining test" if config.dataset == "mvtec_ad"
        else "custom test directory"
    )
    print(
        f"Support  : {len(dataset.normal)} normal + {len(dataset.defect)} labeled defects "
        f"from {support_source}"
    )
    print(
        f"Holdout  : {episode.independent_test_count} {holdout_source} images "
        f"({episode.test_normal_count} normal + {episode.test_defect_count} defect)"
    )
    print(f"MLP head : {category_config.head_epochs} epochs, lr={category_config.head_learning_rate}")

    started = time.perf_counter()
    print("[1/5] Building MultiScale Patch Memory ...")
    print("[2/5] Building Source/Adapted PatchCore branch ...")
    print("[3/5] Building DINO spatial prototype and normal/defect banks ...")
    print(f"[4/5] Training Known Defect MLP head ({category_config.head_epochs} epochs) ...")
    artifacts = adapt_episode(category_config, dataset, run_dir)
    print("[5/5] Reloading saved models and running parity smoke ...")
    reload_receipt = _reload_smoke(category_config, run_dir, artifacts)

    support = _support_receipt(dataset, episode.category, config.dataset)
    write_yaml(run_dir / "support_receipt.yaml", support)
    write_yaml(run_dir / "reload_smoke.yaml", reload_receipt)
    write_yaml(run_dir / "environment.yaml", _environment(category_config))
    write_yaml(
        run_dir / "config_snapshot.yaml",
        yaml.safe_load(category_config.config_path.read_text(encoding="utf-8")),
    )

    bundle = {
        "schema_version": 3,
        "protocol": _protocol_name(config.dataset),
        "benchmark_note": _benchmark_note(config.dataset),
        "run_id": run_dir.name,
        "target_dataset": config.dataset,
        "target_object": episode.category,
        "annotation_mode": support["annotation_mode"],
        "device_used": category_config.device,
        "source_model": category_config.source_assets["source_model"].path.relative_to(
            config.project_root
        ).as_posix(),
        "models": {
            "primary": "models/primary.pt",
            "adapted": "models/adapted.pt",
            "dino": "models/dino.pt",
            "head": "models/head.pt",
        },
        "calibration": "calibration/calibration.yaml",
        "support_receipt": "support_receipt.yaml",
        "support_content_hashes": [s.sha256 for s in dataset.samples],
        "support_counts": {
            "normal": len(dataset.normal),
            "defect": len(dataset.defect),
        },
        "planned_holdout": {
            "total": episode.independent_test_count,
            "normal": episode.test_normal_count,
            "defect": episode.test_defect_count,
            "split": (
                "Test" if config.dataset == "dagm2007"
                else "remaining_test" if config.dataset == "mvtec_ad"
                else "custom_test"
            ),
        },
        "defect_types": list(episode.defect_types),
        "assets": {
            name: {
                "path": asset.path.relative_to(config.project_root).as_posix(),
                "sha256": sha256_file(asset.path) if asset.path.is_file() else None,
            }
            for name, asset in category_config.source_assets.items()
        },
        "adaptation_seconds": artifacts.adaptation_seconds,
        "head_training": artifacts.calibration["head"]["training"],
        "thresholds": {
            "general": float(artifacts.calibration["general"]["general_threshold"]),
            "local": float(artifacts.calibration["head"]["local_threshold"]),
            "known": float(artifacts.calibration["known"]["known_threshold"]),
            "fusion": float(
                artifacts.calibration["continuous_fusion"]["fusion"]["threshold"]
            ),
        },
        "fusion": {
            "schema": artifacts.calibration["continuous_fusion"]["schema"],
            "aggregation": artifacts.calibration["continuous_fusion"]["fusion"][
                "aggregation"
            ],
            "weights": artifacts.calibration["continuous_fusion"]["fusion"]["weights"],
            "support_oof_metrics": artifacts.calibration["continuous_fusion"]["fusion"][
                "oof_metrics"
            ],
            "independent_evaluation_used_for_selection": False,
        },
        "artifact_sha256": {
            "source_model": sha256_file(
                category_config.source_assets["source_model"].path
            ),
            "models": {
                name: sha256_file(path)
                for name, path in artifacts.model_paths.items()
                if name != "calibration"
            },
            "calibration": sha256_file(artifacts.model_paths["calibration"]),
            "support_receipt": sha256_file(run_dir / "support_receipt.yaml"),
        },
        "bundle_hash": "pending",
    }
    bundle["bundle_hash"] = payload_hash({k: v for k, v in bundle.items() if k != "bundle_hash"})
    write_yaml(run_dir / "model_bundle.yaml", bundle)
    (run_dir / "READY.txt").write_text("ADAPTED_MODEL_READY\n", encoding="utf-8")

    elapsed = time.perf_counter() - started
    print(f"ADAPTED_MODEL_READY: {episode.category} ({elapsed:.2f} s)")
    return (run_dir / "model_bundle.yaml").resolve()


def train_custom_episode(config: TrainingConfig, episode: CustomEpisode) -> Path:
    """Train one private 100+30 task and write an isolated model pointer."""
    if config.dataset != "custom":
        raise ValueError("custom episode requires config.dataset='custom'")
    config.output_root.mkdir(parents=True, exist_ok=True)
    bundle = _train_episode(config, episode)
    pointer = {
        "schema_version": 1,
        "dataset": "custom",
        "protocol": _protocol_name("custom"),
        "models": {episode.category: bundle.relative_to(config.output_root).as_posix()},
        "planned_independent_test_images": episode.independent_test_count,
        "support_counts": {"normal": 100, "defect": 30},
        "test_root": None if episode.test_root is None else str(episode.test_root),
        "test_manifest": (
            None if episode.test_manifest is None else str(episode.test_manifest)
        ),
        "test_counts": {
            "total": episode.independent_test_count,
            "normal": episode.test_normal_count,
            "defect": episode.test_defect_count,
        },
        "training_config": str(config.config_path),
        "training_config_sha256": sha256_file(config.config_path),
        "annotation_mode": episode.annotation_mode,
    }
    pointer_path = config.output_root / "active_models.yaml"
    write_yaml(pointer_path, pointer)
    return pointer_path.resolve()


def run_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    builder = build_dagm_episodes if config.dataset == "dagm2007" else build_mvtec_episodes
    episodes, skipped = builder(
        config.data_root,
        categories=config.categories,
        normal_count=config.support_normal_count,
        defect_count=config.support_defect_count,
        seed=config.seed,
        minimum_holdout=config.minimum_holdout_per_category,
    )

    if config.dataset == "dagm2007":
        print("=== AOI DAGM2007 FEW-SHOT TRAIN ===")
        print("Protocol : each DAGM class is an independent 100+30 episode")
        print("Support  : selected only from official ClassN/Train using Labels.txt")
        print("Holdout  : complete disjoint official ClassN/Test split")
    else:
        print("=== AOI MVTec AD FEW-SHOT TRAIN ===")
        print("Protocol : each MVTec category is an independent 100+30 episode")
        print("Normal   : selected from category/train/good")
        print("Defect   : selected from category/test/<defect_type> with masks")
        print("Holdout  : remaining category/test images; support defects are excluded")
        print("Important: this is not the standard unsupervised MVTec benchmark")
    print(f"Python   : {sys.executable}")
    print(f"Device   : {config.device}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Dataset  : {config.dataset}")
    print(f"Data root: {config.data_root}")
    planned_total = sum(e.independent_test_count for e in episodes)
    print(f"Episodes : {len(episodes)} eligible")
    print(f"Holdout  : planned pooled independent test = {planned_total} images")
    if planned_total < config.minimum_pooled_test:
        raise ValueError(
            f"eligible {config.dataset} episodes provide only {planned_total} holdout images; "
            f"competition protocol requires >= {config.minimum_pooled_test}"
        )
    if skipped:
        print(f"Skipped  : {len(skipped)} categories that cannot satisfy the configured 100+30 protocol")
        for row in skipped:
            print(f"           - {row['category']}: {row['reason']}")

    config.output_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, str] = {}
    trained_total = 0
    for episode in episodes:
        bundle = _train_episode(config, episode)
        models[episode.category] = bundle.relative_to(config.output_root).as_posix()
        trained_total += episode.independent_test_count

    pointer = {
        "schema_version": 1,
        "dataset": config.dataset,
        "protocol": _protocol_name(config.dataset),
        "models": models,
        "planned_independent_test_images": trained_total,
        "skipped_categories": skipped,
    }
    active_path = config.output_root / "active_models.yaml"
    write_yaml(active_path, pointer)
    write_yaml(config.output_root / "episode_plan.yaml", {
        "eligible_categories": [e.category for e in episodes],
        "planned_independent_test_images": trained_total,
        "episodes": [
            {
                "category": e.category,
                "support_normal": config.support_normal_count,
                "support_defect": config.support_defect_count,
                "holdout_total": e.independent_test_count,
                "holdout_normal": e.test_normal_count,
                "holdout_defect": e.test_defect_count,
                "defect_types": list(e.defect_types),
            }
            for e in episodes
        ],
        "skipped_categories": skipped,
    })

    ready_status = (
        "ALL_DAGM2007_EPISODES_READY"
        if config.dataset == "dagm2007"
        else "ALL_MVTEC_EPISODES_READY"
    )
    print(f"\n{ready_status}")
    print(f"Models   : {len(models)}")
    print(f"Holdout  : planned pooled independent test = {trained_total} images")
    print(f"Pointer  : {active_path}")
    print("Next     : run test.py")
    return active_path
