from __future__ import annotations

from pathlib import Path

import yaml

from aoi.feedback import FeedbackStore, apply_active_feedback, apply_threshold
from aoi.paths import PROJECT_ROOT, resolve_artifact
from aoi.runtime.config import load_yaml, resolve_device, resolve_path
from aoi.runtime.data import read_rgb
from aoi.runtime.runtime import AdaptedRuntime


def run(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    device = resolve_device(str(config.get("device", "auto")))
    pointer_value = config.get("active_models_pointer")
    if not isinstance(pointer_value, str) or not pointer_value.strip():
        raise ValueError("active_models_pointer is required")
    pointer = resolve_path(config, pointer_value)
    payload = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
    category = str(config.get("category", "")).strip()
    models = payload.get("models", {})
    if category not in models:
        raise KeyError(f"category {category!r} has no adapted model; available: {sorted(models)}")
    bundle = resolve_artifact(models[category], base=pointer.parent)
    image = resolve_path(config, str(config["image"]))
    runtime = AdaptedRuntime(bundle, device=device)
    feedback_enabled = bool(config.get("feedback_enabled", True))
    feedback_root = resolve_path(
        config,
        str(config.get("feedback_root", PROJECT_ROOT / "outputs" / "feedback")),
    )
    feedback_state = apply_active_feedback(
        runtime,
        FeedbackStore(feedback_root),
        category,
        enabled=feedback_enabled,
    )
    image_rgb = read_rgb(image)
    result = apply_threshold(
        runtime.predict(image_rgb, f"{category}/{image.name}"),
        feedback_state.threshold,
    )
    print(
        yaml.safe_dump(
            {
                "category": category,
                "image": str(image),
                "feedback_state": feedback_state.receipt(),
                **result,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )
    return 0
