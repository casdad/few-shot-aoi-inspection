from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from aoi.images import read_rgb
from aoi.paths import PROJECT_ROOT, portable_path, resolve_artifact
from aoi.runtime.config import resolve_device
from aoi.runtime.runtime import AdaptedRuntime
from aoi.video import synchronize


def _load_runtime(pointer_path: Path, category: str, device: str) -> AdaptedRuntime:
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8")) or {}
    models = pointer.get("models", {})
    if category not in models:
        raise KeyError(
            f"category {category!r} has no adapted model; available: {sorted(models)}"
        )
    bundle = resolve_artifact(models[category], base=pointer_path.parent)
    return AdaptedRuntime(bundle, device=device)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def _title_panel(image: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    scale = max(0.55, min(output.shape[:2]) / 900.0)
    thickness = max(1, round(scale * 2))
    line_height = max(24, round(30 * scale))
    panel_height = line_height * len(lines) + 16
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.76, output, 0.24, 0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (12, 8 + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color if index == 0 else (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
    return output


def _labeled_tile(image: np.ndarray, label: str) -> np.ndarray:
    output = np.full(
        (image.shape[0] + 40, image.shape[1], image.shape[2]),
        20,
        dtype=image.dtype,
    )
    output[40:] = image
    cv2.putText(
        output,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def run_demo(
    *,
    input_path: Path,
    pointer_path: Path,
    category: str,
    device: str,
    output_dir: Path,
    warmup: int,
) -> dict[str, Any]:
    image_rgb = read_rgb(input_path)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    load_started = time.perf_counter()
    runtime = _load_runtime(pointer_path, category, device)
    model_load_seconds = time.perf_counter() - load_started

    for index in range(warmup):
        runtime.predict_explain(image_rgb, f"warmup-{index:02d}")
    synchronize(device)
    started = time.perf_counter()
    result = runtime.predict_explain(image_rgb, input_path.name)
    synchronize(device)
    inference_ms = (time.perf_counter() - started) * 1000.0

    anomaly_map = np.asarray(result.pop("_anomaly_map"), dtype=np.float32)
    anomaly_u8 = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_u8, cv2.COLORMAP_TURBO)
    if heatmap.shape[:2] != image_bgr.shape[:2]:
        heatmap = cv2.resize(
            heatmap,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    overlay = cv2.addWeighted(image_bgr, 0.58, heatmap, 0.42, 0)

    decision = bool(result["decision"])
    status = "ANOMALY" if decision else "NORMAL"
    color = (40, 70, 235) if decision else (40, 170, 60)
    annotated = _title_panel(
        overlay,
        [
            f"{category} | {status}",
            (
                f"fusion score {float(result['fusion_score']):.4f} / "
                f"threshold {float(result['fusion_threshold']):.4f}"
            ),
            f"warmed inference {inference_ms:.1f} ms | {device}",
        ],
        color,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "demo_original.png"
    heatmap_path = output_dir / "demo_anomaly_heatmap.png"
    result_path = output_dir / "demo_result.png"
    panel_path = output_dir / "demo_panel.png"
    receipt_path = output_dir / "demo_result.yaml"
    _write_png(original_path, image_bgr)
    _write_png(heatmap_path, heatmap)
    _write_png(result_path, annotated)

    target_height = min(512, image_bgr.shape[0])
    target_width = round(image_bgr.shape[1] * target_height / image_bgr.shape[0])
    tiles = [
        _labeled_tile(
            cv2.resize(value, (target_width, target_height), interpolation=cv2.INTER_AREA),
            label,
        )
        for value, label in (
            (image_bgr, "INPUT"),
            (heatmap, "MODEL ANOMALY MAP"),
            (annotated, "AOI RESULT"),
        )
    ]
    _write_png(panel_path, np.hstack(tiles))

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "input": portable_path(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "active_models_pointer": portable_path(pointer_path),
        "category": category,
        "device": device,
        "decision": decision,
        "result_label": status.lower(),
        "defect_type": "generic_anomaly" if decision else "none",
        "fusion_score": float(result["fusion_score"]),
        "fusion_threshold": float(result["fusion_threshold"]),
        "model_load_seconds": model_load_seconds,
        "warmup_runs": warmup,
        "inference_ms": inference_ms,
        "timing_scope": (
            "one warmed AdaptedRuntime.predict_explain call including in-runtime "
            "preprocessing and score fusion; excludes disk decode, model loading, and rendering"
        ),
        "confidence_note": "fusion_score is not a calibrated probability",
        "localization_note": (
            "the heatmap is the model anomaly map; it is not a ground-truth mask or bounding box"
        ),
        "outputs": {
            "original": portable_path(original_path),
            "anomaly_heatmap": portable_path(heatmap_path),
            "annotated_result": portable_path(result_path),
            "comparison_panel": portable_path(panel_path),
        },
        "raw_result": result,
    }
    receipt_path.write_text(
        yaml.safe_dump(receipt, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(yaml.safe_dump(receipt, sort_keys=False, allow_unicode=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a real AOI anomaly-map demo from an adapted model bundle."
    )
    parser.add_argument("--input", type=Path, required=True, help="input AOI image")
    parser.add_argument(
        "--models", type=Path, required=True, help="active_models.yaml pointer"
    )
    parser.add_argument("--category", required=True, help="model category/product name")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "images",
        help="directory for PNG outputs and the YAML receipt",
    )
    parser.add_argument("--warmup", type=int, default=1, help="warmup predictions")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    pointer_path = args.models if args.models.is_absolute() else PROJECT_ROOT / args.models
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_demo(
        input_path=input_path.resolve(strict=True),
        pointer_path=pointer_path.resolve(strict=True),
        category=args.category.strip(),
        device=resolve_device(args.device),
        output_dir=output_dir.resolve(),
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
