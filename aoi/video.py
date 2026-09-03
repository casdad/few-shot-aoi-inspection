from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import yaml

from aoi.feedback import apply_threshold


class VideoRuntime(Protocol):
    """Minimal runtime interface required by model-only video inference."""

    def predict(self, rgb: np.ndarray, sample_id: str) -> Mapping[str, Any]: ...


ProgressCallback = Callable[[int, int | None], None]
FrameCallback = Callable[[np.ndarray, Mapping[str, Any]], bool | None]


def draw_result(
    frame: np.ndarray,
    *,
    category: str,
    result: Mapping[str, Any],
    latency_ms: float,
) -> np.ndarray:
    """Draw the model result on a BGR frame."""

    output = frame.copy()
    anomaly = bool(result["decision"])
    color = (0, 0, 255) if anomaly else (0, 180, 0)
    status = "ANOMALY" if anomaly else "NORMAL"
    score = float(result["fusion_score"])
    threshold = float(result["fusion_threshold"])
    lines = [
        f"{category}  {status}",
        f"score {score:.3f} / threshold {threshold:.3f}",
        f"inference {latency_ms:.1f} ms",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(output.shape[:2]) / 900.0)
    thickness = max(1, round(scale * 2))
    line_height = round(30 * scale)
    panel_height = line_height * len(lines) + 20
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (12, 10 + line_height * (index + 1)),
            font,
            scale,
            color if index == 0 else (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )
    cv2.rectangle(
        output,
        (1, 1),
        (output.shape[1] - 2, output.shape[0] - 2),
        color,
        thickness + 1,
    )
    return output


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reported_frame_count(capture: cv2.VideoCapture, max_frames: int) -> int | None:
    value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(value) or value <= 0:
        return max_frames if max_frames > 0 else None
    total = int(value)
    return min(total, max_frames) if max_frames > 0 else total


def _latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def run_video_inspection(
    source: str | Path | int,
    *,
    runtime: VideoRuntime,
    category: str,
    device: str,
    output_path: str | Path,
    summary_path: str | Path,
    threshold: float | None = None,
    feedback_state: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_event: Event | None = None,
    max_frames: int = 0,
    frame_callback: FrameCallback | None = None,
) -> dict[str, Any]:
    """Run model inference over a video and write annotated MP4/YAML outputs."""

    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0:
        raise ValueError("max_frames must be a non-negative integer")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")

    resolved_source: Path | int
    if isinstance(source, bool):
        raise TypeError("source must be a path or camera index")
    if isinstance(source, int):
        resolved_source = source
    else:
        resolved_source = Path(source).resolve(strict=True)
        if not resolved_source.is_file():
            raise ValueError(f"video source is not a file: {resolved_source}")

    output = Path(output_path).resolve()
    summary_file = Path(summary_path).resolve()
    if isinstance(resolved_source, Path) and output == resolved_source:
        raise ValueError("output_path must not overwrite the source video")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(
        str(resolved_source) if isinstance(resolved_source, Path) else resolved_source
    )
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source: {resolved_source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("video source reported an invalid frame size")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = _reported_frame_count(capture, max_frames)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create output video: {output}")

    latencies: list[float] = []
    frame_count = 0
    anomaly_frames = 0
    anomaly_events = 0
    previous_anomaly = False
    cancelled = False
    stopped_by_callback = False
    started = time.perf_counter()
    try:
        while max_frames <= 0 or frame_count < max_frames:
            if cancellation_event is not None and cancellation_event.is_set():
                cancelled = True
                break
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            synchronize(device)
            tick = time.perf_counter()
            result = apply_threshold(
                dict(runtime.predict(rgb, f"frame-{frame_count:06d}")), threshold
            )
            synchronize(device)
            latency_ms = (time.perf_counter() - tick) * 1000.0
            latencies.append(latency_ms)

            anomaly = bool(result["decision"])
            anomaly_frames += int(anomaly)
            if anomaly and not previous_anomaly:
                anomaly_events += 1
            previous_anomaly = anomaly

            annotated = draw_result(
                frame,
                category=category,
                result=result,
                latency_ms=latency_ms,
            )
            writer.write(annotated)
            frame_count += 1

            if progress_callback is not None:
                progress_callback(frame_count, total_frames)
            if frame_callback is not None and frame_callback(annotated, result) is False:
                stopped_by_callback = True
                break
            if cancellation_event is not None and cancellation_event.is_set():
                cancelled = True
                break
    finally:
        capture.release()
        writer.release()

    if not latencies and not cancelled:
        raise ValueError("the video source produced no frames")

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "source": str(resolved_source),
        "output": str(output),
        "category": category,
        "device": device,
        "resolution": [width, height],
        "source_fps": fps,
        "frames": frame_count,
        "anomaly_frames": anomaly_frames,
        "anomaly_frame_ratio": anomaly_frames / frame_count if frame_count else 0.0,
        "anomaly_events": anomaly_events,
        "threshold_override": None if threshold is None else float(threshold),
        "feedback_state": dict(feedback_state or {}),
        "cancelled": cancelled,
        "stopped_by_callback": stopped_by_callback,
        "elapsed_seconds": elapsed,
        "processing_fps": frame_count / max(elapsed, 1e-9),
        "latency_ms": _latency_summary(latencies),
        "latency_scope": "model inference only; video decode, annotation, and encode excluded",
    }
    summary_file.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return summary
