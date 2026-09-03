from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

import cv2
import numpy as np
import yaml

from aoi.video import run_video_inspection


class _FakeRuntime:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.sample_ids: list[str] = []

    def predict(self, _rgb: np.ndarray, sample_id: str) -> dict[str, Any]:
        self.sample_ids.append(sample_id)
        score = self.scores[len(self.sample_ids) - 1]
        return {
            "sample_id": sample_id,
            "fusion_score": score,
            "fusion_threshold": 0.5,
            "decision": score >= 0.5,
        }


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (width, height)
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def test_video_pipeline_applies_threshold_and_writes_outputs(tmp_path: Path) -> None:
    frames = [np.full((72, 96, 3), value, dtype=np.uint8) for value in (20, 220)]
    source = tmp_path / "source.avi"
    _write_video(source, frames)
    output = tmp_path / "annotated.mp4"
    summary_path = tmp_path / "summary.yaml"
    runtime = _FakeRuntime([0.2, 0.9])
    progress: list[tuple[int, int | None]] = []

    summary = run_video_inspection(
        source,
        runtime=runtime,
        category="part-a",
        device="cpu",
        output_path=output,
        summary_path=summary_path,
        threshold=0.8,
        feedback_state={"threshold_source": "active_operator_feedback"},
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert runtime.sample_ids == ["frame-000000", "frame-000001"]
    assert progress == [(1, 2), (2, 2)]
    assert summary["frames"] == 2
    assert summary["anomaly_frames"] == 1
    assert summary["anomaly_events"] == 1
    assert summary["threshold_override"] == 0.8
    assert summary["latency_scope"].startswith("model inference only")
    assert yaml.safe_load(summary_path.read_text(encoding="utf-8"))["frames"] == 2
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    finally:
        capture.release()


def test_video_pipeline_stops_after_cancellation(tmp_path: Path) -> None:
    frames = [np.full((72, 96, 3), value, dtype=np.uint8) for value in range(4)]
    source = tmp_path / "source.avi"
    _write_video(source, frames)
    cancelled = Event()
    progress: list[tuple[int, int | None]] = []

    def request_cancel(current: int, total: int | None) -> None:
        progress.append((current, total))
        cancelled.set()

    runtime = _FakeRuntime([0.1] * len(frames))
    summary = run_video_inspection(
        source,
        runtime=runtime,
        category="part-a",
        device="cpu",
        output_path=tmp_path / "annotated.mp4",
        summary_path=tmp_path / "summary.yaml",
        progress_callback=request_cancel,
        cancellation_event=cancelled,
    )

    assert summary["frames"] == 1
    assert summary["cancelled"] is True
    assert progress == [(1, 4)]
    assert runtime.sample_ids == ["frame-000000"]


def test_video_pipeline_can_stop_from_frame_callback(tmp_path: Path) -> None:
    frames = [np.zeros((72, 96, 3), dtype=np.uint8) for _ in range(3)]
    source = tmp_path / "source.avi"
    _write_video(source, frames)

    summary = run_video_inspection(
        source,
        runtime=_FakeRuntime([0.1, 0.1, 0.1]),
        category="part-a",
        device="cpu",
        output_path=tmp_path / "annotated.mp4",
        summary_path=tmp_path / "summary.yaml",
        frame_callback=lambda _frame, _result: False,
    )

    assert summary["frames"] == 1
    assert summary["stopped_by_callback"] is True
    assert summary["cancelled"] is False
