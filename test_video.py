from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from aoi.feedback import FeedbackStore, apply_active_feedback
from aoi.video import draw_result as draw_result
from aoi.video import run_video_inspection
from aoi.video import synchronize as synchronize
from aoi.paths import PROJECT_ROOT, resolve_artifact
from aoi.runtime.config import resolve_device
from aoi.runtime.runtime import AdaptedRuntime


def parse_source(value: str) -> int | Path:
    """Treat an integer as a camera index and everything else as a video path."""
    try:
        return int(value)
    except ValueError:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve(strict=True)


def load_runtime(pointer_path: Path, category: str, device: str) -> AdaptedRuntime:
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8")) or {}
    models = pointer.get("models", {})
    if category not in models:
        raise KeyError(
            f"category {category!r} has no adapted model; available: {sorted(models)}"
        )
    bundle = resolve_artifact(models[category], base=pointer_path.parent)
    return AdaptedRuntime(bundle, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an adapted AOI model on a video file or camera stream."
    )
    parser.add_argument("source", help="video path or camera index, for example 0")
    parser.add_argument(
        "--category",
        required=True,
        help="production product name or explicit demo category",
    )
    parser.add_argument(
        "--models",
        type=Path,
        required=True,
        help="product or explicit demo active_models.yaml path",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "video_test.mp4",
        help="annotated MP4 output path",
    )
    parser.add_argument("--summary", type=Path, help="summary YAML output path")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 means all")
    parser.add_argument("--display", action="store_true", help="show a live preview; press q to stop")
    parser.add_argument(
        "--threshold",
        type=float,
        help="manual fusion-threshold override; takes precedence over active feedback",
    )
    parser.add_argument(
        "--feedback-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "feedback",
        help="operator-feedback store",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="run the frozen model bundle without active operator feedback",
    )
    args = parser.parse_args()

    source = parse_source(args.source)
    device = resolve_device(args.device)
    pointer = args.models.resolve(strict=True)
    output_path = args.output.resolve()
    summary_path = (args.summary or output_path.with_suffix(".yaml")).resolve()
    runtime = load_runtime(pointer, args.category, device)
    feedback_root = args.feedback_root
    if not feedback_root.is_absolute():
        feedback_root = PROJECT_ROOT / feedback_root
    feedback_state = apply_active_feedback(
        runtime,
        FeedbackStore(feedback_root.resolve()),
        args.category,
        enabled=not args.no_feedback,
    )
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else feedback_state.threshold
    )
    feedback_receipt = feedback_state.receipt()
    if args.threshold is not None:
        feedback_receipt["threshold"] = float(args.threshold)
        feedback_receipt["threshold_source"] = "manual_cli_override"
    def show_frame(frame, _result):
        cv2.imshow("AOI video test", frame)
        return cv2.waitKey(1) & 0xFF != ord("q")

    try:
        summary = run_video_inspection(
            source,
            runtime=runtime,
            category=args.category,
            device=device,
            output_path=output_path,
            summary_path=summary_path,
            threshold=threshold,
            feedback_state=feedback_receipt,
            max_frames=args.max_frames,
            frame_callback=show_frame if args.display else None,
        )
    finally:
        if args.display:
            cv2.destroyAllWindows()
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
