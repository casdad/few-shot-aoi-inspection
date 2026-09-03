from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aoi.cli.benchmarking import run as run_benchmark
from aoi.cli.custom_evaluation import run as run_custom_evaluation
from aoi.cli.evaluation import run as run_evaluation
from aoi.cli.inference import run as run_inference


CONFIG_NAMES = {
    "evaluate": "test.yaml",
    "infer": "infer.yaml",
    "benchmark": "benchmark.yaml",
    "custom": None,
}


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    # Feedback has its own verb-level parser while the historical evaluation,
    # inference and benchmark modes retain their flat interface.
    if raw_args and raw_args[0] == "feedback":
        from aoi.cli.feedback_optimization import main as run_feedback

        return run_feedback(raw_args[1:])
    parser = argparse.ArgumentParser(
        description="Test an adapted AOI model by evaluation, inference, or benchmark."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=tuple(CONFIG_NAMES),
        default=None,
        help="test mode; production and demo runs require an explicit YAML",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="explicit product or demo YAML config",
    )
    args = parser.parse_args(raw_args)
    if args.mode is None:
        parser.print_help()
        print(
            "\nProduction workflow: run feedback_ui.py or supply a product YAML. "
            "DAGM requires an explicit configs/*_dagm_demo.yaml file."
        )
        return 0
    if args.config is None:
        suggested = CONFIG_NAMES[args.mode]
        suffix = (
            f"；DAGM 演示示例为 configs/{suggested.replace('.yaml', '_dagm_demo.yaml')}"
            if suggested is not None
            else ""
        )
        parser.error(f"{args.mode} mode requires --config{suffix}")
    config = args.config
    runners = {
        "evaluate": run_evaluation,
        "infer": run_inference,
        "benchmark": run_benchmark,
        "custom": run_custom_evaluation,
    }
    return runners[args.mode](config)


if __name__ == "__main__":
    raise SystemExit(main())
