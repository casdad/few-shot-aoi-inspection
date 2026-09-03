from __future__ import annotations

import argparse
from pathlib import Path

from aoi.cli.custom_training import run as run_custom_training
from aoi.training.pipeline import run_training


def main() -> int:
    demo_config = Path(__file__).resolve().parent / "configs" / "train_dagm_demo.yaml"
    parser = argparse.ArgumentParser(
        description="Train a private AOI product task or an explicit DAGM demo."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("custom", "dagm-demo", "standard"),
        default=None,
        help="private 100+30 task, or explicit DAGM demo training",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="training YAML",
    )
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        print(
            "\nProduction workflow: run feedback_ui.py. CLI training requires "
            "custom --config; DAGM is available only through dagm-demo."
        )
        return 0
    if args.mode == "custom":
        if args.config is None:
            parser.error("custom mode requires --config")
        return run_custom_training(args.config)
    # ``standard`` is retained as a compatibility alias, but it is no longer
    # an implicit default and follows the explicit DAGM demo configuration.
    config = args.config or demo_config
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
