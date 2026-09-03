from __future__ import annotations

import argparse
from pathlib import Path

from aoi.training.config import load_config
from aoi.training.dagm import build_eligible_episodes as build_dagm_episodes
from aoi.training.mvtec import build_eligible_episodes as build_mvtec_episodes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an explicitly configured DAGM or MVTec dataset."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    builder = build_dagm_episodes if config.dataset == "dagm2007" else build_mvtec_episodes
    episodes, skipped = builder(
        config.data_root,
        categories=config.categories,
        normal_count=config.support_normal_count,
        defect_count=config.support_defect_count,
        seed=config.seed,
        minimum_holdout=config.minimum_holdout_per_category,
    )
    total = sum(e.independent_test_count for e in episodes)
    print(f"=== {config.dataset.upper()} DATASET CHECK ===")
    print(f"Root     : {config.data_root}")
    print(f"Eligible : {len(episodes)}")
    for e in episodes:
        print(
            f"  {e.category:14s} support={config.support_normal_count}+{config.support_defect_count} "
            f"holdout={e.independent_test_count} ({e.test_normal_count} normal + {e.test_defect_count} defect)"
        )
    if skipped:
        print("Skipped:")
        for row in skipped:
            print(f"  {row['category']}: {row['reason']}")
    print(f"Pooled independent holdout: {total}")
    print(f"Required minimum          : {config.minimum_pooled_test}")
    print("DATASET_READY" if total >= config.minimum_pooled_test else "DATASET_NOT_READY")
    return 0 if total >= config.minimum_pooled_test else 2


if __name__ == "__main__":
    raise SystemExit(main())
