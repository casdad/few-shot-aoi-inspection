from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from aoi.feedback import FeedbackStore
from aoi.feedback.active_learning import query_unlabeled_pool, rank_candidates


def _png(value: int) -> bytes:
    image = Image.new("RGB", (12, 12), color=(value, value, value))
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_rank_candidates_combines_uncertainty_branch_disagreement_and_diversity() -> None:
    predictions = [
        {
            "sample_id": "near-a",
            "fusion_score": 0.49,
            "fusion_threshold": 0.50,
            "branch_margins": {"general": -0.2, "local": 0.2, "known": 0.0},
        },
        {
            "sample_id": "far-b",
            "fusion_score": 0.10,
            "fusion_threshold": 0.50,
            "branch_margins": {"general": -0.1, "local": -0.1, "known": -0.1},
        },
        {
            "sample_id": "rare-c",
            "fusion_score": 0.25,
            "fusion_threshold": 0.50,
            "branch_margins": {"general": -0.1, "local": -0.1, "known": -0.1},
        },
    ]
    embeddings = np.asarray([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    result = rank_candidates(
        predictions,
        budget=2,
        candidate_embeddings=embeddings,
        existing_embeddings=np.asarray([[1, 0]], dtype=np.float32),
    )
    assert [row["sample_id"] for row in result] == ["near-a", "rare-c"]
    assert result[0]["uncertainty_score"] > result[1]["uncertainty_score"]
    assert "threshold_near" in result[0]["query_reason"]
    assert "feature_rare" in result[1]["query_reason"]
    assert [row["query_rank"] for row in result] == [1, 2]


def test_query_pool_excludes_recorded_content_and_writes_only_manifest(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    duplicate = _png(10)
    (pool / "duplicate.png").write_bytes(duplicate)
    (pool / "candidate_a.png").write_bytes(_png(40))
    (pool / "candidate_b.png").write_bytes(_png(80))
    store = FeedbackStore(tmp_path / "feedback")
    store.save_feedback(
        image_bytes=duplicate,
        original_name="duplicate.png",
        category="Class1",
        model_version="run-1",
        result={
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
        },
        operator_label=0,
        defect_type="",
        note="already labeled",
        run_id="run-1",
    )

    class _Dino:
        normal_global_bank = np.asarray([[1.0, 0.0]], dtype=np.float32)
        defect_global_bank = np.asarray([[0.0, 1.0]], dtype=np.float32)

        def _features(self, images):
            return np.asarray(
                [[[float(image.mean()), 1.0]] for image in images], dtype=np.float32
            )

    class _Runtime:
        bundle = {"run_id": "run-1", "bundle_hash": "a" * 64}
        dino = _Dino()

        def predict_batch(self, images, ids):
            return [
                {
                    "fusion_score": 0.49 + 0.2 * index,
                    "fusion_threshold": 0.5,
                    "branch_margins": {
                        "general": -0.1,
                        "local": 0.1 if index == 0 else -0.1,
                        "known": 0.0,
                    },
                    "decision": False,
                }
                for index, _ in enumerate(images)
            ]

    result = query_unlabeled_pool(
        _Runtime(), store, "Class1", pool, budget=1, model_version="run-1",
        output="queries/named-query.yaml",
    )
    assert result["pool_count"] == 2
    assert result["selected_count"] == 1
    assert "duplicate.png" not in str(result["candidates"])
    manifest = store.root / result["manifest_path"]
    assert manifest.is_file()
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert payload["query_type"] == "operator_active_learning_pool"
    assert payload["candidates"][0]["query_status"] == "pending_label"
    assert len(store.records("Class1", model_version="run-1")) == 1

    with pytest.raises(PermissionError, match="inside feedback root"):
        query_unlabeled_pool(
            _Runtime(), store, "Class1", pool, budget=1, model_version="run-1",
            output="../escape.yaml",
        )
