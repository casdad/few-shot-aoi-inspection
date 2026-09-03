from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from aoi.cli import evaluation
from aoi.runtime import runtime as runtime_implementation
from aoi.runtime.data import EvalSample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeRuntime:
    def __init__(self, bundle_path: Path, *, device: str) -> None:
        assert bundle_path.is_file()
        assert device == "cpu"
        self.calibration: dict[str, object] = {}
        self.integrity = {
            "bundle_hash_verified": True,
            "artifact_hashes_verified": True,
        }

    def predict_batch(self, _images, ids):
        rows = []
        for sample_id in ids:
            defect = sample_id.endswith("/defect")
            score = 0.9 if defect else 0.1
            rows.append(
                {
                    "decision": defect,
                    "fusion_mode": "test_fusion",
                    "fusion_score": score,
                    "fusion_threshold": 0.5,
                    "fusion_weights": {
                        "general": 1.0,
                        "local": 0.0,
                        "known": 0.0,
                    },
                    "general_score": score,
                    "general_threshold": 0.5,
                    "general_decision": defect,
                    "local_score": score,
                    "local_threshold": 0.5,
                    "local_decision": defect,
                    "known_score": score,
                    "known_threshold": 0.5,
                    "known_decision": defect,
                    "branch_margins": {
                        "general": score - 0.5,
                        "local": score - 0.5,
                        "known": score - 0.5,
                    },
                }
            )
        return rows


def test_independent_test_attestation_requires_yaml_boolean() -> None:
    with pytest.raises(TypeError, match="must be a YAML boolean"):
        evaluation._config_boolean(
            {"fresh_independent_test_attested": "false"},
            "fresh_independent_test_attested",
            default=False,
        )


def _complete_dagm_claim_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, count in evaluation._DAGM2007_FINAL_CLAIM_COUNTS:
        for index in range(count):
            sample_id = f"{category}/Test/{index}"
            rows.append(
                {
                    "category": category,
                    "sample_id": sample_id,
                    "image_sha256": hashlib.sha256(
                        sample_id.encode("utf-8")
                    ).hexdigest(),
                }
            )
    return rows


def _complete_dagm_model_evidence(
    *, artifact_hashes_verified: bool = True
) -> dict[str, dict[str, object]]:
    return {
        category: {
            "bundle_hash": hashlib.sha256(category.encode()).hexdigest(),
            "runtime_integrity": {
                "bundle_hash_verified": True,
                "artifact_hashes_verified": artifact_hashes_verified,
                "legacy_bundle_without_artifact_hashes": False,
            },
        }
        for category, _count in evaluation._DAGM2007_FINAL_CLAIM_COUNTS
    }


def test_final_claim_requires_immutable_complete_coverage_and_integrity() -> None:
    eligible, reason, verification = evaluation._final_claim_assessment(
        dataset="dagm2007",
        evaluation_role="independent_test",
        fresh_independent_test_attested=True,
        test_observed_before_freeze=False,
        prediction_rows=_complete_dagm_claim_rows(),
        model_bundle_evidence=_complete_dagm_model_evidence(),
    )

    assert eligible is True
    assert "verified" in reason
    assert verification["complete_coverage_verified"] is True
    assert verification["bundle_and_artifact_integrity_verified"] is True


def test_final_claim_cannot_be_unlocked_by_lowering_minimum_or_weak_integrity() -> None:
    rows = _complete_dagm_claim_rows()[:-1]
    eligible, reason, verification = evaluation._final_claim_assessment(
        dataset="dagm2007",
        evaluation_role="independent_test",
        fresh_independent_test_attested=True,
        test_observed_before_freeze=False,
        prediction_rows=rows,
        model_bundle_evidence=_complete_dagm_model_evidence(
            artifact_hashes_verified=False
        ),
    )

    assert eligible is False
    assert "8050-image coverage" in reason
    assert "artifact hash" in reason
    assert verification["complete_coverage_verified"] is False
    assert verification["bundle_and_artifact_integrity_verified"] is False


def test_standard_evaluation_records_provenance_and_binds_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    bundle_dir = tmp_path / "outputs" / "models"
    config_dir.mkdir()
    bundle_dir.mkdir(parents=True)

    bundle_paths: dict[str, Path] = {}
    bundle_hashes = {"Class1": "1" * 64, "Class2": "2" * 64}
    for category in ("Class1", "Class2"):
        bundle_path = bundle_dir / f"{category}.yaml"
        bundle_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 3,
                    "target_dataset": "dagm2007",
                    "target_object": category,
                    "support_content_hashes": [],
                    "bundle_hash": bundle_hashes[category],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        bundle_paths[category] = bundle_path

    pointer_path = tmp_path / "outputs" / "active_models.yaml"
    pointer_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": "dagm2007",
                "models": {
                    "Class2": "models/Class2.yaml",
                    "Class1": "models/Class1.yaml",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs" / "evaluation"
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": "dagm2007",
                "data_root": "data/dagm2007",
                "active_models_pointer": "outputs/active_models.yaml",
                "device": "cpu",
                "batch_size": 2,
                "minimum_total_images": 4,
                "output_root": "outputs/evaluation",
                "timestamp_output": False,
                "refuse_overwrite": True,
                "fn_diagnostic_categories": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    image_path = tmp_path / "placeholder.png"
    image_path.write_bytes(b"placeholder")
    expected_data_root = (tmp_path / "data" / "dagm2007").resolve()

    def fake_loader(
        data_root: Path, category: str, *, excluded_hashes: set[str]
    ) -> list[EvalSample]:
        assert data_root == expected_data_root
        assert excluded_hashes == set()
        return [
            EvalSample(
                sample_id=f"{category}/Test/normal",
                path=image_path,
                label=0,
                defect_type="normal",
                category=category,
                sha256="a" * 64,
            ),
            EvalSample(
                sample_id=f"{category}/Test/defect",
                path=image_path,
                label=1,
                defect_type="defect",
                category=category,
                sha256="b" * 64,
            ),
        ]

    monkeypatch.setattr(evaluation, "load_dagm_category_test", fake_loader)
    monkeypatch.setattr(evaluation, "AdaptedRuntime", _FakeRuntime)
    monkeypatch.setattr(
        evaluation,
        "read_rgb",
        lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    predictions_path = output_root / "predictions.yaml"
    metrics_path = output_root / "metrics.yaml"
    output_writes: list[str] = []
    original_write_text = Path.write_text

    def recording_write_text(self: Path, *args, **kwargs):
        if self.resolve() in {predictions_path.resolve(), metrics_path.resolve()}:
            output_writes.append(self.name)
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", recording_write_text)

    assert evaluation.run(config_path) == 0
    assert output_writes == ["predictions.yaml", "metrics.yaml"]

    metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    predictions = yaml.safe_load(predictions_path.read_text(encoding="utf-8"))
    assert metrics["images"] == 4
    assert metrics["evaluation_role"] == "independent_test"
    assert metrics["fresh_independent_test_attested"] is False
    assert metrics["eligible_for_final_independent_claim"] is False
    assert metrics["acceptance"]["status"] == "GO_STAGE2_DEVELOPMENT_DIAGNOSTIC"
    assert len(predictions) == 4
    assert predictions[0]["image_sha256"] == "a" * 64
    assert metrics["eligible_for_final_independent_claim"] is False
    assert metrics["fresh_independent_test_attested"] is False
    assert metrics["claim_eligibility_reason"].startswith("requires a fresh")
    evidence = metrics["evidence"]

    assert evidence["config"] == {
        "path": str(config_path.resolve()),
        "sha256": _sha256(config_path),
    }
    assert evidence["active_models_pointer"] == {
        "path": str(pointer_path.resolve()),
        "sha256": _sha256(pointer_path),
    }
    assert list(evidence["model_bundles"]) == ["Class1", "Class2"]
    for category in ("Class1", "Class2"):
        assert evidence["model_bundles"][category] == {
            "path": str(bundle_paths[category].resolve()),
            "sha256": _sha256(bundle_paths[category]),
            "bundle_hash": bundle_hashes[category],
            "runtime_integrity": {
                "bundle_hash_verified": True,
                "artifact_hashes_verified": True,
            },
        }

    evaluation_path = Path(evaluation.__file__).resolve()
    runtime_path = Path(runtime_implementation.__file__).resolve()
    assert evidence["implementation"]["evaluation"] == {
        "path": str(evaluation_path),
        "sha256": _sha256(evaluation_path),
    }
    assert evidence["implementation"]["runtime"] == {
        "path": str(runtime_path),
        "sha256": _sha256(runtime_path),
    }
    assert evidence["predictions"] == {
        "path": str(predictions_path.resolve()),
        "sha256": _sha256(predictions_path),
    }
