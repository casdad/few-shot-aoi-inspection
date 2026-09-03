from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import aoi.cli.custom_evaluation as custom_evaluation
import aoi.cli.custom_training as custom_training
from aoi.custom_manifest import CustomTestSample


ROOT = Path(__file__).resolve().parents[1]


def _support(tmp_path: Path) -> tuple[Path, Path]:
    normal = tmp_path / "normal"
    defect = tmp_path / "defect"
    normal.mkdir()
    defect.mkdir()
    for index in range(100):
        (normal / f"normal-{index:03d}.png").write_bytes(
            f"normal-{index}".encode()
        )
    for index in range(30):
        (defect / f"defect-{index:03d}.png").write_bytes(
            f"defect-{index}".encode()
        )
    return normal, defect


def test_prepare_and_run_noninteractive_custom_training(
    tmp_path: Path, monkeypatch, require_training_assets
) -> None:
    normal, defect = _support(tmp_path)
    output_root = tmp_path / "output"
    config_path = tmp_path / "custom_train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base_training_config": str(ROOT / "configs" / "train.yaml"),
                "task_name": "part_a",
                "category": "housing",
                "normal_dir": str(normal),
                "defect_dir": str(defect),
                "device": "cpu",
                "head": {"epochs": 3, "learning_rate": 0.02},
                "output_root": str(output_root),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config, episode, refuse_overwrite = custom_training.prepare_custom_training(
        config_path
    )

    assert config.dataset == "custom"
    assert config.run_name == "part_a"
    assert config.output_root == output_root.resolve()
    assert config.device == "cpu"
    assert config.head_epochs == 3
    assert config.head_learning_rate == 0.02
    assert episode.category == "housing"
    assert len(episode.dataset.normal) == 100
    assert len(episode.dataset.defect) == 30
    assert refuse_overwrite is True

    expected_pointer = output_root / "active_models.yaml"

    def fake_train(received_config, received_episode):
        assert received_config == config
        assert received_episode == episode
        return expected_pointer

    monkeypatch.setattr(custom_training, "train_custom_episode", fake_train)
    assert custom_training.run(config_path) == 0


class _FakeRuntime:
    def __init__(self, _bundle_path: Path, *, device: str) -> None:
        assert device == "cpu"
        self.integrity = {
            "bundle_hash_verified": True,
            "artifact_hashes_verified": True,
            "legacy_bundle_without_artifact_hashes": False,
        }

    def predict_batch(self, _images, ids):
        rows = []
        for sample_id in ids:
            defect = sample_id.startswith("defect")
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


class _ThresholdRuntime:
    def __init__(self, _bundle_path: Path, *, device: str) -> None:
        assert device == "cpu"
        self.integrity = {
            "bundle_hash_verified": True,
            "artifact_hashes_verified": True,
            "legacy_bundle_without_artifact_hashes": False,
        }

    def predict_batch(self, _images, ids):
        rows = []
        for sample_id in ids:
            is_defect = sample_id.startswith("defect")
            prediction = is_defect
            if sample_id.endswith("-000"):
                prediction = not prediction
            score = 0.9 if prediction else 0.1
            rows.append(
                {
                    "decision": prediction,
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
                    "general_decision": prediction,
                    "local_score": score,
                    "local_threshold": 0.5,
                    "local_decision": prediction,
                    "known_score": score,
                    "known_threshold": 0.5,
                    "known_decision": prediction,
                    "branch_margins": {
                        "general": score - 0.5,
                        "local": score - 0.5,
                        "known": score - 0.5,
                    },
                }
            )
        return rows


def test_prepare_custom_evaluation_validates_pointer_bundle_and_config(
    tmp_path: Path, monkeypatch
) -> None:
    bundle_path = tmp_path / "model_bundle.yaml"
    support_hashes = [f"{index:064x}" for index in range(130)]
    bundle_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "target_dataset": "custom",
                "target_object": "part_a",
                "support_content_hashes": support_hashes,
                "support_counts": {"normal": 100, "defect": 30},
            }
        ),
        encoding="utf-8",
    )
    pointer_path = tmp_path / "active_models.yaml"
    pointer_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": "custom",
                "models": {"part_a": "model_bundle.yaml"},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
    output_root = tmp_path / "results"
    config_path = tmp_path / "custom_test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "active_models_pointer": str(pointer_path),
                "test_manifest": str(manifest_path),
                "device": "cpu",
                "batch_size": 8,
                "minimum_total_images": 1000,
                "output_root": str(output_root),
                "timestamp_output": False,
                "refuse_overwrite": True,
                "evaluation_role": "independent_test",
                "fresh_independent_test_attested": True,
                "test_observed_before_freeze": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sample = CustomTestSample(
        sample_id="normal-1",
        path=tmp_path / "normal.png",
        label=0,
        defect_type="normal",
        sha256="b" * 64,
    )
    calls = {}

    def fake_manifest(path, *, minimum_samples, excluded_hashes):
        calls.update(
            path=path,
            minimum_samples=minimum_samples,
            excluded_hashes=excluded_hashes,
        )
        return (sample,)

    monkeypatch.setattr(
        custom_evaluation, "load_custom_test_manifest", fake_manifest
    )

    prepared = custom_evaluation.prepare_custom_evaluation(config_path)

    assert prepared.pointer_path == pointer_path.resolve()
    assert prepared.bundle_path == bundle_path.resolve()
    assert prepared.category == "part_a"
    assert prepared.device == "cpu"
    assert prepared.batch_size == 8
    assert prepared.output_root == output_root.resolve()
    assert prepared.evaluation_role == "independent_test"
    assert prepared.fresh_independent_test_attested is True
    assert prepared.test_observed_before_freeze is False
    assert prepared.recall_floor == 0.95
    assert prepared.stage1_fpr_strict_upper_bound == 0.10
    assert prepared.stage2_fpr_strict_upper_bound == 0.05
    assert calls == {
        "path": manifest_path.resolve(),
        "minimum_samples": 1000,
        "excluded_hashes": set(support_hashes),
    }

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["evaluation_role"] = "independent_test"
    payload["minimum_total_images"] = 80
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="use development_diagnostic"):
        custom_evaluation.prepare_custom_evaluation(config_path)

    payload["evaluation_role"] = "development_diagnostic"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    calls.clear()
    prepared = custom_evaluation.prepare_custom_evaluation(config_path)
    assert prepared.evaluation_role == "development_diagnostic"
    assert prepared.minimum_total_images == 80
    assert calls == {
        "path": manifest_path.resolve(),
        "minimum_samples": 80,
        "excluded_hashes": set(support_hashes),
    }

    payload["acceptance_thresholds"] = {
        "recall_floor": 0.90,
        "stage1_fpr_strict_upper_bound": 0.12,
        "stage2_fpr_strict_upper_bound": 0.04,
    }
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    prepared = custom_evaluation.prepare_custom_evaluation(config_path)
    assert prepared.recall_floor == 0.90
    assert prepared.stage1_fpr_strict_upper_bound == 0.12
    assert prepared.stage2_fpr_strict_upper_bound == 0.04


@pytest.mark.parametrize(
    ("thresholds", "error_type", "message"),
    [
        (None, TypeError, "must be a YAML mapping"),
        ([], TypeError, "must be a YAML mapping"),
        ({"unexpected": 0.1}, ValueError, "unsupported fields"),
        ({"recall_floor": True}, TypeError, "must be a finite number"),
        (
            {"stage1_fpr_strict_upper_bound": "0.1"},
            TypeError,
            "must be a finite number",
        ),
        ({"recall_floor": float("nan")}, ValueError, "must be finite"),
        ({"recall_floor": -0.01}, ValueError, r"must be in \[0, 1\]"),
        ({"recall_floor": 1.01}, ValueError, r"must be in \[0, 1\]"),
        (
            {"stage1_fpr_strict_upper_bound": 0.0},
            ValueError,
            r"must be in \(0, 1\]",
        ),
        (
            {"stage2_fpr_strict_upper_bound": 1.01},
            ValueError,
            r"must be in \(0, 1\]",
        ),
        (
            {
                "stage1_fpr_strict_upper_bound": 0.04,
                "stage2_fpr_strict_upper_bound": 0.05,
            },
            ValueError,
            "must be <=",
        ),
    ],
)
def test_custom_evaluation_rejects_invalid_acceptance_thresholds(
    tmp_path: Path,
    thresholds,
    error_type: type[Exception],
    message: str,
) -> None:
    config_path = tmp_path / "custom_test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "acceptance_thresholds": thresholds,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(error_type, match=message):
        custom_evaluation.prepare_custom_evaluation(config_path)


@pytest.mark.parametrize(
    ("fresh_attested", "observed_before_freeze", "claim_eligible"),
    [
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_custom_labeled_evaluation_writes_metrics_and_predictions(
    tmp_path: Path,
    monkeypatch,
    fresh_attested: bool,
    observed_before_freeze: bool,
    claim_eligible: bool,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
    bundle_path = tmp_path / "model_bundle.yaml"
    bundle_path.write_text("schema_version: 3\n", encoding="utf-8")
    pointer_path = tmp_path / "active_models.yaml"
    pointer_path.write_text("schema_version: 1\n", encoding="utf-8")
    output_root = tmp_path / "results"
    config_path = tmp_path / "test.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    image_path = tmp_path / "placeholder.png"
    image_path.write_bytes(b"placeholder")
    samples = tuple(
        CustomTestSample(
            sample_id=("defect" if index % 2 else "normal") + f"-{index:04d}",
            path=image_path,
            label=index % 2,
            defect_type="scratch" if index % 2 else "normal",
            sha256=f"{index:064x}",
        )
        for index in range(1000)
    )
    prepared = custom_evaluation.CustomEvaluation(
        config_path=config_path,
        pointer_path=pointer_path,
        bundle_path=bundle_path,
        category="part_a",
        manifest_path=manifest_path,
        samples=samples,
        device="cpu",
        batch_size=100,
        output_root=output_root,
        refuse_overwrite=True,
        evaluation_role="independent_test",
        fresh_independent_test_attested=fresh_attested,
        test_observed_before_freeze=observed_before_freeze,
    )
    monkeypatch.setattr(
        custom_evaluation, "prepare_custom_evaluation", lambda _path: prepared
    )
    monkeypatch.setattr(custom_evaluation, "AdaptedRuntime", _FakeRuntime)
    monkeypatch.setattr(
        custom_evaluation,
        "read_rgb",
        lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert custom_evaluation.run(config_path) == 0

    metrics = yaml.safe_load(
        (output_root / "metrics.yaml").read_text(encoding="utf-8")
    )
    predictions = yaml.safe_load(
        (output_root / "predictions.yaml").read_text(encoding="utf-8")
    )
    assert metrics["images"] == 1000
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["label_counts"] == {"normal": 500, "defect": 500}
    assert metrics["per_defect_type"]["scratch"]["recall"] == 1.0
    assert metrics["support_test_content_overlap"] is False
    assert metrics["eligible_for_final_independent_claim"] is claim_eligible
    expected_scope = (
        "INDEPENDENT_TEST" if claim_eligible else "DEVELOPMENT_DIAGNOSTIC"
    )
    assert metrics["acceptance"]["status"] == f"GO_STAGE2_{expected_scope}"
    assert metrics["acceptance"]["recall_floor"] == 0.95
    assert metrics["acceptance"]["stage1_fpr_strict_upper_bound"] == 0.10
    assert metrics["acceptance"]["stage2_fpr_strict_upper_bound"] == 0.05
    assert metrics["claim_protocol_verification"] == {
        "protocol_id": "custom_private_100_plus_30_to_1000_plus_v1",
        "observed_total_images": 1000,
        "minimum_total_images": 1000,
        "sample_ids_unique": True,
        "prediction_sample_hashes_present": True,
        "prediction_sample_hashes_unique": True,
        "bundle_and_artifact_integrity_verified": True,
    }
    assert metrics["evidence"]["predictions"]["sha256"]
    assert metrics["evidence"]["model_bundle"]["runtime_integrity"] == {
        "bundle_hash_verified": True,
        "artifact_hashes_verified": True,
        "legacy_bundle_without_artifact_hashes": False,
    }
    assert len(predictions) == 1000


def test_small_labeled_evaluation_is_development_diagnostic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
    bundle_path = tmp_path / "model_bundle.yaml"
    bundle_path.write_text("schema_version: 3\n", encoding="utf-8")
    pointer_path = tmp_path / "active_models.yaml"
    pointer_path.write_text("schema_version: 1\n", encoding="utf-8")
    output_root = tmp_path / "diagnostic_results"
    config_path = tmp_path / "test.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    image_path = tmp_path / "placeholder.png"
    image_path.write_bytes(b"placeholder")
    samples = tuple(
        CustomTestSample(
            sample_id=("defect" if index % 2 else "normal") + f"-{index:04d}",
            path=image_path,
            label=index % 2,
            defect_type="scratch" if index % 2 else "normal",
            sha256=f"{index:064x}",
        )
        for index in range(80)
    )
    prepared = custom_evaluation.CustomEvaluation(
        config_path=config_path,
        pointer_path=pointer_path,
        bundle_path=bundle_path,
        category="part_a",
        manifest_path=manifest_path,
        samples=samples,
        device="cpu",
        batch_size=20,
        output_root=output_root,
        refuse_overwrite=True,
        evaluation_role="development_diagnostic",
        fresh_independent_test_attested=False,
        test_observed_before_freeze=True,
        minimum_total_images=80,
    )
    monkeypatch.setattr(
        custom_evaluation, "prepare_custom_evaluation", lambda _path: prepared
    )
    monkeypatch.setattr(custom_evaluation, "AdaptedRuntime", _FakeRuntime)
    monkeypatch.setattr(
        custom_evaluation,
        "read_rgb",
        lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert custom_evaluation.run(config_path) == 0

    metrics = yaml.safe_load(
        (output_root / "metrics.yaml").read_text(encoding="utf-8")
    )
    assert metrics["images"] == 80
    assert metrics["evaluation_role"] == "development_diagnostic"
    assert metrics["configured_minimum_total_images"] == 80
    assert metrics["eligible_for_final_independent_claim"] is False
    assert metrics["acceptance"]["status"] == "GO_STAGE2_DEVELOPMENT_DIAGNOSTIC"
    assert "development holdout diagnostic" in metrics["protocol"]
    assert "at least 1000" in metrics["claim_eligibility_reason"]


def test_custom_acceptance_thresholds_drive_gate_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
    bundle_path = tmp_path / "model_bundle.yaml"
    bundle_path.write_text("schema_version: 3\n", encoding="utf-8")
    pointer_path = tmp_path / "active_models.yaml"
    pointer_path.write_text("schema_version: 1\n", encoding="utf-8")
    output_root = tmp_path / "custom_threshold_results"
    config_path = tmp_path / "test.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    image_path = tmp_path / "placeholder.png"
    image_path.write_bytes(b"placeholder")
    samples = tuple(
        CustomTestSample(
            sample_id=f"{label}-{index:03d}",
            path=image_path,
            label=int(label == "defect"),
            defect_type="scratch" if label == "defect" else "normal",
            sha256=f"{offset:064x}",
        )
        for offset, (label, index) in enumerate(
            (label, index)
            for label in ("normal", "defect")
            for index in range(10)
        )
    )
    prepared = custom_evaluation.CustomEvaluation(
        config_path=config_path,
        pointer_path=pointer_path,
        bundle_path=bundle_path,
        category="part_a",
        manifest_path=manifest_path,
        samples=samples,
        device="cpu",
        batch_size=10,
        output_root=output_root,
        refuse_overwrite=True,
        evaluation_role="development_diagnostic",
        fresh_independent_test_attested=False,
        test_observed_before_freeze=True,
        minimum_total_images=20,
        recall_floor=0.90,
        stage1_fpr_strict_upper_bound=0.11,
        stage2_fpr_strict_upper_bound=0.10,
    )
    monkeypatch.setattr(
        custom_evaluation, "prepare_custom_evaluation", lambda _path: prepared
    )
    monkeypatch.setattr(custom_evaluation, "AdaptedRuntime", _ThresholdRuntime)
    monkeypatch.setattr(
        custom_evaluation,
        "read_rgb",
        lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert custom_evaluation.run(config_path) == 0

    metrics = yaml.safe_load(
        (output_root / "metrics.yaml").read_text(encoding="utf-8")
    )
    assert metrics["recall"] == 0.9
    assert metrics["fpr"] == 0.1
    assert metrics["acceptance"] == {
        "recall_floor": 0.9,
        "stage1_fpr_strict_upper_bound": 0.11,
        "stage2_fpr_strict_upper_bound": 0.1,
        "stage1_go": True,
        "stage2_go": False,
        "status": "GO_STAGE1_DEVELOPMENT_DIAGNOSTIC",
    }
