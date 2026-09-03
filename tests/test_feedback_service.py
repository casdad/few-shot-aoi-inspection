from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import aoi.feedback.service as feedback_service
from aoi.feedback import (
    DuplicateFeedbackError,
    FeedbackStore,
    apply_threshold,
    classification_metrics,
)


@pytest.mark.parametrize(
    (
        "parent",
        "candidate",
        "minimum_recall",
        "expected_mode",
        "expected_absolute",
        "expected_relative",
    ),
    [
        (
            {"recall": 0.80, "f1": 0.70},
            {"recall": 0.90, "f1": 0.70},
            0.90,
            "absolute_gate",
            True,
            True,
        ),
        (
            {"recall": 0.50, "f1": 0.40},
            {"recall": 0.50, "f1": 0.50},
            0.90,
            "relative_improvement",
            False,
            True,
        ),
        (
            {"recall": 0.50, "f1": 0.50},
            {"recall": 0.50, "f1": 0.50},
            0.90,
            "failed",
            False,
            False,
        ),
        (
            {"recall": 0.50, "f1": 0.50},
            {"recall": 0.75, "f1": 0.49},
            0.90,
            "failed",
            False,
            False,
        ),
        (
            {"recall": 0.50, "f1": 0.50},
            {"recall": 0.49, "f1": 0.60},
            0.90,
            "failed",
            False,
            False,
        ),
    ],
)
def test_revision_validation_acceptance_boundaries(
    parent: dict[str, float],
    candidate: dict[str, float],
    minimum_recall: float,
    expected_mode: str,
    expected_absolute: bool,
    expected_relative: bool,
) -> None:
    result = feedback_service._revision_validation_acceptance(
        parent,
        candidate,
        minimum_recall,
    )

    assert result["acceptance_mode"] == expected_mode
    assert result["passed"] is (expected_mode != "failed")
    assert result["absolute_gate_passed"] is expected_absolute
    assert result["relative_improvement_passed"] is expected_relative
    assert result["recall_delta"] == pytest.approx(
        candidate["recall"] - parent["recall"]
    )
    assert result["f1_delta"] == pytest.approx(candidate["f1"] - parent["f1"])


def test_revision_validation_acceptance_rejects_invalid_recall_target() -> None:
    with pytest.raises(ValueError, match="minimum_recall must be between 0 and 1"):
        feedback_service._revision_validation_acceptance(
            {"recall": 0.5, "f1": 0.5},
            {"recall": 0.6, "f1": 0.6},
            1.01,
        )


def _save_sample(
    store: FeedbackStore,
    *,
    category: str,
    name: str,
    dataset_role: str = "training",
) -> dict[str, object]:
    return store.save_feedback(
        image_bytes=f"image-{category}-{name}".encode(),
        original_name=f"{name}.png",
        category=category,
        model_version="test-model",
        result={
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
        },
        operator_label=0,
        defect_type="",
        note="",
        dataset_role=dataset_role,
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_signed_yaml(
    path: Path,
    payload: dict[str, object],
    *,
    hash_field: str,
) -> dict[str, object]:
    signed = dict(payload)
    signed[hash_field] = _canonical_sha256(signed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(signed, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return signed


def _save_scoped_sample(
    store: FeedbackStore,
    *,
    category: str,
    run_id: str,
    bundle_hash: str,
    name: str,
) -> dict[str, object]:
    return store.save_feedback(
        image_bytes=f"image-{category}-{run_id}-{bundle_hash[:8]}-{name}".encode(),
        original_name=f"{name}.png",
        category=category,
        model_version=run_id,
        result={
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
        },
        operator_label=0,
        defect_type="",
        note="",
        run_id=run_id,
        bundle_hash=bundle_hash,
    )


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_revision_receipt(
    store: FeedbackStore,
    *,
    revision_id: str,
    category: str,
    run_id: str,
    bundle_hash: str,
    feedback_id: str | None = None,
) -> Path:
    revision_dir = store.revisions / revision_id
    revision_dir.mkdir(parents=True)
    head_path = revision_dir / "head.pt"
    feature_path = revision_dir / "feature_bank.npz"
    head_path.write_bytes(f"head-{revision_id}".encode())
    feature_path.write_bytes(f"feature-{revision_id}".encode())
    snapshot = {
        "training": (
            [{"feedback_id": feedback_id}] if feedback_id is not None else []
        ),
        "validation": [],
    }
    receipt = store.candidates / f"{revision_id}.yaml"
    _write_signed_yaml(
        receipt,
        {
            "schema_version": 2,
            "revision_id": revision_id,
            "category": category,
            "run_id": run_id,
            "bundle_hash": bundle_hash,
            "artifacts": {
                "head": {"path": head_path.relative_to(store.root).as_posix()},
                "feature_bank": {
                    "path": feature_path.relative_to(store.root).as_posix()
                },
            },
            "feedback_snapshot": snapshot,
        },
        hash_field="payload_sha256",
    )
    return receipt


def _write_threshold_receipt(
    store: FeedbackStore,
    *,
    candidate_id: str,
    category: str,
    model_version: str,
) -> Path:
    path = store.candidates / f"{candidate_id}.yaml"
    _write_signed_yaml(
        path,
        {
            "schema_version": 2,
            "candidate_id": candidate_id,
            "category": category,
            "model_version": model_version,
        },
        hash_field="payload_sha256",
    )
    return path


def _write_query(
    store: FeedbackStore,
    *,
    query_id: str,
    category: str,
    run_id: str,
    bundle_hash: str,
) -> Path:
    path = store.root / "queries" / "nested" / f"{query_id}.yaml"
    _write_signed_yaml(
        path,
        {
            "schema_version": 1,
            "query_id": query_id,
            "category": category,
            "run_id": run_id,
            "bundle_hash": bundle_hash,
        },
        hash_field="manifest_sha256",
    )
    return path


def test_apply_threshold_preserves_base_threshold() -> None:
    result = {"fusion_score": 0.7, "fusion_threshold": 0.8, "decision": False}
    updated = apply_threshold(result, 0.6)
    assert updated["decision"] is True
    assert updated["fusion_threshold"] == 0.6
    assert updated["base_fusion_threshold"] == 0.8


def test_metrics_and_feedback_threshold_search(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 1]
    for index, (score, label) in enumerate(zip(scores, labels, strict=True)):
        store.save_feedback(
            image_bytes=f"image-{index}".encode(),
            original_name=f"sample-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": score >= 0.5,
                "fusion_score": score,
                "fusion_threshold": 0.5,
                "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
        )
    for index, (score, label) in enumerate(
        zip([0.15, 0.25, 0.85, 0.95], [0, 0, 1, 1], strict=True)
    ):
        store.save_feedback(
            image_bytes=f"validation-{index}".encode(),
            original_name=f"validation-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": score >= 0.5,
                "fusion_score": score,
                "fusion_threshold": 0.5,
                "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
            dataset_role="validation",
        )
    candidate = store.propose_threshold(
        "Class1", "test-model", base_threshold=0.95
    )
    assert candidate["candidate_metrics"]["recall"] == 1.0
    assert candidate["candidate_metrics"]["fpr"] == 0.0
    assert candidate["base_metrics"]["recall"] < 1.0
    assert candidate["validation"]["passed"] is True
    active = store.activate(candidate)
    assert store.active_threshold("Class1", "test-model") == active["threshold"]
    assert store.active_threshold("Class1", "other-model") is None
    store.rollback("Class1", "test-model")
    assert store.active_threshold("Class1", "test-model") is None
    assert classification_metrics(np.asarray(labels), np.asarray(scores), 0.5)["f1"] == 1.0


def test_threshold_activation_uses_persisted_receipt_not_caller_validation(
    tmp_path,
) -> None:
    store = FeedbackStore(tmp_path)
    for index, (score, label) in enumerate(
        zip([0.1, 0.2, 0.3, 0.8, 0.9, 1.0], [0, 0, 0, 1, 1, 1], strict=True)
    ):
        store.save_feedback(
            image_bytes=f"training-{index}".encode(),
            original_name=f"training-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": bool(label),
                "fusion_score": score,
                "fusion_threshold": 0.5,
                "branch_margins": {},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
        )
    for index, (score, label) in enumerate(
        zip([0.15, 0.25, 0.85, 0.95], [0, 0, 1, 1], strict=True)
    ):
        store.save_feedback(
            image_bytes=f"validation-{index}".encode(),
            original_name=f"validation-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": bool(label),
                "fusion_score": score,
                "fusion_threshold": 0.5,
                "branch_margins": {},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
            dataset_role="validation",
        )
    candidate = store.propose_threshold(
        "Class1", "test-model", base_threshold=0.95
    )
    forged = dict(candidate)
    forged["validation"] = {"passed": False, "reason": "forged caller data"}
    active = store.activate(forged)
    assert active["threshold"] == candidate["candidate_threshold"]
    assert active["validation_receipt"]["sha256"] == candidate["receipt_sha256"]


def test_threshold_activation_rejects_tampered_receipt_and_changed_records(
    tmp_path,
) -> None:
    store = FeedbackStore(tmp_path)
    for index, label in enumerate([0, 0, 0, 1, 1, 1]):
        store.save_feedback(
            image_bytes=f"training-{index}".encode(),
            original_name=f"training-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": bool(label),
                "fusion_score": 0.1 if not label else 0.9,
                "fusion_threshold": 0.5,
                "branch_margins": {},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
        )
    for index, label in enumerate([0, 0, 1, 1]):
        store.save_feedback(
            image_bytes=f"validation-{index}".encode(),
            original_name=f"validation-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": bool(label),
                "fusion_score": 0.1 if not label else 0.9,
                "fusion_threshold": 0.5,
                "branch_margins": {},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
            dataset_role="validation",
        )
    candidate = store.propose_threshold(
        "Class1", "test-model", base_threshold=0.95
    )
    receipt_path = store.candidates / f"{candidate['candidate_id']}.yaml"
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
        store.activate(candidate)

    # Rebuild a fresh receipt, then mutate a scoped record.  The persisted
    # snapshot must make activation fail even when the caller mapping is intact.
    store = FeedbackStore(tmp_path / "second")
    for index, label in enumerate([0, 0, 0, 1, 1, 1]):
        store.save_feedback(
            image_bytes=f"training-{index}".encode(),
            original_name=f"training-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": bool(label),
                "fusion_score": 0.1 if not label else 0.9,
                "fusion_threshold": 0.5,
                "branch_margins": {},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
        )
    for index, label in enumerate([0, 0, 1, 1]):
        store.save_feedback(
            image_bytes=f"validation-{index}".encode(),
            original_name=f"validation-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": bool(label),
                "fusion_score": 0.1 if not label else 0.9,
                "fusion_threshold": 0.5,
                "branch_margins": {},
            },
            operator_label=label,
            defect_type="defect" if label else "",
            note="",
            dataset_role="validation",
        )
    candidate = store.propose_threshold(
        "Class1", "test-model", base_threshold=0.95
    )
    rows = [
        json.loads(line)
        for line in store.records_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["note"] = "changed after proposal"
    store.records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="feedback snapshot"):
        store.activate(candidate)


def test_feature_bank_is_versioned_and_reloadable(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    training = []
    for index in range(2):
        training.append(
            store.save_feedback(
                image_bytes=f"defect-{index}".encode(),
                original_name=f"defect-{index}.png",
                category="Class1",
                model_version="test-model",
                result={
                    "decision": True,
                    "fusion_score": 0.9,
                    "fusion_threshold": 0.5,
                    "branch_margins": {},
                },
                operator_label=1,
                defect_type="appearance",
                note="",
                defect_region=[0, 0, 1, 1],
            )
        )
    validation = []
    for index, label in enumerate([0, 0, 1, 1]):
        validation.append(
            store.save_feedback(
                image_bytes=f"validation-{index}".encode(),
                original_name=f"validation-{index}.png",
                category="Class1",
                model_version="test-model",
                result={
                    "decision": bool(label),
                    "fusion_score": 0.1 if label == 0 else 0.9,
                    "fusion_threshold": 0.5,
                    "branch_margins": {},
                },
                operator_label=label,
                defect_type="appearance" if label else "",
                note="",
                dataset_role="validation",
            )
        )
    values = np.asarray([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    perfect = classification_metrics(
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]), 0.5
    )
    receipt = {
        "passed": True,
        "category": "Class1",
        "model_version": "test-model",
        "sample_ids": [str(row["feedback_id"]) for row in validation],
        "sample_count": 4,
        "normal_count": 2,
        "defect_count": 2,
        "minimum_recall": 0.9,
        "base_metrics": perfect,
        "candidate_metrics": perfect,
    }
    sample_ids = [str(row["feedback_id"]) for row in training]
    entry = store.save_feature_bank(
        "Class1", "test-model", values, sample_ids, validation=receipt
    )
    loaded, active = store.active_feature_bank("Class1", "test-model")
    assert active["version"] == entry["version"]
    assert active["sample_count"] == 2
    assert np.allclose(np.linalg.norm(loaded, axis=1), 1.0)
    assert store.active_feature_bank("Class1", "other-model") is None


def test_feature_bank_activation_verifies_file_sha256(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    training = []
    for index in range(2):
        training.append(
            store.save_feedback(
                image_bytes=f"defect-{index}".encode(),
                original_name=f"defect-{index}.png",
                category="Class1",
                model_version="test-model",
                result={
                    "decision": True,
                    "fusion_score": 0.9,
                    "fusion_threshold": 0.5,
                    "branch_margins": {},
                },
                operator_label=1,
                defect_type="appearance",
                note="",
                defect_region=[0, 0, 1, 1],
            )
        )
    validation = []
    for index, label in enumerate([0, 0, 1, 1]):
        validation.append(
            store.save_feedback(
                image_bytes=f"validation-{index}".encode(),
                original_name=f"validation-{index}.png",
                category="Class1",
                model_version="test-model",
                result={
                    "decision": bool(label),
                    "fusion_score": 0.1 if not label else 0.9,
                    "fusion_threshold": 0.5,
                    "branch_margins": {},
                },
                operator_label=label,
                defect_type="appearance" if label else "",
                note="",
                dataset_role="validation",
            )
        )
    metrics = classification_metrics(
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]), 0.5
    )
    entry = store.save_feature_bank(
        "Class1",
        "test-model",
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [str(row["feedback_id"]) for row in training],
        validation={
            "passed": True,
            "category": "Class1",
            "model_version": "test-model",
            "sample_ids": [str(row["feedback_id"]) for row in validation],
            "sample_count": 4,
            "normal_count": 2,
            "defect_count": 2,
            "minimum_recall": 0.9,
            "base_metrics": metrics,
            "candidate_metrics": metrics,
        },
    )
    (store.root / entry["path"]).write_bytes(b"tampered feature bank")
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        store.active_feature_bank("Class1", "test-model")


def test_duplicate_feedback_content_is_rejected_within_model_scope(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    kwargs = {
        "image_bytes": b"same-content",
        "original_name": "sample.png",
        "category": "Class1",
        "model_version": "model-a",
        "result": {
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {},
        },
        "operator_label": 0,
        "defect_type": "",
        "note": "",
    }
    store.save_feedback(**kwargs)
    with pytest.raises(ValueError, match="duplicate feedback content"):
        store.save_feedback(**kwargs, dataset_role="validation")

    other = dict(kwargs)
    other["model_version"] = "model-b"
    store.save_feedback(**other)
    assert len(store.records("Class1")) == 2
    assert len(list(store.images.iterdir())) == 2


def test_duplicate_feedback_exposes_exact_existing_and_proposed_records(
    tmp_path,
) -> None:
    store = FeedbackStore(tmp_path)
    kwargs = {
        "image_bytes": b"same-content",
        "original_name": "sample.png",
        "category": "Class1",
        "model_version": "model-a",
        "result": {
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {},
        },
        "operator_label": 0,
        "defect_type": "",
        "note": "",
    }
    existing = store.save_feedback(**kwargs)

    with pytest.raises(DuplicateFeedbackError) as captured:
        store.save_feedback(**kwargs)

    error = captured.value
    assert error.existing_record["feedback_id"] == existing["feedback_id"]
    assert store.feedback_changes(
        error.existing_record, error.proposed_record
    ) == {}


def test_replace_feedback_is_atomic_keeps_id_and_survives_restart(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    kwargs = {
        "image_bytes": b"same-content",
        "original_name": "sample.png",
        "category": "Class1",
        "model_version": "model-a",
        "result": {
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {},
        },
        "operator_label": 0,
        "defect_type": "",
        "note": "old note",
    }
    existing = store.save_feedback(**kwargs)
    with pytest.raises(DuplicateFeedbackError) as captured:
        store.save_feedback(
            **{**kwargs, "operator_label": 1, "defect_type": "appearance", "note": "new note"}
        )
    duplicate = captured.value
    changes = store.feedback_changes(
        duplicate.existing_record, duplicate.proposed_record
    )
    assert {"operator_label", "feedback_type", "defect_type", "note"} <= set(
        changes
    )

    replaced = store.replace_feedback(
        str(existing["feedback_id"]),
        duplicate.proposed_record,
        expected_record_sha256=store.feedback_record_sha256(
            duplicate.existing_record
        ),
    )

    assert replaced["feedback_id"] == existing["feedback_id"]
    assert replaced["image"] == existing["image"]
    assert replaced["created_at"] == existing["created_at"]
    assert replaced["operator_label"] == 1
    assert replaced["note"] == "new note"
    assert replaced["replacement_count"] == 1
    assert replaced["previous_record_sha256"] == existing["record_sha256"]
    assert len(store.records()) == 1
    restarted = FeedbackStore(tmp_path).records()[0]
    assert restarted == replaced
    assert len(list(store.images.iterdir())) == 1


def test_unprotected_role_replace_keeps_train_validation_disjoint(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    kwargs = {
        "image_bytes": b"same-content",
        "original_name": "sample.png",
        "category": "Class1",
        "model_version": "model-a",
        "result": {
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {},
        },
        "operator_label": 0,
        "defect_type": "",
        "note": "",
    }
    existing = store.save_feedback(**kwargs)
    with pytest.raises(DuplicateFeedbackError) as captured:
        store.save_feedback(**kwargs, dataset_role="validation")

    store.replace_feedback(
        str(existing["feedback_id"]), captured.value.proposed_record
    )

    assert store.records("Class1", "training", "model-a") == []
    validation = store.records("Class1", "validation", "model-a")
    assert [row["feedback_id"] for row in validation] == [existing["feedback_id"]]


def test_candidate_evidence_blocks_feedback_replacement(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    existing = _save_sample(store, category="Class1", name="protected")
    (store.candidates / "candidate-test.yaml").write_text(
        yaml.safe_dump(
            {
                "feedback_snapshot": {
                    "training": [
                        {
                            "feedback_id": existing["feedback_id"],
                            "record_sha256": existing["record_sha256"],
                        }
                    ],
                    "validation": [],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DuplicateFeedbackError) as captured:
        store.save_feedback(
            image_bytes=b"image-Class1-protected",
            original_name="protected.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": False,
                "fusion_score": 0.1,
                "fusion_threshold": 0.5,
                "branch_margins": {
                    "general": 0.0,
                    "local": 0.0,
                    "known": 0.0,
                },
            },
            operator_label=0,
            defect_type="",
            note="changed",
        )

    with pytest.raises(PermissionError, match="candidate, active, or rollback"):
        store.replace_feedback(
            str(existing["feedback_id"]), captured.value.proposed_record
        )
    assert store.records()[0] == existing


def test_replace_feedback_write_failure_preserves_original(tmp_path, monkeypatch) -> None:
    store = FeedbackStore(tmp_path)
    existing = _save_sample(store, category="Class1", name="atomic-replace")
    original_records = store.records_path.read_bytes()
    with pytest.raises(DuplicateFeedbackError) as captured:
        store.save_feedback(
            image_bytes=b"image-Class1-atomic-replace",
            original_name="atomic-replace.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": False,
                "fusion_score": 0.1,
                "fusion_threshold": 0.5,
                "branch_margins": {
                    "general": 0.0,
                    "local": 0.0,
                    "known": 0.0,
                },
            },
            operator_label=0,
            defect_type="",
            note="changed",
        )

    def fail_replace(_source, _destination) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        store.replace_feedback(
            str(existing["feedback_id"]), captured.value.proposed_record
        )

    assert store.records_path.read_bytes() == original_records
    assert len(list(store.images.iterdir())) == 1
    assert list(tmp_path.glob(".records.jsonl.*.tmp")) == []


def test_threshold_search_rejects_legacy_train_validation_hash_overlap(
    tmp_path,
) -> None:
    store = FeedbackStore(tmp_path)
    for index, (score, label) in enumerate(
        zip([0.1, 0.2, 0.3, 0.8, 0.9, 1.0], [0, 0, 0, 1, 1, 1], strict=True)
    ):
        store.save_feedback(
            image_bytes=f"training-{index}".encode(),
            original_name=f"training-{index}.png",
            category="Class1",
            model_version="test-model",
            result={
                "decision": score >= 0.5,
                "fusion_score": score,
                "fusion_threshold": 0.5,
                "branch_margins": {},
            },
            operator_label=label,
            defect_type="",
            note="",
        )
    validation = store.save_feedback(
        image_bytes=b"validation",
        original_name="validation.png",
        category="Class1",
        model_version="test-model",
        result={
            "decision": False,
            "fusion_score": 0.2,
            "fusion_threshold": 0.5,
            "branch_margins": {},
        },
        operator_label=0,
        defect_type="",
        note="",
        dataset_role="validation",
    )
    rows = [json.loads(line) for line in store.records_path.read_text().splitlines()]
    rows[-1]["image_sha256"] = rows[0]["image_sha256"]
    assert rows[-1]["feedback_id"] == validation["feedback_id"]
    store.records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(PermissionError, match="TRAIN_VALIDATION_LEAKAGE"):
        store.propose_threshold("Class1", "test-model", base_threshold=0.5)


def test_legacy_unscoped_registry_is_not_applied(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    store.registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "active": {"Class1": {"threshold": 0.25}},
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    assert store.active_threshold("Class1", "test-model") is None
    registry = store.registry()
    assert registry["active"] == {}
    assert registry["legacy_unscoped"]["Class1"]["threshold"] == 0.25


def test_feature_bank_rejects_unvalidated_activation(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    with pytest.raises(ValueError, match="independent validation"):
        store.save_feature_bank(
            "Class1",
            "test-model",
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            ["missing-feedback"],
            validation={"passed": False},
        )


def test_delete_feedback_removes_record_and_unreferenced_image(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    deleted = _save_sample(store, category="Class1", name="delete")
    retained = _save_sample(store, category="Class2", name="retain")
    deleted_image = store.root / str(deleted["image"])
    retained_image = store.root / str(retained["image"])

    summary = store.delete_feedback(
        [str(deleted["feedback_id"])], category="Class1"
    )

    assert summary["deleted_records"] == 1
    assert summary["deleted_images"] == 1
    assert summary["missing_ids"] == []
    assert not deleted_image.exists()
    assert retained_image.is_file()
    assert [row["feedback_id"] for row in store.records()] == [
        retained["feedback_id"]
    ]


def test_delete_feedback_category_guard_has_no_side_effects(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    record = _save_sample(store, category="Class1", name="guarded")
    image = store.root / str(record["image"])
    original_records = store.records_path.read_bytes()

    summary = store.delete_feedback(
        [str(record["feedback_id"])], category="Class2"
    )

    assert summary["deleted_records"] == 0
    assert summary["missing_ids"] == [record["feedback_id"]]
    assert store.records_path.read_bytes() == original_records
    assert image.is_file()


def test_delete_feedback_clears_only_requested_category(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    training = _save_sample(store, category="Class1", name="training")
    validation = _save_sample(
        store, category="Class1", name="validation", dataset_role="validation"
    )
    retained = _save_sample(store, category="Class2", name="retained")

    summary = store.delete_feedback(
        [str(training["feedback_id"]), str(validation["feedback_id"])],
        category="Class1",
    )

    assert summary["deleted_records"] == 2
    assert store.records("Class1") == []
    assert [row["feedback_id"] for row in store.records("Class2")] == [
        retained["feedback_id"]
    ]


def test_delete_feedback_keeps_shared_image_until_last_reference(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    first = _save_sample(store, category="Class1", name="first")
    second = _save_sample(store, category="Class1", name="second")
    shared_image = store.root / str(first["image"])
    rows = [json.loads(line) for line in store.records_path.read_text().splitlines()]
    rows[1]["image"] = rows[0]["image"]
    store.records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    first_summary = store.delete_feedback(
        [str(first["feedback_id"])], category="Class1"
    )
    assert first_summary["deleted_images"] == 0
    assert shared_image.is_file()

    second_summary = store.delete_feedback(
        [str(second["feedback_id"])], category="Class1"
    )

    assert second_summary["deleted_images"] == 1
    assert not shared_image.exists()


def test_delete_feedback_never_removes_external_image(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback")
    record = _save_sample(store, category="Class1", name="external")
    external = tmp_path / "outside.png"
    external.write_bytes(b"outside")
    row = json.loads(store.records_path.read_text(encoding="utf-8"))
    row["image"] = str(external)
    store.records_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = store.delete_feedback(
        [str(record["feedback_id"])], category="Class1"
    )

    assert summary["deleted_records"] == 1
    assert summary["deleted_images"] == 0
    assert external.is_file()


def test_delete_feedback_rejects_corrupt_records_without_changes(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    record = _save_sample(store, category="Class1", name="valid")
    image = store.root / str(record["image"])
    store.records_path.write_text(
        store.records_path.read_text(encoding="utf-8") + "{broken\n",
        encoding="utf-8",
    )
    original_records = store.records_path.read_bytes()

    with pytest.raises(json.JSONDecodeError):
        store.delete_feedback([str(record["feedback_id"])], category="Class1")

    assert store.records_path.read_bytes() == original_records
    assert image.is_file()


def test_delete_feedback_replace_failure_is_atomic(tmp_path, monkeypatch) -> None:
    store = FeedbackStore(tmp_path)
    record = _save_sample(store, category="Class1", name="atomic")
    image = store.root / str(record["image"])
    original_records = store.records_path.read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        store.delete_feedback([str(record["feedback_id"])], category="Class1")

    assert store.records_path.read_bytes() == original_records
    assert image.is_file()
    assert list(tmp_path.glob(".records.jsonl.*.tmp")) == []


def test_delete_feedback_empty_request_has_no_side_effects(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    assert not store.records_path.exists()

    summary = store.delete_feedback([])

    assert summary["deleted_records"] == 0
    assert summary["missing_ids"] == []
    assert not store.records_path.exists()


def test_delete_and_save_from_separate_stores_do_not_lose_records(
    tmp_path, monkeypatch
) -> None:
    deleting_store = FeedbackStore(tmp_path)
    saving_store = FeedbackStore(tmp_path)
    deleted = _save_sample(deleting_store, category="Class1", name="deleted")
    delete_has_read = threading.Event()
    save_has_started = threading.Event()
    errors: list[BaseException] = []
    original_read_text = type(deleting_store.records_path).read_text

    def delayed_read_text(path, *args, **kwargs):
        content = original_read_text(path, *args, **kwargs)
        if (
            path == deleting_store.records_path
            and threading.current_thread().name == "delete-feedback"
        ):
            delete_has_read.set()
            assert save_has_started.wait(2)
            time.sleep(0.1)
        return content

    monkeypatch.setattr(type(deleting_store.records_path), "read_text", delayed_read_text)

    def delete_record() -> None:
        try:
            deleting_store.delete_feedback(
                [str(deleted["feedback_id"])], category="Class1"
            )
        except BaseException as error:
            errors.append(error)

    def save_record() -> None:
        try:
            save_has_started.set()
            _save_sample(saving_store, category="Class1", name="concurrent")
        except BaseException as error:
            errors.append(error)

    delete_thread = threading.Thread(target=delete_record, name="delete-feedback")
    delete_thread.start()
    assert delete_has_read.wait(2)
    save_thread = threading.Thread(target=save_record, name="save-feedback")
    save_thread.start()
    delete_thread.join(5)
    save_thread.join(5)

    assert not delete_thread.is_alive()
    assert not save_thread.is_alive()
    assert errors == []
    assert [row["original_name"] for row in deleting_store.records("Class1")] == [
        "concurrent.png"
    ]


def test_plan_and_purge_product_evidence_is_exact_scope_and_complete(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path)
    target_bundle = "a" * 64
    retained_bundle = "b" * 64
    target = _save_scoped_sample(
        store,
        category="product-a",
        run_id="run-a",
        bundle_hash=target_bundle,
        name="target",
    )
    retained_other_scope = _save_scoped_sample(
        store,
        category="product-b",
        run_id="run-b",
        bundle_hash=retained_bundle,
        name="retained-scope",
    )
    retained_other_bundle = _save_scoped_sample(
        store,
        category="product-a",
        run_id="run-a",
        bundle_hash=retained_bundle,
        name="retained-bundle",
    )

    target_threshold = _write_threshold_receipt(
        store,
        candidate_id="threshold-target",
        category="product-a",
        model_version="run-a",
    )
    retained_threshold = _write_threshold_receipt(
        store,
        candidate_id="threshold-retained",
        category="product-b",
        model_version="run-b",
    )
    target_revision = _write_revision_receipt(
        store,
        revision_id="revision-target",
        category="product-a",
        run_id="run-a",
        bundle_hash=target_bundle,
        feedback_id=str(target["feedback_id"]),
    )
    retained_revision = _write_revision_receipt(
        store,
        revision_id="revision-retained",
        category="product-b",
        run_id="run-b",
        bundle_hash=retained_bundle,
    )
    retained_bundle_revision = _write_revision_receipt(
        store,
        revision_id="revision-retained-bundle",
        category="product-a",
        run_id="run-a",
        bundle_hash=retained_bundle,
    )

    target_feature = store.feature_banks / "features-target.npz"
    retained_feature = store.feature_banks / "features-retained.npz"
    target_feature.write_bytes(b"target-feature")
    retained_feature.write_bytes(b"retained-feature")
    target_feature_entry = {
        "version": "features-target",
        "category": "product-a",
        "model_version": "run-a",
        "path": target_feature.relative_to(store.root).as_posix(),
        "file_sha256": hashlib.sha256(target_feature.read_bytes()).hexdigest(),
    }
    retained_feature_entry = {
        "version": "features-retained",
        "category": "product-b",
        "model_version": "run-b",
        "path": retained_feature.relative_to(store.root).as_posix(),
        "file_sha256": hashlib.sha256(retained_feature.read_bytes()).hexdigest(),
    }
    store.feature_registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "active": {
                    "target": target_feature_entry,
                    "retained": retained_feature_entry,
                },
                "history": [target_feature_entry, retained_feature_entry],
                "legacy_unscoped": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target_threshold_entry = {
        "candidate_id": "threshold-target",
        "category": "product-a",
        "model_version": "run-a",
    }
    retained_threshold_entry = {
        "candidate_id": "threshold-retained",
        "category": "product-b",
        "model_version": "run-b",
    }
    store.registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "active": {
                    "target": target_threshold_entry,
                    "retained": retained_threshold_entry,
                },
                "history": [target_threshold_entry, retained_threshold_entry],
                "legacy_unscoped": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target_revision_entry = {
        "revision_id": "revision-target",
        "category": "product-a",
        "run_id": "run-a",
        "bundle_hash": target_bundle,
    }
    retained_revision_entry = {
        "revision_id": "revision-retained",
        "category": "product-b",
        "run_id": "run-b",
        "bundle_hash": retained_bundle,
    }
    retained_bundle_revision_entry = {
        "revision_id": "revision-retained-bundle",
        "category": "product-a",
        "run_id": "run-a",
        "bundle_hash": retained_bundle,
    }
    store.revision_registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "active": {
                    "target": target_revision_entry,
                    "retained": retained_revision_entry,
                    "retained-bundle": retained_bundle_revision_entry,
                },
                "history": [
                    {"action": "activate", **target_revision_entry},
                    {"action": "activate", **retained_revision_entry},
                    {"action": "activate", **retained_bundle_revision_entry},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target_query = _write_query(
        store,
        query_id="query-target",
        category="product-a",
        run_id="run-a",
        bundle_hash=target_bundle,
    )
    retained_query = _write_query(
        store,
        query_id="query-retained",
        category="product-b",
        run_id="run-b",
        bundle_hash=retained_bundle,
    )
    retained_bundle_query = _write_query(
        store,
        query_id="query-retained-bundle",
        category="product-a",
        run_id="run-a",
        bundle_hash=retained_bundle,
    )

    with pytest.raises(PermissionError, match="cannot delete feedback evidence"):
        store.delete_feedback([str(target["feedback_id"])], category="product-a")

    before_plan = _tree_snapshot(store.root)
    plan = store.plan_product_evidence_purge(
        "product-a", "run-a", target_bundle
    )
    repeated_plan = store.plan_product_evidence_purge(
        "product-a", "run-a", target_bundle
    )

    assert _tree_snapshot(store.root) == before_plan
    assert repeated_plan["plan_sha256"] == plan["plan_sha256"]
    assert plan["counts"] == {
        "records": 1,
        "images": 1,
        "candidates": 2,
        "revisions": 1,
        "feature_banks": 1,
        "queries": 1,
        "threshold_registry_entries": 2,
        "feature_registry_entries": 2,
        "revision_registry_entries": 2,
    }

    result = store.purge_product_evidence(
        "product-a",
        "run-a",
        target_bundle,
        expected_plan_sha256=str(plan["plan_sha256"]),
    )

    assert result["counts"] == plan["counts"]
    assert result["pending_cleanup_path"] is None
    assert result["failed_cleanup"] == []
    assert list(store.root.glob(".purge-*")) == []
    assert not (store.root / str(target["image"])).exists()
    assert not target_threshold.exists()
    assert not target_revision.exists()
    assert not (store.revisions / "revision-target").exists()
    assert not target_feature.exists()
    assert not target_query.exists()

    assert (store.root / str(retained_other_scope["image"])).is_file()
    assert (store.root / str(retained_other_bundle["image"])).is_file()
    assert retained_threshold.is_file()
    assert retained_revision.is_file()
    assert retained_bundle_revision.is_file()
    assert (store.revisions / "revision-retained").is_dir()
    assert (store.revisions / "revision-retained-bundle").is_dir()
    assert retained_feature.is_file()
    assert retained_query.is_file()
    assert retained_bundle_query.is_file()
    remaining = {
        (str(row["category"]), str(row["model_version"]), str(row["bundle_hash"]))
        for row in store.records()
    }
    assert remaining == {
        ("product-b", "run-b", retained_bundle),
        ("product-a", "run-a", retained_bundle),
    }
    assert set(store.registry()["active"]) == {"retained"}
    assert set(store.feature_registry()["active"]) == {"retained"}
    assert set(store._revision_registry()["active"]) == {
        "retained",
        "retained-bundle",
    }


def test_purge_product_evidence_rejects_stale_plan_without_changes(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path)
    bundle_hash = "c" * 64
    _save_scoped_sample(
        store,
        category="product-a",
        run_id="run-a",
        bundle_hash=bundle_hash,
        name="first",
    )
    plan = store.plan_product_evidence_purge(
        "product-a", "run-a", bundle_hash
    )
    _save_scoped_sample(
        FeedbackStore(tmp_path),
        category="product-a",
        run_id="run-a",
        bundle_hash=bundle_hash,
        name="concurrent",
    )
    after_change = _tree_snapshot(tmp_path)

    with pytest.raises(RuntimeError, match="plan changed"):
        store.purge_product_evidence(
            "product-a",
            "run-a",
            bundle_hash,
            expected_plan_sha256=str(plan["plan_sha256"]),
        )

    assert _tree_snapshot(tmp_path) == after_change
    assert len(store.records("product-a", model_version="run-a")) == 2


def test_product_evidence_purge_plan_rejects_corrupt_registry_without_changes(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path)
    bundle_hash = "d" * 64
    _save_scoped_sample(
        store,
        category="product-a",
        run_id="run-a",
        bundle_hash=bundle_hash,
        name="target",
    )
    store.registry_path.write_text("active: [broken\n", encoding="utf-8")
    before = _tree_snapshot(tmp_path)

    with pytest.raises(yaml.YAMLError):
        store.plan_product_evidence_purge("product-a", "run-a", bundle_hash)

    assert _tree_snapshot(tmp_path) == before


def test_product_evidence_purge_plan_rejects_escaping_revision_artifact(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must-survive")
    revision_id = "revision-escape"
    (store.revisions / revision_id).mkdir()
    _write_signed_yaml(
        store.candidates / f"{revision_id}.yaml",
        {
            "schema_version": 2,
            "revision_id": revision_id,
            "category": "product-a",
            "run_id": "run-a",
            "bundle_hash": "e" * 64,
            "artifacts": {"head": {"path": "../outside.bin"}},
        },
        hash_field="payload_sha256",
    )
    store.records()  # Materialise the normal inter-process lock before snapshotting.
    before = _tree_snapshot(store.root)

    with pytest.raises(PermissionError, match="relative and contained"):
        store.plan_product_evidence_purge("product-a", "run-a", "e" * 64)

    assert outside.read_bytes() == b"must-survive"
    assert _tree_snapshot(store.root) == before


def test_product_evidence_purge_removes_legacy_registry_scope(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path)
    feature_path = store.feature_banks / "legacy-feature.npz"
    feature_path.write_bytes(b"legacy-feature")
    store.registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "active": {"product-a": {"threshold": 0.25}},
                "history": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    store.feature_registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "active": {
                    "product-a": {
                        "version": "legacy-feature",
                        "path": feature_path.relative_to(store.root).as_posix(),
                        "file_sha256": hashlib.sha256(
                            feature_path.read_bytes()
                        ).hexdigest(),
                    }
                },
                "history": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = store.plan_product_evidence_purge("product-a", "run-a")

    assert plan["counts"]["threshold_registry_entries"] == 1
    assert plan["counts"]["feature_registry_entries"] == 1
    assert plan["counts"]["feature_banks"] == 1
    store.purge_product_evidence(
        "product-a",
        "run-a",
        expected_plan_sha256=str(plan["plan_sha256"]),
    )
    assert not feature_path.exists()
    assert "product-a" not in store.registry()["legacy_unscoped"]
    assert "product-a" not in store.feature_registry()["legacy_unscoped"]


def test_product_evidence_purge_metadata_failure_rolls_back_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FeedbackStore(tmp_path)
    bundle_hash = "f" * 64
    target = _save_scoped_sample(
        store,
        category="product-a",
        run_id="run-a",
        bundle_hash=bundle_hash,
        name="target",
    )
    candidate = _write_threshold_receipt(
        store,
        candidate_id="threshold-target",
        category="product-a",
        model_version="run-a",
    )
    entry = {
        "candidate_id": "threshold-target",
        "category": "product-a",
        "model_version": "run-a",
    }
    store.registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "active": {"target": entry},
                "history": [entry],
                "legacy_unscoped": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    plan = store.plan_product_evidence_purge(
        "product-a", "run-a", bundle_hash
    )
    before = _tree_snapshot(tmp_path)
    original_write = feedback_service._write_bytes_atomic
    failed = False

    def fail_registry_once(path: Path, payload: bytes) -> None:
        nonlocal failed
        if path == store.registry_path and not failed:
            failed = True
            raise OSError("simulated metadata failure")
        original_write(path, payload)

    monkeypatch.setattr(feedback_service, "_write_bytes_atomic", fail_registry_once)
    with pytest.raises(OSError, match="simulated metadata failure"):
        store.purge_product_evidence(
            "product-a",
            "run-a",
            bundle_hash,
            expected_plan_sha256=str(plan["plan_sha256"]),
        )

    assert failed is True
    assert _tree_snapshot(tmp_path) == before
    assert candidate.is_file()
    assert (store.root / str(target["image"])).is_file()
    assert list(store.root.glob(".purge-*")) == []
