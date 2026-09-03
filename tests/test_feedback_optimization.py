from __future__ import annotations

import io
import json
import threading
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import aoi.feedback.runtime as feedback_runtime
from aoi.competition.models.known_defect_heads.dinov2_tiled_mlp import (
    DinoTiledMLPKnownHead,
)
from aoi.feedback import FeedbackStore, apply_active_feedback
from aoi.feedback.optimization import (
    _defect_feature_crops,
    _defect_head_features,
    _read_feedback_images,
    propose_adaptive_revision,
)
from aoi.feedback.service import _canonical_sha256


CATEGORY = "Class1"
RUN_ID = "feedback-run"
BUNDLE_HASH = "a" * 64
OTHER_BUNDLE_HASH = "b" * 64


def _head() -> DinoTiledMLPKnownHead:
    """Return a fitted-shaped tiled MLP suitable for feedback-only tuning."""
    head = DinoTiledMLPKnownHead(
        tile_count=9,
        hidden_dimension=32,
        epochs=1,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(20260830)
    head.feature_mean = torch.zeros(4)
    head.feature_scale = torch.ones(4)
    head.weight1 = torch.randn(4, 32, generator=generator) * 0.05
    head.bias1 = torch.full((32,), 0.05)
    head.weight2 = torch.linspace(-0.03, 0.03, 32)
    head.bias2 = torch.tensor(0.0)
    return head


class _Dino:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.defect_global_bank = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    def _features(self, images: list[np.ndarray]) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for image in images:
            value = float(np.asarray(image).mean())
            if value >= 128.0:
                vector = torch.tensor([0.0, 1.0, 0.1, 0.0])
            else:
                vector = torch.tensor([1.0, 0.0, 0.0, 0.1])
            features.append(vector.repeat(4, 1))
        return torch.stack(features)


class _Runtime:
    def __init__(self) -> None:
        self.bundle = {"run_id": RUN_ID, "bundle_hash": BUNDLE_HASH}
        self.bundle_path = Path("outputs/feedback-run/model_bundle.yaml")
        self.device = "cpu"
        self.head = _head()
        self.dino = _Dino()
        self._feedback_lock = threading.RLock()

    def predict_batch(self, images: list[np.ndarray], ids: list[str]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for image, sample_id in zip(images, ids, strict=True):
            score = 0.9 if float(np.asarray(image).mean()) >= 128.0 else 0.1
            rows.append(
                {
                    "sample_id": sample_id,
                    "fusion_score": score,
                    "fusion_threshold": 0.5,
                    "decision": score >= 0.5,
                    "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
                }
            )
        return rows


class _RelativeImprovementRuntime(_Runtime):
    """Expose a candidate-only predictor that improves F1 below the recall target."""

    @staticmethod
    def _row(sample_id: str, score: float) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "fusion_score": score,
            "fusion_threshold": 0.5,
            "decision": score >= 0.5,
            "branch_margins": {
                "general": score,
                "local": 0.0,
                "known": 0.0,
            },
        }

    def predict_batch(
        self,
        images: list[np.ndarray],
        ids: list[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for image, sample_id in zip(images, ids, strict=True):
            intensity = int(round(float(np.asarray(image).mean())))
            if intensity in {50, 60}:
                score = 0.8
            elif intensity == 200:
                score = 0.8
            elif intensity == 240:
                score = 0.2
            else:
                score = 0.9 if intensity >= 128 else 0.1
            rows.append(self._row(sample_id, score))
        return rows

    def predict_with_feedback(
        self,
        images: list[np.ndarray],
        ids: list[str],
        *,
        head: object,
        feature_bank: object,
    ) -> list[dict[str, object]]:
        assert head is not None
        assert feature_bank is not None
        rows: list[dict[str, object]] = []
        for image, sample_id in zip(images, ids, strict=True):
            intensity = int(round(float(np.asarray(image).mean())))
            if intensity == 200:
                score = 0.95
            elif intensity == 240:
                score = 0.30
            elif intensity == 60:
                score = 0.20
            else:
                score = 0.90 if intensity >= 128 else 0.10
            rows.append(self._row(sample_id, score))
        return rows


def _png(value: int) -> bytes:
    image = Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_records(store: FeedbackStore) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    samples = [
        ("training", 20, 0),
        ("training", 30, 0),
        ("training", 40, 0),
        ("training", 210, 1),
        ("training", 220, 1),
        ("training", 230, 1),
        ("validation", 50, 0),
        ("validation", 60, 0),
        ("validation", 200, 1),
        ("validation", 240, 1),
    ]
    for index, (role, intensity, label) in enumerate(samples):
        score = 0.9 if label else 0.1
        rows.append(
            store.save_feedback(
                image_bytes=_png(intensity),
                original_name=f"{role}-{index}.png",
                category=CATEGORY,
                model_version=RUN_ID,
                result={
                    "decision": bool(label),
                    "fusion_score": score,
                    "fusion_threshold": 0.5,
                    "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
                },
                operator_label=label,
                defect_type="appearance" if label else "",
                note="feedback fixture",
                dataset_role=role,
                defect_region=[4, 4, 24, 24] if label else None,
                image_size=(32, 32),
                run_id=RUN_ID,
                bundle_hash=BUNDLE_HASH,
            )
        )
    return rows


def _proposal(store: FeedbackStore, runtime: _Runtime) -> dict[str, object]:
    return propose_adaptive_revision(
        runtime,
        store,
        CATEGORY,
        epochs=3,
        learning_rate=0.02,
        seed=29,
        minimum_training_samples=6,
        minimum_validation_samples=4,
        minimum_recall=0.9,
    )


def _fixture(tmp_path: Path) -> tuple[FeedbackStore, _Runtime, list[dict[str, object]]]:
    store = FeedbackStore(tmp_path / "feedback")
    records = _save_records(store)
    return store, _Runtime(), records


def test_relative_validation_improvement_below_target_can_save_and_activate(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback")
    _save_records(store)
    runtime = _RelativeImprovementRuntime()

    candidate = propose_adaptive_revision(
        runtime,
        store,
        CATEGORY,
        epochs=3,
        learning_rate=0.02,
        seed=29,
        minimum_training_samples=6,
        minimum_validation_samples=4,
        minimum_recall=0.90,
    )

    validation = candidate["validation"]
    parent_metrics = candidate["parent_validation_metrics"]
    candidate_metrics = candidate["validation_metrics"]
    assert validation["acceptance_mode"] == "relative_improvement"
    assert validation["absolute_gate_passed"] is False
    assert validation["relative_improvement_passed"] is True
    assert candidate_metrics["recall"] == pytest.approx(0.50)
    assert candidate_metrics["recall"] < validation["minimum_recall"]
    assert candidate_metrics["recall"] >= parent_metrics["recall"]
    assert candidate_metrics["f1"] > parent_metrics["f1"]
    assert validation["recall_delta"] == pytest.approx(0.0)
    assert validation["f1_delta"] == pytest.approx(
        candidate_metrics["f1"] - parent_metrics["f1"]
    )
    assert (store.root / str(candidate["receipt_path"])).is_file()

    activated = store.activate_revision(candidate)

    assert activated["revision_id"] == candidate["revision_id"]
    active = store.active_revision(CATEGORY, RUN_ID, BUNDLE_HASH, verify=True)
    assert active is not None
    assert active["revision_id"] == candidate["revision_id"]


def test_feedback_image_decoder_supports_chinese_windows_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FeedbackStore(tmp_path / "中文反馈")
    record = store.save_feedback(
        image_bytes=_png(137),
        original_name="缺陷样本.png",
        category=CATEGORY,
        model_version=RUN_ID,
        result={
            "decision": True,
            "fusion_score": 0.9,
            "fusion_threshold": 0.5,
            "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
        },
        operator_label=1,
        defect_type="appearance",
        note="unicode path regression",
        dataset_role="training",
        defect_region=[4, 4, 24, 24],
        image_size=(32, 32),
        run_id=RUN_ID,
        bundle_hash=BUNDLE_HASH,
    )
    cv2 = pytest.importorskip("cv2")

    def reject_path_based_decode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("feedback decoding must not use cv2.imread(path)")

    monkeypatch.setattr(cv2, "imread", reject_path_based_decode)
    images = _read_feedback_images(store, [record])

    assert len(images) == 1
    assert images[0].shape == (32, 32, 3)
    assert images[0].dtype == np.uint8
    assert np.all(images[0] == 137)


def _rewrite_receipt(
    store: FeedbackStore,
    candidate: dict[str, object],
    mutate,
) -> None:
    receipt_path = store.root / str(candidate["receipt_path"])
    payload = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    mutate(payload)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = _canonical_sha256(payload)
    receipt_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_complete_revision_proposal_activation_restart_rollback_and_deletion_guard(
    tmp_path: Path,
) -> None:
    store, runtime, records = _fixture(tmp_path)
    base_fingerprint = runtime.head.parameter_fingerprint()
    candidate_one = _proposal(store, runtime)

    assert candidate_one["updated_parameter_count"] == 33
    assert candidate_one["feature_count"] == 3
    assert candidate_one["validation_metrics"]["f1"] == 1.0
    assert not store.revision_registry_path.exists()

    active_one = store.activate_revision(candidate_one)
    state_one = apply_active_feedback(runtime, store, CATEGORY)
    assert state_one.revision_id == candidate_one["revision_id"]
    assert state_one.head_parameter_count == 33
    assert state_one.threshold_source == "active_operator_feedback"
    assert runtime.head.frozen_parameter_fingerprint() == {
        name: base_fingerprint[name]
        for name in ("feature_mean", "feature_scale", "weight1", "bias1")
    }
    assert runtime.head.parameter_fingerprint()["overall_sha256"] == candidate_one[
        "head_parameter_fingerprint"
    ]["candidate"]["overall_sha256"]

    restarted = _Runtime()
    restarted_state = apply_active_feedback(restarted, FeedbackStore(store.root), CATEGORY)
    assert restarted_state.revision_id == state_one.revision_id
    assert restarted.head.parameter_fingerprint() == runtime.head.parameter_fingerprint()

    # New evidence is expected after activation. It must not invalidate the
    # immutable rows bound to the active revision.
    store.save_feedback(
        image_bytes=_png(70),
        original_name="new-normal.png",
        category=CATEGORY,
        model_version=RUN_ID,
        result={
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
        },
        operator_label=0,
        defect_type="",
        note="post-activation evidence",
        dataset_role="training",
        image_size=(32, 32),
        run_id=RUN_ID,
        bundle_hash=BUNDLE_HASH,
    )
    still_active = apply_active_feedback(restarted, FeedbackStore(store.root), CATEGORY)
    assert still_active.revision_id == state_one.revision_id

    candidate_two = _proposal(store, runtime)
    assert candidate_two["parent_revision_id"] == active_one["revision_id"]
    store.activate_revision(candidate_two)
    state_two = apply_active_feedback(runtime, store, CATEGORY)
    assert state_two.revision_id == candidate_two["revision_id"]

    restored = store.rollback_revision(CATEGORY, RUN_ID, BUNDLE_HASH)
    assert restored is not None
    assert restored["revision_id"] == candidate_one["revision_id"]
    rolled_back = apply_active_feedback(runtime, store, CATEGORY)
    assert rolled_back.revision_id == candidate_one["revision_id"]

    with pytest.raises(PermissionError, match="cannot delete feedback evidence"):
        store.delete_feedback([str(records[0]["feedback_id"])], category=CATEGORY)


def test_activation_restores_all_runtime_state_when_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, runtime, _records = _fixture(tmp_path)
    candidate = _proposal(store, runtime)
    store.activate_revision(candidate)
    head_before = runtime.head.parameter_fingerprint()
    bank_before = runtime.dino.defect_global_bank.detach().clone()

    original = feedback_runtime.set_runtime_feature_bank

    def fail_feature_commit(target, embeddings, *, feature_version):
        if embeddings is not None:
            raise RuntimeError("simulated feature-bank commit failure")
        return original(target, embeddings, feature_version=feature_version)

    monkeypatch.setattr(feedback_runtime, "set_runtime_feature_bank", fail_feature_commit)
    with pytest.raises(RuntimeError, match="simulated feature-bank commit failure"):
        apply_active_feedback(runtime, store, CATEGORY)

    assert runtime.head.parameter_fingerprint() == head_before
    assert torch.equal(runtime.dino.defect_global_bank, bank_before)
    assert runtime._feedback_threshold is None
    assert runtime._feedback_head_version is None


def test_revision_rejects_tampered_artifact_scope_and_unsafe_artifact_path(tmp_path: Path) -> None:
    store, runtime, _records = _fixture(tmp_path)
    candidate = _proposal(store, runtime)

    with pytest.raises(ValueError, match="bundle hash does not match"):
        store.activate_revision(candidate, bundle_hash=OTHER_BUNDLE_HASH)
    with pytest.raises(ValueError, match="scope does not match"):
        store.activate_revision(candidate, run_id="other-run")

    head_path = store.root / str(candidate["artifacts"]["head"]["path"])
    head_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="head artifact SHA-256 mismatch"):
        store.activate_revision(candidate)

    store, runtime, _records = _fixture(tmp_path / "unsafe")
    candidate = _proposal(store, runtime)
    _rewrite_receipt(
        store,
        candidate,
        lambda payload: payload["artifacts"]["head"].update({"path": "../../escape.pt"}),
    )
    with pytest.raises(PermissionError, match="must be relative and contained"):
        store.activate_revision(str(candidate["revision_id"]))


def test_revision_rejects_forged_validation_acceptance_mode(tmp_path: Path) -> None:
    store, runtime, _records = _fixture(tmp_path)
    candidate = _proposal(store, runtime)
    assert candidate["validation"]["acceptance_mode"] == "absolute_gate"

    _rewrite_receipt(
        store,
        candidate,
        lambda payload: payload["validation"].update(
            {"acceptance_mode": "relative_improvement"}
        ),
    )

    with pytest.raises(
        ValueError,
        match="validation acceptance mode does not match metrics",
    ):
        store.activate_revision(str(candidate["revision_id"]))


def test_relative_revision_requires_explicit_auditable_acceptance_fields(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback")
    _save_records(store)
    candidate = _proposal(store, _RelativeImprovementRuntime())
    assert candidate["validation"]["acceptance_mode"] == "relative_improvement"

    def remove_relative_acceptance_fields(payload: dict[str, object]) -> None:
        validation = payload["validation"]
        assert isinstance(validation, dict)
        for key in (
            "acceptance_policy",
            "acceptance_mode",
            "absolute_gate_passed",
            "relative_improvement_passed",
            "recall_delta",
            "f1_delta",
            "reason",
        ):
            validation.pop(key, None)

    _rewrite_receipt(store, candidate, remove_relative_acceptance_fields)

    with pytest.raises(
        ValueError,
        match="relative validation improvement receipt is missing auditable fields",
    ):
        store.activate_revision(str(candidate["revision_id"]))


def test_relative_revision_rejects_forged_acceptance_reason(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback")
    _save_records(store)
    candidate = _proposal(store, _RelativeImprovementRuntime())

    _rewrite_receipt(
        store,
        candidate,
        lambda payload: payload["validation"].update(
            {"reason": "absolute_gate_passed"}
        ),
    )

    with pytest.raises(
        ValueError,
        match="revision validation reason does not match metrics",
    ):
        store.activate_revision(str(candidate["revision_id"]))


def test_legacy_absolute_revision_without_new_acceptance_fields_still_activates(
    tmp_path: Path,
) -> None:
    store, runtime, _records = _fixture(tmp_path)
    candidate = _proposal(store, runtime)
    assert candidate["validation"]["acceptance_mode"] == "absolute_gate"

    def remove_new_acceptance_fields(payload: dict[str, object]) -> None:
        validation = payload["validation"]
        assert isinstance(validation, dict)
        for key in (
            "acceptance_policy",
            "acceptance_mode",
            "absolute_gate_passed",
            "relative_improvement_passed",
            "recall_delta",
            "f1_delta",
        ):
            validation.pop(key, None)
        validation["reason"] = "passed"

    _rewrite_receipt(store, candidate, remove_new_acceptance_fields)

    activated = store.activate_revision(str(candidate["revision_id"]))
    assert activated["revision_id"] == candidate["revision_id"]


def test_revision_rejects_symlinked_artifact_path_when_supported(tmp_path: Path) -> None:
    store, runtime, _records = _fixture(tmp_path)
    candidate = _proposal(store, runtime)
    head_path = store.root / str(candidate["artifacts"]["head"]["path"])
    link_path = head_path.with_name("head-link.pt")
    try:
        link_path.symlink_to(head_path.name)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable on this Windows host: {error}")

    _rewrite_receipt(
        store,
        candidate,
        lambda payload: payload["artifacts"]["head"].update(
            {"path": str(link_path.relative_to(store.root)).replace("\\\\", "/")}
        ),
    )
    with pytest.raises(PermissionError, match="symbolic link"):
        store.activate_revision(str(candidate["revision_id"]))


def test_proposal_rejects_train_validation_content_leakage(tmp_path: Path) -> None:
    store, runtime, _records = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in store.records_path.read_text(encoding="utf-8").splitlines()
    ]
    training_hash = next(row["image_sha256"] for row in rows if row["dataset_role"] == "training")
    validation = next(row for row in rows if row["dataset_role"] == "validation")
    validation["image_sha256"] = training_hash
    store.records_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="TRAIN_VALIDATION_LEAKAGE"):
        _proposal(store, runtime)


def test_invalid_candidate_overlay_is_rejected_without_leaving_artifacts(tmp_path: Path) -> None:
    store, runtime, _records = _fixture(tmp_path)
    valid = _proposal(store, runtime)
    invalid = deepcopy(valid)
    invalid_revision_id = "revision-invalid-overlay"
    invalid["revision_id"] = invalid_revision_id
    invalid["parent_revision_id"] = None
    feature_path = store.root / str(valid["artifacts"]["feature_bank"]["path"])

    with pytest.raises(ValueError, match="head overlay is malformed"):
        store.save_revision_candidate(
            invalid,
            head_artifact=b"not-a-torch-overlay",
            feature_artifact=feature_path,
        )

    assert not (store.revisions / invalid_revision_id).exists()
    assert not (store.candidates / f"{invalid_revision_id}.yaml").exists()


def test_feedback_roi_selects_the_matching_defect_tile_and_feature_crop() -> None:
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[6:9, 6:9] = 255
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[6:9, 6:9] = 1
    seen: list[np.ndarray] = []

    class Dino:
        def _features(self, images):
            seen.extend(np.asarray(value).copy() for value in images)
            return torch.zeros((len(images), 4, 4), dtype=torch.float32)

    class Runtime:
        dino = Dino()

    features, tile_masks = _defect_head_features(Runtime(), [image], [mask])
    assert features is not None and tuple(features.shape) == (1, 4, 4)
    assert len(seen) == 1
    assert np.array_equal(seen[0], image[6:9, 6:9])
    assert len(tile_masks) == 1 and int(tile_masks[0].sum()) > 0

    rows = [{"feedback_id": "roi", "defect_region": [6, 6, 9, 9]}]
    crops = _defect_feature_crops(rows, [image])
    assert len(crops) == 1
    assert np.array_equal(crops[0], image[6:9, 6:9])
