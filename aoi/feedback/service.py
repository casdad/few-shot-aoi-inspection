from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import stat
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import torch
import yaml


class DuplicateFeedbackError(ValueError):
    """A same-image feedback record already exists in the model scope."""

    def __init__(
        self,
        existing_record: Mapping[str, Any],
        proposed_record: Mapping[str, Any],
    ) -> None:
        self.existing_record = dict(existing_record)
        self.proposed_record = dict(proposed_record)
        super().__init__(
            "duplicate feedback content for the same model version: "
            f"{existing_record.get('feedback_id')} is already assigned to "
            f"{existing_record.get('dataset_role', 'training')}"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scope_text(value: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value.strip()


def _scope_key(category: str, model_version: str) -> str:
    payload = json.dumps(
        [
            _scope_text(category, "category"),
            _scope_text(model_version, "model_version"),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{context} must be a valid SHA-256")
    return normalized


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _record_hash(row: Mapping[str, Any]) -> str:
    """Return and validate the content hash of one persisted feedback row."""
    unsigned = dict(row)
    declared = unsigned.pop("record_sha256", None)
    calculated = _canonical_sha256(unsigned)
    if declared is None:
        return calculated
    normalized = _require_sha256(declared, "feedback record_sha256")
    if normalized != calculated:
        raise ValueError("feedback record SHA-256 mismatch")
    return normalized


def _tensor_sha256(value: object, context: str) -> str:
    """Match the tiled-MLP tensor fingerprint format without loading a head."""
    try:
        tensor = torch.as_tensor(value).detach().to("cpu").contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("tensor contains a non-finite value")
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite tensor") from error


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(
                payload,
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Replace one metadata file without exposing a partial write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.uint8)
    normalized_scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(normalized_scores)):
        raise ValueError("classification scores must be finite")
    predictions = normalized_scores >= _finite_float(threshold, "threshold")
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }


def _revision_validation_acceptance(
    parent_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    minimum_recall: float,
) -> dict[str, Any]:
    """Accept an absolute gate pass or a safe relative validation improvement.

    A below-target candidate is useful to the operator only when recall and F1
    are both non-regressing and at least one improves materially.  This keeps
    incremental development visible without relabelling a regression as a
    successful optimization.
    """

    target = _finite_float(minimum_recall, "minimum_recall")
    if not 0.0 <= target <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")
    parent_recall = _finite_float(parent_metrics.get("recall"), "parent recall")
    parent_f1 = _finite_float(parent_metrics.get("f1"), "parent f1")
    candidate_recall = _finite_float(
        candidate_metrics.get("recall"), "candidate recall"
    )
    candidate_f1 = _finite_float(candidate_metrics.get("f1"), "candidate f1")
    epsilon = 1e-12
    absolute_gate_passed = bool(
        candidate_recall + epsilon >= target
        and candidate_f1 + epsilon >= parent_f1
    )
    non_regressing = bool(
        candidate_recall + epsilon >= parent_recall
        and candidate_f1 + epsilon >= parent_f1
    )
    materially_improved = bool(
        candidate_recall > parent_recall + epsilon
        or candidate_f1 > parent_f1 + epsilon
    )
    relative_improvement_passed = bool(
        non_regressing and materially_improved
    )
    passed = bool(absolute_gate_passed or relative_improvement_passed)
    if absolute_gate_passed:
        mode = "absolute_gate"
        reason = "absolute_gate_passed"
    elif relative_improvement_passed:
        mode = "relative_improvement"
        reason = "relative_improvement_passed"
    else:
        mode = "failed"
        reason = "no_safe_validation_improvement"
    return {
        "acceptance_policy": "absolute_or_safe_relative_v1",
        "passed": passed,
        "reason": reason,
        "acceptance_mode": mode,
        "absolute_gate_passed": absolute_gate_passed,
        "relative_improvement_passed": relative_improvement_passed,
        "recall_delta": candidate_recall - parent_recall,
        "f1_delta": candidate_f1 - parent_f1,
    }


def apply_threshold(result: dict[str, Any], threshold: float | None) -> dict[str, Any]:
    updated = dict(result)
    updated["base_fusion_threshold"] = _finite_float(
        result["fusion_threshold"], "fusion_threshold"
    )
    if threshold is not None:
        active_threshold = _finite_float(threshold, "feedback threshold")
        fusion_score = _finite_float(result["fusion_score"], "fusion_score")
        updated["fusion_threshold"] = active_threshold
        updated["decision"] = bool(fusion_score >= active_threshold)
        updated["feedback_threshold_active"] = True
    else:
        updated["feedback_threshold_active"] = False
    return updated


def _decode_image_size(image_bytes: bytes) -> tuple[int, int] | None:
    """Best-effort image dimensions for ROI validation.

    Feedback can be submitted by lightweight integrations with opaque bytes;
    those records are still accepted when no ROI is supplied.  A supplied ROI
    is always checked against an explicit ``image_size`` or a decodable image.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            return int(width), int(height)
    except Exception:
        return None


def _strict_roi(
    value: object,
    *,
    image_size: tuple[int, int] | list[int] | None = None,
) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("defect_region must be [left, top, right, bottom]")
    values: list[int] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, np.integer)):
            raise TypeError("defect_region coordinates must be integers")
        values.append(int(coordinate))
    left, top, right, bottom = values
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("defect_region must have positive area and non-negative bounds")
    if image_size is not None:
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise TypeError("image_size must be [width, height]")
        width, height = image_size
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, (int, np.integer))
            or not isinstance(height, (int, np.integer))
            or int(width) <= 0
            or int(height) <= 0
        ):
            raise TypeError("image_size must contain positive integers")
        if right > int(width) or bottom > int(height):
            raise ValueError("defect_region lies outside image bounds")
    return values


class FeedbackStore:
    _feedback_comparison_fields = (
        "prediction",
        "operator_label",
        "feedback_type",
        "defect_type",
        "note",
        "dataset_role",
        "defect_region",
        "fusion_score",
        "fusion_threshold",
        "base_fusion_threshold",
        "branch_margins",
        "defect_types",
        "model_decision",
        "appearance_decision",
        "overall_decision",
        "optimization_target",
        "adaptive_revision_id",
        "head_version",
        "feature_version",
        "threshold_version",
        "run_id",
        "bundle_hash",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.images = self.root / "images"
        self.records_path = self.root / "records.jsonl"
        self.records_lock_path = self.root / ".records.lock"
        self.candidates = self.root / "candidates"
        self.feature_banks = self.root / "feature_banks"
        self.registry_path = self.root / "threshold_registry.yaml"
        self.feature_registry_path = self.root / "feature_registry.yaml"
        # A single registry is the source of truth for a complete feedback
        # revision.  The older threshold/feature registries remain readable so
        # existing bundles and deployments continue to work.
        self.revision_registry_path = self.root / "revision_registry.yaml"
        self.revisions = self.root / "revisions"
        self._records_lock = threading.RLock()
        self.images.mkdir(parents=True, exist_ok=True)
        self.candidates.mkdir(parents=True, exist_ok=True)
        self.feature_banks.mkdir(parents=True, exist_ok=True)
        self.revisions.mkdir(parents=True, exist_ok=True)

    def records(
        self,
        category: str | None = None,
        dataset_role: str | None = None,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._records_guard():
            return self._records_unlocked(category, dataset_role, model_version)

    def _records_unlocked(
        self,
        category: str | None = None,
        dataset_role: str | None = None,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.records_path.is_file():
            return []
        rows = []
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("feedback record must be a JSON object")
            row.setdefault("dataset_role", "training")
            row.setdefault("defect_region", None)
            row.setdefault("image_size", None)
            row.setdefault("source_type", "image")
            row.setdefault("defect_types", [])
            row.setdefault("model_decision", bool(row.get("prediction", 0)))
            row.setdefault("appearance_decision", bool(row.get("model_decision", row.get("prediction", 0))))
            row.setdefault("overall_decision", bool(row.get("decision", row.get("prediction", 0))))
            row.setdefault(
                "optimization_target",
                "appearance",
            )
            row.setdefault("adaptive_revision_id", None)
            row.setdefault("head_version", None)
            row.setdefault("feature_version", None)
            row.setdefault("threshold_version", None)
            row.setdefault("run_id", row.get("model_version"))
            row.setdefault("bundle_hash", None)
            if "record_sha256" not in row:
                unsigned = dict(row)
                unsigned.pop("record_sha256", None)
                row["record_sha256"] = _canonical_sha256(unsigned)
            if (
                (category is None or row.get("category") == category)
                and (dataset_role is None or row["dataset_role"] == dataset_role)
                and (
                    model_version is None
                    or row.get("model_version") == model_version
                )
            ):
                rows.append(row)
        return rows

    def save_feedback(
        self,
        *,
        image_bytes: bytes,
        original_name: str,
        category: str,
        model_version: str,
        result: dict[str, Any],
        operator_label: int,
        defect_type: str,
        note: str,
        dataset_role: str = "training",
        defect_region: list[int] | None = None,
        source_type: str = "image",
        source_name: str | None = None,
        frame_index: int | None = None,
        image_size: tuple[int, int] | list[int] | None = None,
        adaptive_revision_id: str | None = None,
        head_version: str | None = None,
        feature_version: str | None = None,
        threshold_version: str | None = None,
        appearance_decision: bool | None = None,
        overall_decision: bool | None = None,
        optimization_target: str | None = None,
        run_id: str | None = None,
        bundle_hash: str | None = None,
    ) -> dict[str, Any]:
        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        if dataset_role not in {"training", "validation"}:
            raise ValueError("dataset_role must be training or validation")
        if operator_label not in {0, 1}:
            raise ValueError("operator_label must be 0 or 1")
        if image_size is None and defect_region is not None:
            image_size = _decode_image_size(image_bytes)
        defect_region = _strict_roi(defect_region, image_size=image_size)
        if run_id is not None:
            run_id = _scope_text(run_id, "run_id")
        if bundle_hash is not None:
            bundle_hash = _require_sha256(bundle_hash, "bundle_hash")
        if adaptive_revision_id is not None:
            adaptive_revision_id = _scope_text(adaptive_revision_id, "adaptive_revision_id")
        for name, value in (
            ("head_version", head_version),
            ("feature_version", feature_version),
            ("threshold_version", threshold_version),
        ):
            if value is not None:
                _scope_text(value, name)
        if optimization_target is None:
            optimization_target = "appearance"
        optimization_target = _scope_text(optimization_target, "optimization_target")
        if optimization_target != "appearance":
            raise ValueError("new feedback must target the appearance model")
        suffix = Path(original_name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            suffix = ".png"
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        feedback_id = (
            f"feedback-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
            f"{uuid4().hex[:8]}"
        )
        image_path = self.images / f"{feedback_id}{suffix}"
        prediction = int(bool(result["decision"]))
        if prediction == operator_label:
            feedback_type = "confirmed"
        elif prediction == 1:
            feedback_type = "false_positive"
        else:
            feedback_type = "false_negative"
        appearance_value = bool(
            result.get(
                "appearance_decision",
                result.get("model_decision", result["decision"]),
            )
            if appearance_decision is None
            else appearance_decision
        )
        overall_value = bool(
            result.get("overall_decision", result["decision"])
            if overall_decision is None
            else overall_decision
        )
        record = {
            "schema_version": 2,
            "feedback_id": feedback_id,
            "created_at": _now(),
            "category": category,
            "model_version": model_version,
            "original_name": Path(original_name).name,
            "image": image_path.relative_to(self.root).as_posix(),
            "image_sha256": image_sha256,
            "prediction": prediction,
            "operator_label": int(operator_label),
            "feedback_type": feedback_type,
            "defect_type": defect_type.strip(),
            "note": note.strip(),
            "dataset_role": dataset_role,
            "defect_region": defect_region,
            "image_size": None if image_size is None else [int(image_size[0]), int(image_size[1])],
            "source_type": source_type,
            "source_name": Path(source_name or original_name).name,
            "frame_index": frame_index,
            "fusion_score": _finite_float(result["fusion_score"], "fusion_score"),
            "fusion_threshold": _finite_float(
                result["fusion_threshold"], "fusion_threshold"
            ),
            "base_fusion_threshold": _finite_float(
                result.get("base_fusion_threshold", result["fusion_threshold"]),
                "base_fusion_threshold",
            ),
            "branch_margins": {
                key: _finite_float(value, f"branch_margins.{key}")
                for key, value in result["branch_margins"].items()
            },
            "defect_types": [str(value) for value in result.get("defect_types", [])],
            "model_decision": bool(result.get("model_decision", result["decision"])),
            "appearance_decision": appearance_value,
            "overall_decision": overall_value,
            "optimization_target": optimization_target,
            "adaptive_revision_id": adaptive_revision_id,
            "head_version": head_version,
            "feature_version": feature_version,
            "threshold_version": threshold_version,
            "run_id": run_id or model_version,
            "bundle_hash": bundle_hash,
        }
        record["record_sha256"] = _canonical_sha256(record)
        with self._records_guard():
            existing_rows: list[dict[str, Any]] = []
            if self.records_path.is_file():
                existing_rows = [
                    json.loads(line)
                    for line in self.records_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
            duplicate = next(
                (
                    row
                    for row in existing_rows
                    if row.get("category") == category
                    and row.get("model_version") == model_version
                    and row.get("image_sha256") == image_sha256
                ),
                None,
            )
            if duplicate is not None:
                raise DuplicateFeedbackError(duplicate, record)
            # The image and its record form one logical submission.  Keep the
            # previous file length and remove the image on any write failure so
            # callers never observe an orphan image or partial JSON line.
            records_existed = self.records_path.is_file()
            previous_length = self.records_path.stat().st_size if records_existed else 0
            image_written = False
            try:
                image_path.write_bytes(image_bytes)
                image_written = True
                with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                if self.records_path.is_file():
                    try:
                        if records_existed:
                            with self.records_path.open("r+b") as stream:
                                stream.truncate(previous_length)
                                stream.flush()
                                os.fsync(stream.fileno())
                        elif previous_length == 0:
                            self.records_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if image_written or image_path.exists():
                    image_path.unlink(missing_ok=True)
                raise
        return record

    @classmethod
    def feedback_changes(
        cls,
        existing_record: Mapping[str, Any],
        proposed_record: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Return persisted feedback/evidence fields that would change."""
        if not isinstance(existing_record, Mapping) or not isinstance(
            proposed_record, Mapping
        ):
            raise TypeError("feedback records must be mappings")
        for key in ("category", "model_version", "image_sha256"):
            if existing_record.get(key) != proposed_record.get(key):
                raise ValueError(f"replacement feedback {key} does not match")

        defaults: dict[str, Any] = {
            "dataset_role": "training",
            "defect_region": None,
            "defect_types": [],
            "model_decision": bool(existing_record.get("prediction", 0)),
            "appearance_decision": bool(
                existing_record.get(
                    "model_decision", existing_record.get("prediction", 0)
                )
            ),
            "overall_decision": bool(
                existing_record.get("decision", existing_record.get("prediction", 0))
            ),
            "optimization_target": "appearance",
            "adaptive_revision_id": None,
            "head_version": None,
            "feature_version": None,
            "threshold_version": None,
            "run_id": existing_record.get("model_version"),
            "bundle_hash": None,
        }
        changes: dict[str, dict[str, Any]] = {}
        for field in cls._feedback_comparison_fields:
            old_value = existing_record.get(field, defaults.get(field))
            new_value = proposed_record.get(field, defaults.get(field))
            if old_value != new_value:
                changes[field] = {"old": old_value, "new": new_value}
        return changes

    @staticmethod
    def feedback_record_sha256(record: Mapping[str, Any]) -> str:
        """Return the verified persisted-record hash for compare-and-swap."""
        if not isinstance(record, Mapping):
            raise TypeError("feedback record must be a mapping")
        return _record_hash(record)

    def replace_feedback(
        self,
        existing_feedback_id: str,
        proposed_record: Mapping[str, Any],
        *,
        expected_record_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Atomically replace one unprotected same-image feedback record.

        The original feedback ID, creation time, and managed image are retained.
        A compare-and-swap hash prevents a confirmation dialog from overwriting a
        record that another process changed while the operator was deciding.
        """
        feedback_id = _scope_text(existing_feedback_id, "existing_feedback_id")
        if not isinstance(proposed_record, Mapping):
            raise TypeError("proposed_record must be a mapping")
        proposed = dict(proposed_record)
        _record_hash(proposed)
        expected_hash = (
            None
            if expected_record_sha256 is None
            else _require_sha256(expected_record_sha256, "expected_record_sha256")
        )

        with self._records_guard():
            if not self.records_path.is_file():
                raise FileNotFoundError("feedback records file does not exist")
            original_text = self.records_path.read_text(encoding="utf-8")
            parsed_lines: list[tuple[str, dict[str, Any] | None]] = []
            existing: dict[str, Any] | None = None
            existing_index = -1
            for line_number, line in enumerate(original_text.splitlines(), start=1):
                if not line.strip():
                    parsed_lines.append((line, None))
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(
                        f"feedback record at line {line_number} must be a JSON object"
                    )
                parsed_lines.append((line, row))
                if str(row.get("feedback_id", "")) == feedback_id:
                    if existing is not None:
                        raise ValueError(f"duplicate feedback_id in records: {feedback_id}")
                    existing = row
                    existing_index = len(parsed_lines) - 1
            if existing is None:
                raise KeyError(f"feedback record not found: {feedback_id}")

            # Archived audit-only records remain readable and hash-verifiable,
            # but the removed write path must not rewrite them.
            if existing.get("optimization_target") == "audit_only":
                raise PermissionError("archived audit feedback cannot be replaced")

            current_hash = _record_hash(existing)
            if expected_hash is not None and current_hash != expected_hash:
                raise RuntimeError(
                    "feedback record changed after replacement confirmation"
                )
            self.feedback_changes(existing, proposed)
            if feedback_id in self._revision_referenced_feedback_ids_unlocked():
                raise PermissionError(
                    "cannot replace feedback evidence referenced by a candidate, "
                    "active, or rollback revision: " + feedback_id
                )

            replacement = dict(proposed)
            for field in (
                "feedback_id",
                "created_at",
                "original_name",
                "image",
                "image_sha256",
                "image_size",
                "source_type",
                "source_name",
                "frame_index",
            ):
                replacement[field] = existing.get(field)
            replacement["updated_at"] = _now()
            replacement["replacement_count"] = int(
                existing.get("replacement_count", 0)
            ) + 1
            replacement["previous_record_sha256"] = current_hash
            replacement.pop("record_sha256", None)
            replacement["record_sha256"] = _canonical_sha256(replacement)

            replacement_line = json.dumps(
                replacement, ensure_ascii=False, separators=(",", ":")
            )
            output_lines = [line for line, _row in parsed_lines]
            output_lines[existing_index] = replacement_line
            replacement_text = "\n".join(output_lines)
            if output_lines:
                replacement_text += "\n"
            temporary_path = self.records_path.with_name(
                f".{self.records_path.name}.{uuid4().hex}.tmp"
            )
            try:
                with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(replacement_text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.records_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            return replacement

    def delete_feedback(
        self,
        feedback_ids: list[str] | tuple[str, ...],
        *,
        category: str | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        """Delete matching records and their now-unreferenced managed images."""
        if isinstance(feedback_ids, (str, bytes)):
            raise TypeError("feedback_ids must be a list or tuple of strings")
        requested: set[str] = set()
        for feedback_id in feedback_ids:
            if not isinstance(feedback_id, str):
                raise TypeError("feedback_ids must contain only strings")
            if normalized := feedback_id.strip():
                requested.add(normalized)

        result: dict[str, Any] = {
            "deleted_records": 0,
            "deleted_images": 0,
            "deleted_ids": [],
            "missing_ids": sorted(requested),
            "missing_images": [],
            "failed_images": [],
        }
        if not requested:
            return result

        with self._records_guard():
            if not self.records_path.is_file():
                return result

            original_text = self.records_path.read_text(encoding="utf-8")
            parsed_lines: list[tuple[str, dict[str, Any] | None]] = []
            for line_number, line in enumerate(original_text.splitlines(), start=1):
                if not line.strip():
                    parsed_lines.append((line, None))
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(
                        f"feedback record at line {line_number} must be a JSON object"
                    )
                parsed_lines.append((line, row))

            kept_lines: list[str] = []
            kept_rows: list[dict[str, Any]] = []
            deleted_rows: list[dict[str, Any]] = []
            matched_ids: set[str] = set()
            for original_line, row in parsed_lines:
                if row is None:
                    kept_lines.append(original_line)
                    continue
                feedback_id = str(row.get("feedback_id", ""))
                matches = feedback_id in requested and (
                    category is None or row.get("category") == category
                ) and (
                    model_version is None
                    or row.get("model_version") == model_version
                )
                if matches:
                    matched_ids.add(feedback_id)
                    deleted_rows.append(row)
                else:
                    kept_lines.append(original_line)
                    kept_rows.append(row)

            if not deleted_rows:
                return result

            protected_ids = self._revision_referenced_feedback_ids_unlocked()
            blocked = sorted(
                str(row.get("feedback_id"))
                for row in deleted_rows
                if str(row.get("feedback_id")) in protected_ids
            )
            if blocked:
                raise PermissionError(
                    "cannot delete feedback evidence referenced by an active or "
                    "rollback revision: " + ", ".join(blocked)
                )

            temporary_path = self.records_path.with_name(
                f".{self.records_path.name}.{uuid4().hex}.tmp"
            )
            replacement = "\n".join(kept_lines)
            if kept_lines:
                replacement += "\n"
            try:
                with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(replacement)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.records_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

            retained_images = {
                path
                for row in kept_rows
                if (path := self._managed_image_path(row.get("image"))) is not None
            }
            delete_images = {
                path
                for row in deleted_rows
                if (path := self._managed_image_path(row.get("image"))) is not None
                and path not in retained_images
            }
            for image_path in sorted(delete_images):
                try:
                    if image_path.is_file():
                        image_path.unlink()
                        result["deleted_images"] += 1
                    else:
                        result["missing_images"].append(str(image_path))
                except OSError as error:
                    result["failed_images"].append(
                        {"path": str(image_path), "error": str(error)}
                    )

            result["deleted_records"] = len(deleted_rows)
            result["deleted_ids"] = sorted(matched_ids)
            result["missing_ids"] = sorted(requested - matched_ids)
            return result

    def _revision_referenced_feedback_ids_unlocked(self) -> set[str]:
        """Collect IDs retained by candidate, active, or rollback evidence."""
        protected: set[str] = set()

        def collect(value: object) -> None:
            if isinstance(value, Mapping):
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, (list, tuple, set)):
                for nested in value:
                    collect(nested)
            elif isinstance(value, str) and value.startswith("feedback-"):
                protected.add(value)

        evidence_paths = [
            self.revision_registry_path,
            self.registry_path,
            self.feature_registry_path,
            *sorted(self.candidates.glob("*.yaml")),
        ]
        for evidence_path in evidence_paths:
            if not evidence_path.exists():
                continue
            try:
                if evidence_path.is_symlink() or not evidence_path.is_file():
                    raise ValueError("evidence path is not a regular file")
                payload = yaml.safe_load(
                    evidence_path.read_text(encoding="utf-8")
                ) or {}
            except Exception as error:
                raise PermissionError(
                    "cannot modify feedback while version evidence is corrupt: "
                    f"{evidence_path}"
                ) from error
            collect(payload)
        return protected

    def clear_feedback(
        self,
        *,
        category: str,
        model_version: str,
    ) -> dict[str, Any]:
        rows = self.records(category=category, model_version=model_version)
        return self.delete_feedback(
            [str(row["feedback_id"]) for row in rows],
            category=category,
            model_version=model_version,
        )

    @staticmethod
    def _purge_is_link(path: Path) -> bool:
        junction_checker = getattr(os.path, "isjunction", None)
        if path.is_symlink() or bool(
            callable(junction_checker) and junction_checker(path)
        ):
            return True
        # ``os.path.isjunction`` is unavailable on the project's Python 3.10.
        # Windows exposes junctions and other reparse points through lstat.
        try:
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
        except OSError:
            return False
        return bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )

    def _purge_managed_path(
        self,
        path: Path,
        managed_root: Path,
        *,
        context: str,
        expected: str,
        allow_root: bool = False,
    ) -> Path:
        """Resolve one existing path without following a managed-store link."""

        managed_root = Path(managed_root)
        path = Path(path)
        if self._purge_is_link(managed_root):
            raise PermissionError(f"{context} managed root must not be a link")
        if not managed_root.exists():
            raise FileNotFoundError(f"{context} managed root does not exist: {managed_root}")
        raw_root = managed_root.absolute()
        raw_path = path.absolute()
        try:
            raw_relative = raw_path.relative_to(raw_root)
        except ValueError as error:
            raise PermissionError(f"{context} path escapes its managed root") from error
        if not allow_root and not raw_relative.parts:
            raise PermissionError(f"{context} path must name a managed child")
        cursor = raw_root
        for part in raw_relative.parts:
            cursor = cursor / part
            if self._purge_is_link(cursor):
                raise PermissionError(f"{context} path must not contain a link: {cursor}")
        resolved_root = managed_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as error:
            raise PermissionError(f"{context} path escapes its managed root") from error
        if not allow_root and not relative.parts:
            raise PermissionError(f"{context} path must name a managed child")
        if expected == "file" and not resolved.is_file():
            raise ValueError(f"{context} path is not a regular file: {resolved}")
        if expected == "directory" and not resolved.is_dir():
            raise ValueError(f"{context} path is not a directory: {resolved}")
        return resolved

    def _purge_yaml_files(self, root: Path) -> list[Path]:
        """Return YAML files below a managed directory, rejecting all links."""

        root = Path(root)
        if self._purge_is_link(root):
            raise PermissionError(
                f"purge evidence directory must not be a link: {root}"
            )
        if not root.exists():
            return []
        resolved_root = self._purge_managed_path(
            root,
            self.root,
            context="purge evidence directory",
            expected="directory",
            allow_root=True,
        )
        result: list[Path] = []
        for current, directory_names, file_names in os.walk(
            resolved_root, followlinks=False
        ):
            current_path = Path(current)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = current_path / name
                if self._purge_is_link(child):
                    raise PermissionError(
                        f"purge evidence directory must not contain a link: {child}"
                    )
            for name in file_names:
                child = current_path / name
                if self._purge_is_link(child):
                    raise PermissionError(
                        f"purge evidence directory must not contain a link: {child}"
                    )
                if child.suffix.lower() not in {".yaml", ".yml"}:
                    continue
                result.append(
                    self._purge_managed_path(
                        child,
                        resolved_root,
                        context="purge YAML evidence",
                        expected="file",
                    )
                )
        return sorted(result, key=lambda value: value.as_posix().casefold())

    @staticmethod
    def _purge_yaml_mapping(path: Path, context: str) -> dict[str, Any]:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as error:
            raise ValueError(f"cannot parse {context}: {path}") from error
        if not isinstance(payload, dict):
            raise TypeError(f"{context} must be a YAML mapping: {path}")
        return payload

    @staticmethod
    def _purge_verify_payload_hash(
        payload: Mapping[str, Any], field: str, context: str
    ) -> None:
        expected = _require_sha256(payload.get(field), f"{context} {field}")
        unsigned = dict(payload)
        unsigned.pop(field, None)
        if _canonical_sha256(unsigned) != expected:
            raise ValueError(f"{context} payload SHA-256 mismatch")

    @staticmethod
    def _purge_scope_matches(
        payload: Mapping[str, Any],
        *,
        category: str,
        model_version: str,
        bundle_hash: str | None,
        require_bundle: bool = False,
    ) -> bool:
        raw_category = payload.get("category")
        if raw_category is None:
            return False
        if not isinstance(raw_category, str) or not raw_category.strip():
            raise TypeError("purge evidence category must be a non-empty string")
        if raw_category.strip() != category:
            return False
        versions = {
            str(payload[field]).strip()
            for field in ("model_version", "run_id")
            if payload.get(field) not in {None, ""}
        }
        if not versions:
            return False
        if len(versions) != 1:
            raise ValueError("purge evidence model_version/run_id scope is inconsistent")
        if next(iter(versions)) != model_version:
            return False
        declared_bundle = payload.get("bundle_hash")
        if declared_bundle in {None, ""}:
            if bundle_hash is not None and require_bundle:
                raise ValueError("purge revision evidence is missing bundle_hash")
            return True
        normalized_bundle = _require_sha256(
            declared_bundle, "purge evidence bundle_hash"
        )
        return bundle_hash is None or normalized_bundle == bundle_hash

    @staticmethod
    def _purge_filter_registry(
        payload: Mapping[str, Any],
        *,
        matcher: Any,
        legacy_category: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
        updated = copy.deepcopy(dict(payload))
        active = updated.setdefault("active", {})
        history = updated.setdefault("history", [])
        if not isinstance(active, dict) or not isinstance(history, list):
            raise TypeError("purge registry active/history fields are invalid")
        removed: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        new_active: dict[str, Any] = {}
        for key, raw_entry in active.items():
            if not isinstance(raw_entry, Mapping):
                raise TypeError("purge registry active entry must be a mapping")
            entry = copy.deepcopy(dict(raw_entry))
            if matcher(entry):
                removed.append(entry)
            else:
                new_active[str(key)] = entry
                retained.append(entry)
        new_history: list[Any] = []
        for raw_entry in history:
            if not isinstance(raw_entry, Mapping):
                raise TypeError("purge registry history entry must be a mapping")
            entry = copy.deepcopy(dict(raw_entry))
            if matcher(entry):
                removed.append(entry)
            else:
                new_history.append(entry)
                retained.append(entry)
        updated["active"] = new_active
        updated["history"] = new_history
        legacy_removed = 0
        if "legacy_unscoped" in updated:
            legacy = updated["legacy_unscoped"]
            if not isinstance(legacy, dict):
                raise TypeError("purge registry legacy_unscoped must be a mapping")
            if legacy_category is not None and legacy_category in legacy:
                raw_entry = legacy.pop(legacy_category)
                if not isinstance(raw_entry, Mapping):
                    raise TypeError(
                        "purge registry legacy_unscoped entry must be a mapping"
                    )
                removed.append(copy.deepcopy(dict(raw_entry)))
                legacy_removed = 1
            for raw_entry in legacy.values():
                if not isinstance(raw_entry, Mapping):
                    raise TypeError(
                        "purge registry legacy_unscoped entry must be a mapping"
                    )
                retained.append(copy.deepcopy(dict(raw_entry)))
        return updated, removed, retained, legacy_removed

    @staticmethod
    def _purge_collect_fields(value: object, fields: set[str]) -> set[str]:
        result: set[str] = set()

        def collect(item: object) -> None:
            if isinstance(item, Mapping):
                for key, nested in item.items():
                    if key in fields and isinstance(nested, str) and nested.strip():
                        result.add(nested.strip())
                    collect(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    collect(nested)

        collect(value)
        return result

    @staticmethod
    def _purge_feature_paths(value: object) -> set[str]:
        result: set[str] = set()

        def collect(item: object) -> None:
            if isinstance(item, Mapping):
                if (
                    isinstance(item.get("path"), str)
                    and "file_sha256" in item
                    and "version" in item
                ):
                    result.add(str(item["path"]).strip())
                for nested in item.values():
                    collect(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    collect(nested)

        collect(value)
        return {path for path in result if path}

    def _purge_tree_evidence(self, path: Path) -> dict[str, Any]:
        if self._purge_is_link(path):
            raise PermissionError(f"purge artifact tree must not be a link: {path}")
        if path.is_file():
            return {
                "type": "file",
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        entries: list[dict[str, Any]] = []
        for current, directory_names, file_names in os.walk(path, followlinks=False):
            current_path = Path(current)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = current_path / name
                if self._purge_is_link(child):
                    raise PermissionError(
                        f"purge artifact tree must not contain a link: {child}"
                    )
                entries.append(
                    {
                        "path": child.relative_to(path).as_posix(),
                        "type": "directory",
                    }
                )
            for name in file_names:
                child = current_path / name
                if self._purge_is_link(child) or not child.is_file():
                    raise PermissionError(
                        f"purge artifact tree must contain only regular files: {child}"
                    )
                entries.append(
                    {
                        "path": child.relative_to(path).as_posix(),
                        "type": "file",
                        "bytes": int(child.stat().st_size),
                        "sha256": _sha256_file(child),
                    }
                )
        return {"type": "directory", "entries": entries}

    def _build_product_evidence_purge_plan_unlocked(
        self,
        *,
        category: str,
        model_version: str,
        bundle_hash: str | None,
    ) -> dict[str, Any]:
        """Build a validated internal purge transaction while holding the lock."""

        watched_files: dict[str, str | None] = {}

        def watch(path: Path) -> None:
            relative = path.relative_to(self.root).as_posix()
            watched_files[relative] = _sha256_file(path) if path.is_file() else None

        # Parse records without normalising away corrupt persisted fields.
        original_record_lines: list[str] = []
        target_rows: list[dict[str, Any]] = []
        retained_rows: list[dict[str, Any]] = []
        if self._purge_is_link(self.records_path):
            raise PermissionError("feedback records must not be a link")
        if self.records_path.exists():
            records_path = self._purge_managed_path(
                self.records_path,
                self.root,
                context="feedback records",
                expected="file",
            )
            watch(records_path)
            original_record_lines = records_path.read_text(encoding="utf-8").splitlines()
        kept_record_lines: list[str] = []
        for line_number, line in enumerate(original_record_lines, start=1):
            if not line.strip():
                kept_record_lines.append(line)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"cannot parse feedback record at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(
                    f"feedback record at line {line_number} must be a JSON object"
                )
            _record_hash(row)
            if self._purge_scope_matches(
                row,
                category=category,
                model_version=model_version,
                bundle_hash=bundle_hash,
            ):
                target_rows.append(row)
            else:
                kept_record_lines.append(line)
                retained_rows.append(row)
        replacement_records = "\n".join(kept_record_lines)
        if kept_record_lines:
            replacement_records += "\n"

        retained_images: set[Path] = set()
        for row in retained_rows:
            candidate = self._managed_image_path(row.get("image"))
            if candidate is not None and candidate.exists():
                retained_images.add(
                    self._purge_managed_path(
                        candidate,
                        self.images,
                        context="retained feedback image",
                        expected="file",
                    )
                )
        target_images: set[Path] = set()
        for row in target_rows:
            candidate = self._managed_image_path(row.get("image"))
            if candidate is None:
                raise PermissionError(
                    f"target feedback image is outside the managed image directory: {row.get('image')}"
                )
            managed = self._purge_managed_path(
                candidate,
                self.images,
                context="target feedback image",
                expected="file",
            )
            if managed not in retained_images:
                target_images.add(managed)

        # Candidate receipts share one directory across legacy and unified flows.
        candidate_files = self._purge_yaml_files(self.candidates)
        target_candidates: set[Path] = set()
        threshold_candidate_ids: set[str] = set()
        revision_candidate_ids: set[str] = set()
        target_revision_dirs: set[Path] = set()
        for path in candidate_files:
            watch(path)
            payload = self._purge_yaml_mapping(path, "feedback candidate receipt")
            is_revision = payload.get("revision_id") not in {None, ""}
            matches = self._purge_scope_matches(
                payload,
                category=category,
                model_version=model_version,
                bundle_hash=bundle_hash,
                require_bundle=is_revision and bundle_hash is not None,
            )
            if not matches:
                continue
            self._purge_verify_payload_hash(
                payload,
                "payload_sha256",
                "feedback revision" if is_revision else "threshold candidate",
            )
            target_candidates.add(path)
            if is_revision:
                revision_id = _scope_text(payload.get("revision_id"), "revision_id")
                if Path(revision_id).name != revision_id or revision_id in {".", ".."}:
                    raise ValueError("revision_id must be a safe file identifier")
                if path != (self.candidates / f"{revision_id}.yaml").resolve():
                    raise ValueError("revision receipt filename does not match revision_id")
                revision_candidate_ids.add(revision_id)
                revision_dir = self._purge_managed_path(
                    self.revisions / revision_id,
                    self.revisions,
                    context="revision artifact directory",
                    expected="directory",
                )
                artifacts = payload.get("artifacts")
                if not isinstance(artifacts, Mapping):
                    raise ValueError("revision candidate has no artifact mapping")
                for artifact in artifacts.values():
                    if artifact is None:
                        continue
                    if not isinstance(artifact, Mapping):
                        raise TypeError("revision artifact entry must be a mapping")
                    artifact_path = self._safe_revision_path(
                        artifact.get("path"), must_exist=True
                    )
                    try:
                        artifact_path.relative_to(revision_dir)
                    except ValueError as error:
                        raise PermissionError(
                            "revision purge artifact escapes its revision directory"
                        ) from error
                target_revision_dirs.add(revision_dir)
            else:
                candidate_id = _scope_text(payload.get("candidate_id"), "candidate_id")
                if Path(candidate_id).name != candidate_id or candidate_id in {".", ".."}:
                    raise ValueError("candidate_id must be a safe file identifier")
                if path != (self.candidates / f"{candidate_id}.yaml").resolve():
                    raise ValueError("candidate receipt filename does not match candidate_id")
                threshold_candidate_ids.add(candidate_id)

        # Load and filter all three registries.  Existing files are included in
        # the CAS digest even when the selected scope has no entry in one of them.
        metadata_updates: dict[Path, bytes] = {}
        registry_counts = {
            "threshold_registry_entries": 0,
            "feature_registry_entries": 0,
            "revision_registry_entries": 0,
        }

        if self._purge_is_link(self.registry_path):
            raise PermissionError("threshold registry must not be a link")
        if self.registry_path.exists():
            registry_path = self._purge_managed_path(
                self.registry_path,
                self.root,
                context="threshold registry",
                expected="file",
            )
            watch(registry_path)
            threshold_registry = self.registry()
            updated, removed, retained, legacy_removed = self._purge_filter_registry(
                threshold_registry,
                matcher=lambda entry: self._purge_scope_matches(
                    entry,
                    category=category,
                    model_version=model_version,
                    bundle_hash=None,
                ),
                legacy_category=category,
            )
            removed_count = len(removed)
            registry_counts["threshold_registry_entries"] = removed_count
            if removed_count:
                metadata_updates[registry_path] = yaml.safe_dump(
                    updated, sort_keys=False, allow_unicode=True
                ).encode("utf-8")
            removed_ids = self._purge_collect_fields(
                removed, {"candidate_id", "from_candidate_id", "to_candidate_id"}
            )
            retained_ids = self._purge_collect_fields(
                retained, {"candidate_id", "from_candidate_id", "to_candidate_id"}
            )
            if removed_ids - threshold_candidate_ids:
                raise FileNotFoundError(
                    "threshold registry references missing candidate receipts: "
                    + ", ".join(sorted(removed_ids - threshold_candidate_ids))
                )
            if retained_ids & threshold_candidate_ids:
                raise PermissionError(
                    "retained threshold registry references purge candidates"
                )

        feature_paths: set[Path] = set()
        if self._purge_is_link(self.feature_registry_path):
            raise PermissionError("feature registry must not be a link")
        if self.feature_registry_path.exists():
            registry_path = self._purge_managed_path(
                self.feature_registry_path,
                self.root,
                context="feature registry",
                expected="file",
            )
            watch(registry_path)
            feature_registry = self.feature_registry()
            updated, removed, retained, legacy_removed = self._purge_filter_registry(
                feature_registry,
                matcher=lambda entry: self._purge_scope_matches(
                    entry,
                    category=category,
                    model_version=model_version,
                    bundle_hash=None,
                ),
                legacy_category=category,
            )
            removed_count = len(removed)
            registry_counts["feature_registry_entries"] = removed_count
            if removed_count:
                metadata_updates[registry_path] = yaml.safe_dump(
                    updated, sort_keys=False, allow_unicode=True
                ).encode("utf-8")
            removed_paths = self._purge_feature_paths(removed)
            retained_paths = self._purge_feature_paths(retained)
            for raw in sorted(removed_paths - retained_paths):
                candidate = Path(raw)
                if candidate.is_absolute():
                    raise PermissionError("feature-bank purge path must be relative")
                feature_paths.add(
                    self._purge_managed_path(
                        self.root / candidate,
                        self.feature_banks,
                        context="feature-bank artifact",
                        expected="file",
                    )
                )
            for raw in sorted(retained_paths):
                candidate = Path(raw)
                if candidate.is_absolute():
                    raise PermissionError("retained feature-bank path must be relative")
                self._purge_managed_path(
                    self.root / candidate,
                    self.feature_banks,
                    context="retained feature-bank artifact",
                    expected="file",
                )

        if self._purge_is_link(self.revision_registry_path):
            raise PermissionError("revision registry must not be a link")
        if self.revision_registry_path.exists():
            registry_path = self._purge_managed_path(
                self.revision_registry_path,
                self.root,
                context="revision registry",
                expected="file",
            )
            watch(registry_path)
            revision_registry = self._revision_registry()
            updated, removed, retained, _legacy_removed = self._purge_filter_registry(
                revision_registry,
                matcher=lambda entry: self._purge_scope_matches(
                    entry,
                    category=category,
                    model_version=model_version,
                    bundle_hash=bundle_hash,
                    require_bundle=bundle_hash is not None,
                ),
            )
            removed_count = len(removed)
            registry_counts["revision_registry_entries"] = removed_count
            if removed_count:
                metadata_updates[registry_path] = yaml.safe_dump(
                    updated, sort_keys=False, allow_unicode=True
                ).encode("utf-8")
            removed_ids = self._purge_collect_fields(
                removed, {"revision_id", "from_revision_id", "to_revision_id"}
            )
            retained_ids = self._purge_collect_fields(
                retained, {"revision_id", "from_revision_id", "to_revision_id"}
            )
            if removed_ids - revision_candidate_ids:
                raise FileNotFoundError(
                    "revision registry references missing candidate receipts: "
                    + ", ".join(sorted(removed_ids - revision_candidate_ids))
                )
            if retained_ids & revision_candidate_ids:
                raise PermissionError(
                    "retained revision registry references purge revisions"
                )

        # Query manifests are an auditable product/run output but are not part
        # of the revision registries.
        query_root = self.root / "queries"
        query_files = self._purge_yaml_files(query_root)
        target_queries: set[Path] = set()
        for path in query_files:
            watch(path)
            payload = self._purge_yaml_mapping(path, "active-learning query")
            if not self._purge_scope_matches(
                payload,
                category=category,
                model_version=model_version,
                bundle_hash=bundle_hash,
            ):
                continue
            self._purge_verify_payload_hash(
                payload, "manifest_sha256", "active-learning query"
            )
            target_queries.add(path)

        if target_rows:
            metadata_updates[self.records_path] = replacement_records.encode("utf-8")

        move_paths: set[Path] = {
            *target_images,
            *target_candidates,
            *target_revision_dirs,
            *feature_paths,
            *target_queries,
        }
        ordered_move_paths: list[Path] = []
        for path in sorted(move_paths, key=lambda value: (len(value.parts), value.as_posix())):
            if any(path == parent or parent in path.parents for parent in ordered_move_paths):
                continue
            ordered_move_paths.append(path)

        counts = {
            "records": len(target_rows),
            "images": len(target_images),
            "candidates": len(target_candidates),
            "revisions": len(target_revision_dirs),
            "feature_banks": len(feature_paths),
            "queries": len(target_queries),
            **registry_counts,
        }
        paths = {
            "images": sorted(path.relative_to(self.root).as_posix() for path in target_images),
            "candidates": sorted(path.relative_to(self.root).as_posix() for path in target_candidates),
            "revisions": sorted(path.relative_to(self.root).as_posix() for path in target_revision_dirs),
            "feature_banks": sorted(path.relative_to(self.root).as_posix() for path in feature_paths),
            "queries": sorted(path.relative_to(self.root).as_posix() for path in target_queries),
        }
        target_evidence = {
            path.relative_to(self.root).as_posix(): self._purge_tree_evidence(path)
            for path in ordered_move_paths
        }
        plan_state = {
            "schema_version": 1,
            "category": category,
            "model_version": model_version,
            "bundle_hash": bundle_hash,
            "counts": counts,
            "paths": paths,
            "watched_files": dict(sorted(watched_files.items())),
            "target_evidence": target_evidence,
        }
        plan_sha256 = _canonical_sha256(plan_state)
        public = {
            "schema_version": 1,
            "category": category,
            "model_version": model_version,
            "bundle_hash": bundle_hash,
            "counts": counts,
            "paths": paths,
            "plan_sha256": plan_sha256,
        }
        return {
            "public": public,
            "metadata_updates": metadata_updates,
            "move_paths": ordered_move_paths,
        }

    def plan_product_evidence_purge(
        self,
        category: str,
        model_version: str,
        bundle_hash: str | None = None,
    ) -> dict[str, Any]:
        """Return a stable, read-only inventory for one product evidence purge."""

        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        normalized_bundle = (
            None
            if bundle_hash in {None, ""}
            else _require_sha256(bundle_hash, "bundle_hash")
        )
        with self._records_guard():
            plan = self._build_product_evidence_purge_plan_unlocked(
                category=category,
                model_version=model_version,
                bundle_hash=normalized_bundle,
            )
            return copy.deepcopy(plan["public"])

    def purge_product_evidence(
        self,
        category: str,
        model_version: str,
        bundle_hash: str | None = None,
        expected_plan_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Atomically detach and delete all managed evidence for one model scope."""

        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        normalized_bundle = (
            None
            if bundle_hash in {None, ""}
            else _require_sha256(bundle_hash, "bundle_hash")
        )
        expected_plan = (
            None
            if expected_plan_sha256 in {None, ""}
            else _require_sha256(expected_plan_sha256, "expected_plan_sha256")
        )
        with self._records_guard():
            plan = self._build_product_evidence_purge_plan_unlocked(
                category=category,
                model_version=model_version,
                bundle_hash=normalized_bundle,
            )
            public = copy.deepcopy(plan["public"])
            if expected_plan is not None and public["plan_sha256"] != expected_plan:
                raise RuntimeError(
                    "product evidence purge plan changed after confirmation"
                )

            move_paths: list[Path] = plan["move_paths"]
            metadata_updates: dict[Path, bytes] = plan["metadata_updates"]
            if not move_paths and not metadata_updates:
                return {
                    **public,
                    "pending_cleanup_path": None,
                    "failed_cleanup": [],
                }

            quarantine = self.root / f".purge-{uuid4().hex}"
            moved: list[tuple[Path, Path]] = []
            metadata_backups = {
                path: path.read_bytes() if path.exists() else None
                for path in metadata_updates
            }

            def restore_moved() -> list[str]:
                failures: list[str] = []
                for source, destination in reversed(moved):
                    try:
                        source.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(destination, source)
                    except OSError as error:
                        failures.append(f"{destination} -> {source}: {error}")
                return failures

            try:
                if move_paths:
                    quarantine.mkdir(parents=False, exist_ok=False)
                    for source in move_paths:
                        destination = quarantine / source.relative_to(self.root)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(source, destination)
                        moved.append((source, destination))
            except Exception as error:
                rollback_failures = restore_moved()
                try:
                    if quarantine.exists():
                        shutil.rmtree(quarantine)
                except OSError as cleanup_error:
                    rollback_failures.append(str(cleanup_error))
                if rollback_failures:
                    raise RuntimeError(
                        "product evidence isolation failed and rollback was incomplete: "
                        + "; ".join(rollback_failures)
                    ) from error
                raise

            try:
                for path in sorted(
                    metadata_updates, key=lambda value: value.as_posix().casefold()
                ):
                    _write_bytes_atomic(path, metadata_updates[path])
            except Exception as error:
                rollback_failures = restore_moved()
                for path, original in metadata_backups.items():
                    try:
                        if original is None:
                            path.unlink(missing_ok=True)
                        else:
                            _write_bytes_atomic(path, original)
                    except OSError as restore_error:
                        rollback_failures.append(f"{path}: {restore_error}")
                try:
                    if quarantine.exists():
                        shutil.rmtree(quarantine)
                except OSError as cleanup_error:
                    rollback_failures.append(str(cleanup_error))
                if rollback_failures:
                    raise RuntimeError(
                        "product evidence metadata commit failed and rollback was incomplete: "
                        + "; ".join(rollback_failures)
                    ) from error
                raise

            failed_cleanup: list[str] = []
            pending_cleanup_path: str | None = None
            if quarantine.exists():
                try:
                    shutil.rmtree(quarantine)
                except OSError as error:
                    failed_cleanup.append(str(error))
                    pending_cleanup_path = str(quarantine)
            return {
                **public,
                "pending_cleanup_path": pending_cleanup_path,
                "failed_cleanup": failed_cleanup,
            }

    def _managed_image_path(self, reference: object) -> Path | None:
        if not isinstance(reference, str) or not reference.strip():
            return None
        candidate = self.root / reference
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(self.images.resolve())
        except ValueError:
            return None
        return None if not relative.parts else resolved

    @contextmanager
    def _records_guard(self):
        with self._records_lock:
            self.records_lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.records_lock_path.open("a+b") as stream:
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                deadline = time.monotonic() + 10.0
                while True:
                    try:
                        if os.name == "nt":
                            import msvcrt

                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as error:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "timed out waiting for the feedback records lock"
                            ) from error
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _feedback_snapshot_unlocked(
        self,
        category: str,
        model_version: str,
        bundle_hash: str | None = None,
    ) -> dict[str, Any]:
        """Build a content-bound snapshot for a model scope.

        ``bundle_hash`` is optional for legacy feedback written before bundle
        binding was introduced.  When present on a record it must match the
        requested scope; a record bound to another bundle can never be used to
        activate this revision.
        """
        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        expected_bundle = (
            None if bundle_hash is None else _require_sha256(bundle_hash, "bundle_hash")
        )
        rows = self._records_unlocked(
            category=category,
            model_version=model_version,
        )
        bindings: dict[str, list[dict[str, str]]] = {
            "training": [],
            "validation": [],
        }
        seen_ids: set[str] = set()
        for row in sorted(rows, key=lambda item: str(item.get("feedback_id", ""))):
            role = str(row.get("dataset_role", "training"))
            if role not in bindings:
                raise ValueError(f"unsupported feedback dataset role in snapshot: {role}")
            feedback_id = _scope_text(
                row.get("feedback_id"), "feedback record feedback_id"
            )
            if feedback_id in seen_ids:
                raise ValueError(f"duplicate feedback_id in scoped records: {feedback_id}")
            seen_ids.add(feedback_id)
            declared_bundle = row.get("bundle_hash")
            if declared_bundle not in {None, ""}:
                declared_bundle = _require_sha256(
                    declared_bundle, f"feedback {feedback_id} bundle_hash"
                )
                if expected_bundle is not None and declared_bundle != expected_bundle:
                    raise ValueError(
                        f"feedback {feedback_id} belongs to a different model bundle"
                    )
            declared_image_hash = _require_sha256(
                row.get("image_sha256"),
                f"feedback {feedback_id} image_sha256",
            )
            image_path = self._managed_image_path(row.get("image"))
            if image_path is None or not image_path.is_file():
                raise ValueError(
                    f"feedback {feedback_id} does not reference a managed image file"
                )
            actual_image_hash = _sha256_file(image_path)
            if actual_image_hash != declared_image_hash:
                raise ValueError(
                    f"feedback image SHA-256 mismatch for {feedback_id}: "
                    f"expected {declared_image_hash}, got {actual_image_hash}"
                )
            try:
                record_hash = _record_hash(row)
            except (TypeError, ValueError) as error:
                # A stale/forged row is an evidence-snapshot failure.  Keep
                # the public error at this boundary so callers never mistake
                # a changed record for an unrelated serialization problem.
                raise ValueError(
                    f"feedback snapshot record hash mismatch for {feedback_id}"
                ) from error
            bindings[role].append(
                {
                    "feedback_id": feedback_id,
                    "image_sha256": actual_image_hash,
                    "record_sha256": record_hash,
                }
            )
        snapshot: dict[str, Any] = {
            "category": category,
            "model_version": model_version,
            "training": bindings["training"],
            "validation": bindings["validation"],
        }
        if expected_bundle is not None:
            snapshot["bundle_hash"] = expected_bundle
        snapshot["sha256"] = _canonical_sha256(snapshot)
        return snapshot

    def _candidate_receipt_path(self, candidate_id: str) -> Path:
        candidate_id = _scope_text(candidate_id, "candidate_id")
        if Path(candidate_id).name != candidate_id or candidate_id in {".", ".."}:
            raise ValueError("candidate_id must be a safe file identifier")
        unresolved = self.candidates / f"{candidate_id}.yaml"
        if unresolved.is_symlink():
            raise PermissionError("candidate receipt must not be a symbolic link")
        path = unresolved.resolve(strict=True)
        try:
            path.relative_to(self.candidates.resolve(strict=True))
        except ValueError as error:
            raise PermissionError(
                "candidate receipt path escapes the managed candidates directory"
            ) from error
        return path

    def _load_candidate_receipt(
        self, candidate: dict[str, Any]
    ) -> tuple[dict[str, Any], Path, str]:
        if not isinstance(candidate, dict):
            raise TypeError("candidate must be the proposal receipt mapping")
        candidate_id = _scope_text(candidate.get("candidate_id"), "candidate_id")
        expected_file_hash = _require_sha256(
            candidate.get("receipt_sha256"), "candidate receipt_sha256"
        )
        path = self._candidate_receipt_path(candidate_id)
        receipt_bytes = path.read_bytes()
        actual_file_hash = hashlib.sha256(receipt_bytes).hexdigest()
        if actual_file_hash != expected_file_hash:
            raise ValueError(
                "threshold candidate receipt SHA-256 mismatch: "
                f"expected {expected_file_hash}, got {actual_file_hash}"
            )
        payload = yaml.safe_load(receipt_bytes.decode("utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError("threshold candidate receipt must be a YAML mapping")
        if payload.get("candidate_id") != candidate_id:
            raise ValueError("threshold candidate receipt id does not match")
        recorded_payload_hash = _require_sha256(
            payload.get("payload_sha256"), "candidate payload_sha256"
        )
        unsigned_payload = dict(payload)
        unsigned_payload.pop("payload_sha256", None)
        actual_payload_hash = _canonical_sha256(unsigned_payload)
        if actual_payload_hash != recorded_payload_hash:
            raise ValueError(
                "threshold candidate payload SHA-256 mismatch: "
                f"expected {recorded_payload_hash}, got {actual_payload_hash}"
            )
        return payload, path, actual_file_hash

    def registry(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {
                "schema_version": 2,
                "active": {},
                "history": [],
                "legacy_unscoped": {},
            }
        payload = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"invalid threshold registry: {self.registry_path}")
        payload.setdefault("active", {})
        payload.setdefault("history", [])
        payload.setdefault("legacy_unscoped", {})
        if payload.get("schema_version", 1) < 2:
            payload["legacy_unscoped"].update(payload["active"])
            payload["active"] = {}
            payload["schema_version"] = 2
        return payload

    def active_threshold(self, category: str, model_version: str) -> float | None:
        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        entry = self.registry()["active"].get(
            _scope_key(category, model_version)
        )
        if entry is None:
            return None
        if entry.get("category") != category or entry.get("model_version") != model_version:
            raise ValueError("active threshold scope does not match registry lookup")
        threshold = _finite_float(entry.get("threshold"), "active threshold")
        receipt = entry.get("validation_receipt")
        if not isinstance(receipt, dict):
            raise ValueError("active threshold has no validation receipt")
        payload, path, receipt_sha256 = self._load_candidate_receipt(
            {
                "candidate_id": entry.get("candidate_id"),
                "receipt_sha256": receipt.get("sha256"),
            }
        )
        if receipt.get("path") != str(path):
            raise ValueError("active threshold validation receipt path does not match")
        if receipt.get("sha256") != receipt_sha256:
            raise ValueError("active threshold validation receipt hash does not match")
        if payload.get("category") != category or payload.get("model_version") != model_version:
            raise ValueError("active threshold validation receipt scope does not match")
        validation = payload.get("validation")
        if (
            int(payload.get("schema_version", 0)) < 2
            or not isinstance(validation, dict)
            or validation.get("passed") is not True
        ):
            raise ValueError("active threshold receipt did not pass validation")
        if int(entry.get("sample_count", -1)) != int(payload.get("sample_count", -2)):
            raise ValueError("active threshold sample count does not match receipt")
        persisted_snapshot = payload.get("feedback_snapshot")
        if (
            not isinstance(persisted_snapshot, dict)
            or entry.get("feedback_snapshot_sha256")
            != persisted_snapshot.get("sha256")
        ):
            raise ValueError("active threshold feedback snapshot hash does not match")
        persisted_threshold = _finite_float(
            payload.get("candidate_threshold"), "persisted candidate threshold"
        )
        if threshold != persisted_threshold:
            raise ValueError("active threshold does not match validation receipt")
        return threshold

    def propose_threshold(
        self,
        category: str,
        model_version: str,
        *,
        base_threshold: float,
        minimum_samples: int = 6,
        minimum_recall: float = 0.90,
    ) -> dict[str, Any]:
        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        base_threshold = _finite_float(base_threshold, "base_threshold")
        minimum_recall = _finite_float(minimum_recall, "minimum_recall")
        if not 0.0 <= minimum_recall <= 1.0:
            raise ValueError("minimum_recall must be between 0 and 1")
        if isinstance(minimum_samples, bool) or int(minimum_samples) < 1:
            raise ValueError("minimum_samples must be a positive integer")
        minimum_samples = int(minimum_samples)

        with self._records_guard():
            rows = self._records_unlocked(category, "training", model_version)
            if len(rows) < minimum_samples:
                raise ValueError(
                    f"at least {minimum_samples} feedback samples are required; "
                    f"found {len(rows)}"
                )
            labels = np.asarray(
                [row["operator_label"] for row in rows], dtype=np.uint8
            )
            if set(labels.tolist()) != {0, 1}:
                raise ValueError("feedback must contain both normal and defect labels")
            scores = np.asarray(
                [row["fusion_score"] for row in rows], dtype=np.float64
            )
            if not np.all(np.isfinite(scores)):
                raise ValueError("training feedback scores must be finite")
            training_hashes = [
                _require_sha256(
                    row.get("image_sha256"), "training feedback image_sha256"
                )
                for row in rows
            ]
            if len(set(training_hashes)) != len(training_hashes):
                raise ValueError(
                    "training feedback contains duplicate image content for this model"
                )
            unique = np.unique(scores)
            raw_candidates = np.concatenate(
                (
                    [np.nextafter(unique[0], -np.inf)],
                    (unique[:-1] + unique[1:]) / 2.0,
                    [unique[-1], np.nextafter(unique[-1], np.inf)],
                )
            )
            evaluated = []
            for threshold_value in raw_candidates:
                if not np.isfinite(threshold_value):
                    continue
                threshold_value = float(threshold_value)
                threshold_metrics = classification_metrics(
                    labels, scores, threshold_value
                )
                evaluated.append((threshold_value, threshold_metrics))
            feasible = [
                item
                for item in evaluated
                if float(item[1]["recall"]) >= minimum_recall
            ]
            if not feasible:
                raise ValueError("no threshold satisfies the minimum recall constraint")
            threshold, metrics = min(
                feasible,
                key=lambda item: (
                    float(item[1]["fpr"]),
                    -float(item[1]["f1"]),
                    -item[0],
                ),
            )
            validation_rows = self._records_unlocked(
                category, "validation", model_version
            )
            validation_hashes = [
                _require_sha256(
                    row.get("image_sha256"), "validation feedback image_sha256"
                )
                for row in validation_rows
            ]
            if len(set(validation_hashes)) != len(validation_hashes):
                raise ValueError(
                    "validation feedback contains duplicate image content for this model"
                )
            if set(training_hashes) & set(validation_hashes):
                raise PermissionError(
                    "FEEDBACK_TRAIN_VALIDATION_LEAKAGE: identical image content is "
                    "present in training and validation feedback"
                )
            validation_labels = np.asarray(
                [row["operator_label"] for row in validation_rows], dtype=np.uint8
            )
            validation_ids = [
                str(row.get("feedback_id")) for row in validation_rows
            ]
            validation: dict[str, Any] = {
                "category": category,
                "model_version": model_version,
                "sample_ids": validation_ids,
                "sample_sha256": validation_hashes,
                "sample_count": len(validation_rows),
                "normal_count": int(np.sum(validation_labels == 0)),
                "defect_count": int(np.sum(validation_labels == 1)),
                "minimum_recall": minimum_recall,
            }
            if (
                len(validation_rows) < 4
                or set(validation_labels.tolist()) != {0, 1}
            ):
                validation.update(
                    {
                        "passed": False,
                        "reason": "at least 4 independent validation samples "
                        "containing normal and defect labels are required",
                    }
                )
            else:
                validation_scores = np.asarray(
                    [row["fusion_score"] for row in validation_rows],
                    dtype=np.float64,
                )
                base_validation = classification_metrics(
                    validation_labels, validation_scores, base_threshold
                )
                candidate_validation = classification_metrics(
                    validation_labels, validation_scores, threshold
                )
                passed = bool(
                    candidate_validation["recall"] >= minimum_recall
                    and candidate_validation["f1"] >= base_validation["f1"]
                )
                validation.update(
                    {
                        "passed": passed,
                        "reason": (
                            "passed"
                            if passed
                            else "candidate must preserve F1 and satisfy the "
                            "recall constraint"
                        ),
                        "base_metrics": base_validation,
                        "candidate_metrics": candidate_validation,
                    }
                )
            candidate_id = (
                f"threshold-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
                f"{uuid4().hex}"
            )
            payload: dict[str, Any] = {
                "schema_version": 2,
                "candidate_id": candidate_id,
                "created_at": _now(),
                "category": category,
                "model_version": model_version,
                "method": "operator_feedback_constrained_threshold_search",
                "selection_scope": "operator_confirmed_feedback",
                "sample_count": len(rows),
                "normal_count": int(np.sum(labels == 0)),
                "defect_count": int(np.sum(labels == 1)),
                "minimum_recall": minimum_recall,
                "base_threshold": base_threshold,
                "candidate_threshold": threshold,
                "base_metrics": classification_metrics(
                    labels, scores, base_threshold
                ),
                "candidate_metrics": metrics,
                "validation": validation,
                "feedback_snapshot": self._feedback_snapshot_unlocked(
                    category, model_version
                ),
                "warning": "Training metrics are apparent results. Activation "
                "requires this persisted independent-validation receipt.",
            }
            payload["payload_sha256"] = _canonical_sha256(payload)
            path = self.candidates / f"{candidate_id}.yaml"
            _write_yaml_atomic(path, payload)
            receipt_sha256 = _sha256_file(path)

        result = dict(payload)
        result["path"] = str(path.resolve())
        result["receipt_sha256"] = receipt_sha256
        return result

    def _verify_threshold_receipt_unlocked(
        self, payload: dict[str, Any]
    ) -> tuple[str, str, float, dict[str, Any]]:
        if int(payload.get("schema_version", 0)) < 2:
            raise ValueError("threshold candidate receipt schema is not activatable")
        category = _scope_text(payload.get("category"), "candidate.category")
        model_version = _scope_text(
            payload.get("model_version"), "candidate.model_version"
        )
        threshold = _finite_float(
            payload.get("candidate_threshold"), "candidate_threshold"
        )
        base_threshold = _finite_float(
            payload.get("base_threshold"), "base_threshold"
        )
        minimum_recall = _finite_float(
            payload.get("minimum_recall"), "minimum_recall"
        )
        if not 0.0 <= minimum_recall <= 1.0:
            raise ValueError("candidate minimum_recall must be between 0 and 1")
        training_rows = self._records_unlocked(category, "training", model_version)
        validation_rows = self._records_unlocked(
            category, "validation", model_version
        )
        snapshot = payload.get("feedback_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("threshold candidate is missing feedback_snapshot")
        current_snapshot = self._feedback_snapshot_unlocked(category, model_version)
        if snapshot != current_snapshot:
            raise ValueError(
                "threshold candidate feedback snapshot does not match current records"
            )

        if int(payload.get("sample_count", -1)) != len(training_rows):
            raise ValueError("threshold candidate training sample count does not match")
        labels = np.asarray(
            [row["operator_label"] for row in training_rows], dtype=np.uint8
        )
        scores = np.asarray(
            [row["fusion_score"] for row in training_rows], dtype=np.float64
        )
        if set(labels.tolist()) != {0, 1} or not np.all(np.isfinite(scores)):
            raise ValueError("threshold candidate training records are invalid")
        training_hashes = [
            _require_sha256(row.get("image_sha256"), "training feedback image_sha256")
            for row in training_rows
        ]
        validation_hashes = [
            _require_sha256(
                row.get("image_sha256"), "validation feedback image_sha256"
            )
            for row in validation_rows
        ]
        if len(set(training_hashes)) != len(training_hashes):
            raise ValueError("training feedback contains duplicate image content")
        if len(set(validation_hashes)) != len(validation_hashes):
            raise ValueError("validation feedback contains duplicate image content")
        if set(training_hashes) & set(validation_hashes):
            raise PermissionError(
                "FEEDBACK_TRAIN_VALIDATION_LEAKAGE: identical image content is "
                "present in training and validation feedback"
            )

        expected_base_metrics = classification_metrics(
            labels, scores, base_threshold
        )
        expected_candidate_metrics = classification_metrics(
            labels, scores, threshold
        )
        if payload.get("base_metrics") != expected_base_metrics:
            raise ValueError("threshold candidate base metrics do not match records")
        if payload.get("candidate_metrics") != expected_candidate_metrics:
            raise ValueError(
                "threshold candidate metrics do not match current training records"
            )

        validation = payload.get("validation")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            reason = validation.get("reason", "unknown") if isinstance(validation, dict) else "missing"
            raise ValueError(f"candidate failed independent validation: {reason}")
        if validation.get("category") != category:
            raise ValueError("candidate validation category does not match")
        if validation.get("model_version") != model_version:
            raise ValueError("candidate validation model version does not match")
        if validation.get("sample_ids") != [
            str(row.get("feedback_id")) for row in validation_rows
        ]:
            raise ValueError("candidate validation sample IDs do not match records")
        if validation.get("sample_sha256") != validation_hashes:
            raise ValueError("candidate validation image hashes do not match records")
        if len(validation_rows) < 4:
            raise ValueError("candidate validation set is too small")
        validation_labels = np.asarray(
            [row["operator_label"] for row in validation_rows], dtype=np.uint8
        )
        validation_scores = np.asarray(
            [row["fusion_score"] for row in validation_rows], dtype=np.float64
        )
        if set(validation_labels.tolist()) != {0, 1} or not np.all(
            np.isfinite(validation_scores)
        ):
            raise ValueError("candidate validation records are invalid")
        if int(validation.get("sample_count", -1)) != len(validation_rows):
            raise ValueError("candidate validation sample count does not match")
        if int(validation.get("normal_count", -1)) != int(
            np.sum(validation_labels == 0)
        ):
            raise ValueError("candidate validation normal count does not match")
        if int(validation.get("defect_count", -1)) != int(
            np.sum(validation_labels == 1)
        ):
            raise ValueError("candidate validation defect count does not match")
        if _finite_float(validation.get("minimum_recall"), "validation.minimum_recall") != minimum_recall:
            raise ValueError("candidate validation minimum_recall does not match")
        expected_base_validation = classification_metrics(
            validation_labels, validation_scores, base_threshold
        )
        expected_candidate_validation = classification_metrics(
            validation_labels, validation_scores, threshold
        )
        if validation.get("base_metrics") != expected_base_validation:
            raise ValueError("candidate validation base metrics do not match records")
        if validation.get("candidate_metrics") != expected_candidate_validation:
            raise ValueError(
                "candidate validation metrics do not match current records"
            )
        if not (
            expected_candidate_validation["recall"] >= minimum_recall
            and expected_candidate_validation["f1"]
            >= expected_base_validation["f1"]
        ):
            raise ValueError("candidate failed independent validation constraints")
        _scope_text(payload.get("candidate_id"), "candidate_id")
        return category, model_version, threshold, snapshot

    def activate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        # The caller's validation mapping is deliberately ignored.  Only the
        # persisted receipt, addressed by candidate_id and its file hash, can
        # authorize activation.
        payload, receipt_path, receipt_sha256 = self._load_candidate_receipt(candidate)
        if "category" in candidate and candidate.get("category") != payload.get("category"):
            raise ValueError("candidate category does not match persisted receipt")
        if "model_version" in candidate and candidate.get("model_version") != payload.get("model_version"):
            raise ValueError("candidate model version does not match persisted receipt")
        with self._records_guard():
            category, model_version, threshold, snapshot = self._verify_threshold_receipt_unlocked(
                payload
            )
            registry = self.registry()
            scope = _scope_key(category, model_version)
            previous = registry["active"].get(scope)
            entry = {
                "candidate_id": str(payload["candidate_id"]),
                "threshold": threshold,
                "activated_at": _now(),
                "sample_count": int(payload["sample_count"]),
                "category": category,
                "model_version": model_version,
                "validation_receipt": {
                    "path": str(receipt_path),
                    "sha256": receipt_sha256,
                },
                "feedback_snapshot_sha256": snapshot["sha256"],
                "previous": previous,
            }
            registry["active"][scope] = entry
            registry["history"].append(entry)
            _write_yaml_atomic(self.registry_path, registry)
        return entry

    def rollback(
        self, category: str, model_version: str
    ) -> dict[str, Any] | None:
        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        with self._records_guard():
            registry = self.registry()
            scope = _scope_key(category, model_version)
            current = registry["active"].get(scope)
            if current is None:
                return None
            previous = current.get("previous")
            if previous is None:
                registry["active"].pop(scope, None)
            else:
                registry["active"][scope] = previous
            registry["history"].append({
                "category": category,
                "model_version": model_version,
                "action": "rollback",
                "rolled_back_at": _now(),
                "from_candidate_id": current["candidate_id"],
            })
            _write_yaml_atomic(self.registry_path, registry)
        return previous

    def feature_registry(self) -> dict[str, Any]:
        if not self.feature_registry_path.is_file():
            return {
                "schema_version": 2,
                "active": {},
                "history": [],
                "legacy_unscoped": {},
            }
        payload = yaml.safe_load(
            self.feature_registry_path.read_text(encoding="utf-8")
        ) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"invalid feature registry: {self.feature_registry_path}")
        payload.setdefault("active", {})
        payload.setdefault("history", [])
        payload.setdefault("legacy_unscoped", {})
        if payload.get("schema_version", 1) < 2:
            payload["legacy_unscoped"].update(payload["active"])
            payload["active"] = {}
            payload["schema_version"] = 2
        return payload

    def save_feature_bank(
        self,
        category: str,
        model_version: str,
        embeddings: np.ndarray,
        sample_ids: list[str],
        *,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or len(values) != len(sample_ids) or not len(values):
            raise ValueError("feature bank requires a non-empty rank-2 embedding matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError("feature bank embeddings must be finite")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("feature bank sample_ids must be unique")
        validation = self._validate_feature_bank_receipt(
            category,
            model_version,
            sample_ids,
            validation,
        )
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("feature bank embeddings must have finite non-zero norms")
        values = values / np.maximum(norms, 1e-12)
        version = (
            f"features-{category}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
            f"{uuid4().hex[:8]}"
        )
        path = self.feature_banks / f"{version}.npz"
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, embeddings=values, sample_ids=np.asarray(sample_ids)
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        file_sha256 = _sha256_file(path)
        entry = {
            "schema_version": 3,
            "version": version,
            "category": category,
            "model_version": model_version,
            "created_at": _now(),
            "path": path.relative_to(self.root).as_posix(),
            "sample_count": len(sample_ids),
            "embedding_dimension": int(values.shape[1]),
            "sample_ids": sample_ids,
            "file_sha256": file_sha256,
            "validation": validation,
        }
        with self._records_guard():
            try:
                registry = self.feature_registry()
                scope = _scope_key(category, model_version)
                previous = registry["active"].get(scope)
                entry["previous"] = previous
                registry["active"][scope] = entry
                registry["history"].append(entry)
                _write_yaml_atomic(self.feature_registry_path, registry)
            except Exception:
                # The feature file is not active until its registry pointer is
                # durable.  Remove it if pointer persistence fails so a failed
                # save cannot leave an orphan artifact that looks usable.
                path.unlink(missing_ok=True)
                raise
        return entry

    def _validate_feature_bank_receipt(
        self,
        category: str,
        model_version: str,
        sample_ids: list[str],
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(validation, dict) or not validation.get("passed"):
            raise ValueError(
                "feature bank activation requires passed independent validation"
            )
        if validation.get("category") != category:
            raise ValueError("feature-bank validation category does not match")
        if validation.get("model_version") != model_version:
            raise ValueError("feature-bank validation model version does not match")

        training_rows = self.records(category, "training", model_version)
        training_by_id = {
            str(row.get("feedback_id")): row for row in training_rows
        }
        missing = sorted(set(sample_ids) - set(training_by_id))
        if missing:
            raise ValueError(
                "feature bank sample_ids are not scoped training feedback: "
                + ", ".join(missing)
            )
        for sample_id in sample_ids:
            row = training_by_id[sample_id]
            if int(row.get("operator_label", 0)) != 1 or row.get("defect_region") is None:
                raise ValueError(
                    "feature bank samples must be defect feedback with an annotated region"
                )

        validation_rows = self.records(category, "validation", model_version)
        labels = [int(row.get("operator_label", -1)) for row in validation_rows]
        if len(validation_rows) < 4 or set(labels) != {0, 1}:
            raise ValueError(
                "feature bank requires at least 4 scoped validation samples "
                "containing normal and defect labels"
            )
        validation_ids = [str(row.get("feedback_id")) for row in validation_rows]
        if set(validation.get("sample_ids", [])) != set(validation_ids):
            raise ValueError(
                "feature-bank validation receipt does not cover the current scoped validation set"
            )
        validation_hashes = [
            str(row.get("image_sha256", "")) for row in validation_rows
        ]
        training_hashes = [
            str(row.get("image_sha256", "")) for row in training_rows
        ]
        if (
            "" in validation_hashes
            or len(set(validation_hashes)) != len(validation_hashes)
        ):
            raise ValueError("validation feedback hashes must be present and unique")
        if set(training_hashes) & set(validation_hashes):
            raise PermissionError(
                "FEEDBACK_TRAIN_VALIDATION_LEAKAGE: identical image content is "
                "present in training and validation feedback"
            )
        if int(validation.get("sample_count", -1)) != len(validation_rows):
            raise ValueError("feature-bank validation sample count does not match")
        if int(validation.get("normal_count", -1)) != labels.count(0):
            raise ValueError("feature-bank validation normal count does not match")
        if int(validation.get("defect_count", -1)) != labels.count(1):
            raise ValueError("feature-bank validation defect count does not match")

        base_metrics = validation.get("base_metrics")
        candidate_metrics = validation.get("candidate_metrics")
        if not isinstance(base_metrics, dict) or not isinstance(candidate_metrics, dict):
            raise ValueError("feature-bank validation metrics are required")
        minimum_recall = _finite_float(
            validation.get("minimum_recall", 0.90),
            "feature-bank validation minimum_recall",
        )
        if not 0.0 <= minimum_recall <= 1.0:
            raise ValueError(
                "feature-bank validation minimum_recall must be between 0 and 1"
            )
        candidate_recall = _finite_float(
            candidate_metrics.get("recall"),
            "feature-bank candidate validation recall",
        )
        candidate_f1 = _finite_float(
            candidate_metrics.get("f1"),
            "feature-bank candidate validation f1",
        )
        base_f1 = _finite_float(
            base_metrics.get("f1"), "feature-bank base validation f1"
        )
        if candidate_recall < minimum_recall or candidate_f1 < base_f1:
            raise ValueError(
                "feature-bank candidate must preserve validation F1 and satisfy recall"
            )
        return dict(validation)

    def active_feature_bank(
        self, category: str, model_version: str
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        entry = self.feature_registry()["active"].get(
            _scope_key(category, model_version)
        )
        if entry is None:
            return None
        if entry.get("category") != category or entry.get("model_version") != model_version:
            raise ValueError("active feature bank scope does not match registry lookup")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("active feature bank has no relative path")
        path = self._safe_revision_path(raw_path, must_exist=True)
        try:
            path.relative_to(self.feature_banks.resolve(strict=True))
        except ValueError as error:
            raise PermissionError(
                "active feature bank path escapes the managed feature_banks directory"
            ) from error
        expected_file_sha256 = _require_sha256(
            entry.get("file_sha256"), "active feature bank file_sha256"
        )
        feature_bytes = path.read_bytes()
        actual_file_sha256 = hashlib.sha256(feature_bytes).hexdigest()
        if actual_file_sha256 != expected_file_sha256:
            raise ValueError(
                "active feature bank file SHA-256 mismatch: "
                f"expected {expected_file_sha256}, got {actual_file_sha256}"
            )
        with np.load(io.BytesIO(feature_bytes), allow_pickle=False) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            stored_ids = payload.get("sample_ids")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("active feature bank embeddings must be finite")
        if embeddings.ndim != 2 or len(embeddings) != int(entry["sample_count"]):
            raise ValueError("active feature bank shape does not match its registry entry")
        if embeddings.shape[1] != int(entry["embedding_dimension"]):
            raise ValueError(
                "active feature bank dimension does not match its registry entry"
            )
        if stored_ids is None:
            raise ValueError("active feature bank is missing sample_ids")
        normalized_ids = [str(value) for value in stored_ids.tolist()]
        expected_ids = [str(value) for value in entry.get("sample_ids", [])]
        if normalized_ids != expected_ids:
            raise ValueError("active feature bank sample_ids do not match its registry entry")
        return embeddings, entry

    # ------------------------------------------------------------------
    # Unified adaptive revision registry
    # ------------------------------------------------------------------

    @staticmethod
    def revision_scope_key(category: str, run_id: str, bundle_hash: str) -> str:
        """Return the stable scope key used by the revision registry."""
        category = _scope_text(category, "category")
        run_id = _scope_text(run_id, "run_id")
        bundle_hash = _require_sha256(bundle_hash, "bundle_hash")
        return _canonical_sha256({"category": category, "run_id": run_id, "bundle_hash": bundle_hash})

    def _revision_registry(self) -> dict[str, Any]:
        if not self.revision_registry_path.is_file():
            return {"schema_version": 1, "active": {}, "history": []}
        payload = yaml.safe_load(
            self.revision_registry_path.read_text(encoding="utf-8")
        ) or {}
        if not isinstance(payload, dict):
            raise TypeError("revision registry must be a YAML mapping")
        payload.setdefault("schema_version", 1)
        payload.setdefault("active", {})
        payload.setdefault("history", [])
        if not isinstance(payload["active"], dict) or not isinstance(payload["history"], list):
            raise TypeError("revision registry active/history fields are invalid")
        return payload

    def _safe_revision_path(self, raw: object, *, must_exist: bool = True) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("revision artifact path must be a non-empty relative path")
        value = raw.replace("\\", "/")
        candidate = Path(value)
        if (
            "\x00" in value
            or candidate.is_absolute()
            or ".." in candidate.parts
            or any(part in {"", "."} for part in candidate.parts)
        ):
            raise PermissionError("revision artifact path must be relative and contained")
        unresolved = self.root.joinpath(*candidate.parts)
        # Reject symlinks in every path component, including a symlinked
        # directory.  Resolving first and checking containment alone would
        # permit a link that happens to point back inside the store.
        cursor = self.root
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PermissionError(
                    "revision artifact path must not contain a symbolic link"
                )
        resolved = unresolved.resolve(strict=must_exist)
        try:
            resolved.relative_to(self.root.resolve(strict=True))
        except ValueError as error:
            raise PermissionError("revision artifact path escapes feedback root") from error
        return resolved

    def _revision_artifact_bytes(
        self,
        value: object,
        *,
        context: str,
    ) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"{context} artifact does not exist: {path}")
            return path.read_bytes()
        raise TypeError(f"{context} artifact must be bytes or a file path")

    @staticmethod
    def _normalise_feedback_bindings(value: object) -> dict[str, list[dict[str, str]]]:
        if value is None:
            return {"training": [], "validation": []}
        if not isinstance(value, Mapping):
            raise TypeError("feedback_snapshot must be a mapping")
        result: dict[str, list[dict[str, str]]] = {"training": [], "validation": []}
        seen: set[str] = set()
        for role in result:
            rows = value.get(role, [])
            if not isinstance(rows, (list, tuple)):
                raise TypeError(f"feedback_snapshot.{role} must be a list")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError("feedback snapshot rows must be mappings")
                feedback_id = _scope_text(str(row.get("feedback_id", "")), "feedback_id")
                if feedback_id in seen:
                    raise ValueError(f"duplicate feedback_id in revision snapshot: {feedback_id}")
                seen.add(feedback_id)
                image_hash = _require_sha256(row.get("image_sha256"), f"{role} image_sha256")
                record_hash = _require_sha256(row.get("record_sha256"), f"{role} record_sha256")
                result[role].append(
                    {
                        "feedback_id": feedback_id,
                        "image_sha256": image_hash,
                        "record_sha256": record_hash,
                    }
                )
        return result

    def _write_binary_atomic(self, path: Path, value: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise PermissionError(f"refusing to replace symbolic-link artifact: {path}")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_revision_receipt(
        self,
        candidate: Mapping[str, Any] | str,
    ) -> tuple[dict[str, Any], Path, str]:
        if isinstance(candidate, str):
            revision_id = _scope_text(candidate, "revision_id")
            raw_path = f"candidates/{revision_id}.yaml"
            expected_receipt_hash = None
        elif isinstance(candidate, Mapping):
            revision_id = _scope_text(
                candidate.get("revision_id", candidate.get("candidate_id")),
                "revision_id",
            )
            raw_path = candidate.get("receipt_path", candidate.get("path"))
            expected_receipt_hash = candidate.get("receipt_sha256")
            if raw_path is None:
                raw_path = f"candidates/{revision_id}.yaml"
        else:
            raise TypeError("revision candidate must be a mapping or id")
        path = self._safe_revision_path(raw_path, must_exist=True)
        # Candidate IDs are never interpreted as filesystem paths.  The
        # persisted receipt's ID is checked after parsing as an additional
        # traversal/TOCTOU guard.
        receipt_bytes = path.read_bytes()
        actual_receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
        if expected_receipt_hash is not None and _require_sha256(
            expected_receipt_hash, "revision receipt_sha256"
        ) != actual_receipt_hash:
            raise ValueError("revision receipt SHA-256 mismatch")
        payload = yaml.safe_load(receipt_bytes.decode("utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError("revision receipt must be a YAML mapping")
        if payload.get("revision_id") != revision_id:
            raise ValueError("revision receipt id does not match")
        recorded_payload_hash = _require_sha256(
            payload.get("payload_sha256"), "revision payload_sha256"
        )
        unsigned = dict(payload)
        unsigned.pop("payload_sha256", None)
        if _canonical_sha256(unsigned) != recorded_payload_hash:
            raise ValueError("revision receipt payload SHA-256 mismatch")
        return payload, path, actual_receipt_hash

    def _verify_revision_artifacts(self, payload: Mapping[str, Any]) -> None:
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("revision receipt artifacts are missing")
        revision_id = _scope_text(payload.get("revision_id"), "revision_id")
        for name in ("head", "feature_bank"):
            entry = artifacts.get(name)
            if entry is None:
                # A candidate may explicitly inherit an empty bank, but the
                # head is always required for a complete revision.
                if name == "head":
                    raise ValueError("revision receipt head artifact is missing")
                continue
            if not isinstance(entry, Mapping):
                raise TypeError(f"revision artifact {name} must be a mapping")
            path = self._safe_revision_path(entry.get("path"), must_exist=True)
            # Artifacts are immutable members of their own revision directory;
            # allowing a receipt to point at an arbitrary sibling would make a
            # forged candidate able to borrow another revision's state.
            expected_parent = (self.revisions / revision_id).resolve(strict=True)
            try:
                path.relative_to(expected_parent)
            except ValueError as error:
                raise PermissionError(
                    f"revision {name} artifact is outside its revision directory"
                ) from error
            expected = _require_sha256(entry.get("sha256"), f"revision {name} sha256")
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(f"revision {name} artifact SHA-256 mismatch")
            declared_bytes = entry.get("bytes")
            if declared_bytes is not None and (
                isinstance(declared_bytes, bool)
                or not isinstance(declared_bytes, int)
                or declared_bytes != path.stat().st_size
            ):
                raise ValueError(f"revision {name} artifact byte count does not match")
            if name == "head":
                # Complete feedback revisions carry a narrow output-layer
                # overlay.  A serialised full head would make it possible to
                # replace frozen feature normalisation or hidden-layer tensors.
                try:
                    overlay = torch.load(path, map_location="cpu", weights_only=True)
                except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
                    raise ValueError("revision head overlay is malformed") from error
                if not isinstance(overlay, Mapping):
                    raise ValueError("revision head overlay must be a mapping")
                if overlay.get("format_version") != 1 or overlay.get("artifact_type") != (
                    "dinov2_tiled_mlp_feedback_output_layer_overlay"
                ):
                    raise ValueError("revision head artifact is not an output-layer overlay")
                if overlay.get("trainable_parameters") != ["weight2", "bias2"]:
                    raise ValueError("revision head overlay trainable parameter set is invalid")
                if int(overlay.get("updated_parameter_count", -1)) != 33:
                    raise ValueError("revision head overlay must update exactly 33 parameters")
                state = overlay.get("output_layer")
                frozen = overlay.get("parent_frozen_fingerprint")
                parent = overlay.get("parent_parameter_fingerprint")
                candidate = overlay.get("candidate_parameter_fingerprint")
                if (
                    not isinstance(state, Mapping)
                    or set(state) != {"weight2", "bias2"}
                    or not isinstance(frozen, Mapping)
                    or not isinstance(parent, Mapping)
                    or not isinstance(candidate, Mapping)
                ):
                    raise ValueError("revision head overlay content is invalid")
                weight_count = int(torch.as_tensor(state["weight2"]).numel())
                bias_count = int(torch.as_tensor(state["bias2"]).numel())
                if weight_count + bias_count != 33 or bias_count != 1:
                    raise ValueError("revision head overlay does not contain exactly 33 parameters")
                output_hashes = {
                    "weight2": _tensor_sha256(state["weight2"], "revision weight2"),
                    "bias2": _tensor_sha256(state["bias2"], "revision bias2"),
                }
                frozen_names = ("feature_mean", "feature_scale", "weight1", "bias1")
                frozen_hashes: dict[str, str] = {}
                for frozen_name in frozen_names:
                    frozen_hash = _require_sha256(
                        frozen.get(frozen_name), f"revision frozen tensor {frozen_name}"
                    )
                    if parent.get(frozen_name) != frozen_hash or candidate.get(frozen_name) != frozen_hash:
                        raise ValueError(
                            f"revision head overlay frozen fingerprint mismatch: {frozen_name}"
                        )
                    frozen_hashes[frozen_name] = frozen_hash
                for output_name, output_hash in output_hashes.items():
                    if candidate.get(output_name) != output_hash:
                        raise ValueError(
                            f"revision head overlay candidate fingerprint mismatch: {output_name}"
                        )
                    _require_sha256(parent.get(output_name), f"revision parent {output_name}")
                if (
                    parent.get("weight2") == candidate.get("weight2")
                    and parent.get("bias2") == candidate.get("bias2")
                ):
                    raise ValueError("revision head overlay did not change an output parameter")
                ordered_hashes = {
                    **frozen_hashes,
                    "weight2": output_hashes["weight2"],
                    "bias2": output_hashes["bias2"],
                }
                aggregate = "|".join(
                    f"{key}:{ordered_hashes[key]}"
                    for key in ("feature_mean", "feature_scale", "weight1", "bias1", "weight2", "bias2")
                )
                if candidate.get("overall_sha256") != hashlib.sha256(aggregate.encode("ascii")).hexdigest():
                    raise ValueError("revision head overlay aggregate fingerprint mismatch")
            if name == "feature_bank":
                try:
                    with np.load(path, allow_pickle=False) as loaded:
                        if "embeddings" not in loaded or "sample_ids" not in loaded:
                            raise ValueError("feature bank artifact must contain embeddings and sample_ids")
                        values = np.asarray(loaded["embeddings"], dtype=np.float32)
                        ids = [str(value) for value in loaded["sample_ids"].tolist()]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise ValueError("revision feature bank artifact is malformed") from error
                if (
                    values.ndim != 2
                    or len(values) == 0
                    or len(values) != len(ids)
                    or len(set(ids)) != len(ids)
                    or not np.all(np.isfinite(values))
                ):
                    raise ValueError("revision feature bank artifact is malformed")
                if np.any(np.linalg.norm(values, axis=1) <= 0):
                    raise ValueError("revision feature bank vectors must have non-zero norms")
                if entry.get("sample_count") is not None and int(entry["sample_count"]) != len(values):
                    raise ValueError("revision feature bank sample count does not match artifact")
                if entry.get("embedding_dimension") is not None and int(entry["embedding_dimension"]) != int(values.shape[1]):
                    raise ValueError("revision feature bank dimension does not match artifact")
                if entry.get("sample_ids") is not None and [str(v) for v in entry["sample_ids"]] != ids:
                    raise ValueError("revision feature bank sample IDs do not match artifact")

    def _current_revision_snapshot_unlocked(
        self, category: str, run_id: str, bundle_hash: str | None = None
    ) -> dict[str, Any]:
        return self._feedback_snapshot_unlocked(category, run_id, bundle_hash)

    def save_revision_candidate(
        self,
        candidate: Mapping[str, Any] | None = None,
        *,
        category: str | None = None,
        run_id: str | None = None,
        model_version: str | None = None,
        bundle_hash: str | None = None,
        parent_revision_id: str | None = None,
        head_artifact: bytes | bytearray | str | Path | None = None,
        feature_embeddings: np.ndarray | None = None,
        feature_sample_ids: list[str] | tuple[str, ...] | None = None,
        feature_artifact: bytes | bytearray | str | Path | None = None,
        threshold: float | None = None,
        validation: Mapping[str, Any] | None = None,
        feedback_snapshot: Mapping[str, Any] | None = None,
        training: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist an immutable, non-active complete revision candidate.

        The method accepts both a prepared mapping and explicit artifact
        arguments to keep the service convenient for the CLI and lightweight
        test/fake runtimes.  No registry active pointer is changed here.
        """
        source = dict(candidate or {})
        category = _scope_text(category or source.get("category"), "category")
        run_id = _scope_text(
            run_id or model_version or source.get("run_id", source.get("model_version")),
            "run_id",
        )
        bundle_hash = _require_sha256(
            bundle_hash or source.get("bundle_hash"), "bundle_hash"
        )
        # A caller may pass explicit scope arguments together with a prepared
        # payload.  Never silently accept a conflicting payload scope.
        for key, expected in (
            ("category", category),
            ("run_id", run_id),
            ("bundle_hash", bundle_hash),
        ):
            declared = source.get(key)
            if declared is not None and str(declared) != str(expected):
                raise ValueError(f"revision {key} does not match requested scope")
        revision_id = _scope_text(
            source.get("revision_id", source.get("candidate_id", f"revision-{uuid4().hex}")),
            "revision_id",
        )
        if Path(revision_id).name != revision_id or revision_id in {".", ".."}:
            raise ValueError("revision_id must be a safe identifier")
        threshold_value = threshold
        if threshold_value is None:
            threshold_value = source.get("threshold", source.get("candidate_threshold"))
        threshold_value = _finite_float(threshold_value, "revision threshold")
        parent_revision_id = parent_revision_id or source.get("parent_revision_id")
        if parent_revision_id is not None:
            parent_revision_id = _scope_text(parent_revision_id, "parent_revision_id")

        if head_artifact is None:
            head_artifact = source.get("head_artifact", source.get("head_bytes"))
        if head_artifact is None:
            head_path = source.get("head_path")
            if head_path is not None:
                head_artifact = head_path
        head_bytes = self._revision_artifact_bytes(head_artifact, context="head")
        if head_bytes is None:
            # A mapping containing output tensors is serialized in a portable
            # torch container.  This is intentionally local to the candidate;
            # the original bundle/head file is never opened for writing.
            overlay = source.get("head_overlay")
            if overlay is not None:
                buffer = io.BytesIO()
                try:
                    import torch

                    torch.save(dict(overlay), buffer)
                except Exception as error:
                    raise TypeError("head_overlay cannot be serialized") from error
                head_bytes = buffer.getvalue()
        if head_bytes is None:
            raise ValueError("complete revision candidate requires a head artifact")

        if feature_artifact is None:
            feature_artifact = source.get("feature_artifact", source.get("feature_bank_path"))
        feature_bytes = self._revision_artifact_bytes(feature_artifact, context="feature bank")
        if feature_bytes is None:
            if feature_embeddings is None:
                feature_embeddings = source.get("feature_embeddings")
            if feature_sample_ids is None:
                feature_sample_ids = source.get("feature_sample_ids", source.get("sample_ids"))
            if feature_embeddings is not None:
                values = np.asarray(feature_embeddings, dtype=np.float32)
                ids = [str(value) for value in (feature_sample_ids or [])]
                if values.ndim != 2 or len(values) == 0 or len(values) != len(ids):
                    raise ValueError("feature bank requires rank-2 values and matching sample IDs")
                if len(set(ids)) != len(ids):
                    raise ValueError("feature bank sample IDs must be unique")
                if not np.all(np.isfinite(values)):
                    raise ValueError("feature bank values must be finite")
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
                    raise ValueError("feature bank vectors must have non-zero finite norms")
                values = values / np.maximum(norms, 1e-12)
                buffer = io.BytesIO()
                np.savez_compressed(buffer, embeddings=values, sample_ids=np.asarray(ids))
                feature_bytes = buffer.getvalue()
        if feature_bytes is None:
            # A complete adaptive revision always carries all three mutable
            # components.  Keeping this strict prevents accidental activation
            # of a head-only or threshold-only proposal through the unified
            # registry.
            raise ValueError("complete revision candidate requires a feature-bank artifact")
        scope = {
            "category": category,
            "run_id": run_id,
            "bundle_hash": bundle_hash,
            "scope_key": self.revision_scope_key(category, run_id, bundle_hash),
        }
        snapshot = feedback_snapshot or source.get("feedback_snapshot")
        if snapshot is None:
            with self._records_guard():
                snapshot = self._current_revision_snapshot_unlocked(
                    category, run_id, bundle_hash
                )
        bindings = self._normalise_feedback_bindings(snapshot)
        snapshot_payload = {
            "category": category,
            "model_version": run_id,
            "bundle_hash": bundle_hash,
            "training": bindings["training"],
            "validation": bindings["validation"],
        }
        snapshot_payload["sha256"] = _canonical_sha256(snapshot_payload)
        validation_payload = dict(validation or source.get("validation") or {})
        training_payload = dict(training or source.get("training") or {})
        artifacts_dir = self.revisions / revision_id
        head_path = artifacts_dir / "head.pt"
        feature_path = artifacts_dir / "feature_bank.npz"
        receipt_path = self.candidates / f"{revision_id}.yaml"
        if receipt_path.exists() or receipt_path.is_symlink():
            raise FileExistsError(f"revision receipt already exists: {receipt_path}")
        created: list[Path] = []
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=False)
            self._write_binary_atomic(head_path, head_bytes)
            created.append(head_path)
            head_hash = _sha256_file(head_path)
            feature_entry: dict[str, Any] | None = None
            if feature_bytes is not None:
                self._write_binary_atomic(feature_path, feature_bytes)
                created.append(feature_path)
                # Parse the bytes written to disk so receipt metadata cannot be
                # forged by supplying a mismatched sample-id list.
                with np.load(io.BytesIO(feature_bytes), allow_pickle=False) as loaded:
                    parsed_values = np.asarray(loaded["embeddings"], dtype=np.float32)
                    parsed_ids = [str(value) for value in loaded["sample_ids"].tolist()]
                if (
                    parsed_values.ndim != 2
                    or len(parsed_values) == 0
                    or len(parsed_values) != len(parsed_ids)
                    or len(set(parsed_ids)) != len(parsed_ids)
                    or not np.all(np.isfinite(parsed_values))
                ):
                    raise ValueError("feature bank artifact is malformed")
                feature_entry = {
                    "path": feature_path.relative_to(self.root).as_posix(),
                    "sha256": _sha256_file(feature_path),
                    "bytes": int(feature_path.stat().st_size),
                    "sample_count": int(len(parsed_values)),
                    "sample_ids": parsed_ids,
                    "embedding_dimension": int(parsed_values.shape[1]),
                }
            payload: dict[str, Any] = {
                "schema_version": 2,
                "revision_type": "complete_adaptive_feedback",
                "revision_id": revision_id,
                "created_at": _now(),
                **scope,
                "parent_revision_id": parent_revision_id,
                "threshold": threshold_value,
                "artifacts": {
                    "head": {
                        "path": head_path.relative_to(self.root).as_posix(),
                        "sha256": head_hash,
                        "bytes": int(head_path.stat().st_size),
                    },
                    "feature_bank": feature_entry,
                },
                "feedback_snapshot": snapshot_payload,
                "training": training_payload,
                "validation": validation_payload,
                "metadata": dict(metadata or source.get("metadata") or {}),
            }
            # Preserve auditable, JSON/YAML-safe proposal fields at the receipt
            # top level.  Artifact bytes/arrays are intentionally excluded.
            for key, value in source.items():
                if key in payload or key in {
                    "head_artifact", "head_bytes", "head_path", "head_overlay",
                    "feature_embeddings", "feature_sample_ids", "feature_artifact",
                    "feature_bank_path", "path", "receipt_path", "receipt_sha256",
                }:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None or isinstance(value, (dict, list, tuple)):
                    try:
                        json.dumps(value, ensure_ascii=False, allow_nan=False)
                    except (TypeError, ValueError):
                        continue
                    payload[key] = value
            payload["payload_sha256"] = _canonical_sha256(payload)
            _write_yaml_atomic(receipt_path, payload)
            created.append(receipt_path)
            receipt_hash = _sha256_file(receipt_path)
            # A revision candidate is immutable once returned to a caller.
            # Verify the complete receipt while every newly-created artifact
            # is still inside this transaction, so malformed overlays or
            # stale evidence never accumulate as unusable candidates.
            with self._records_guard():
                self._verify_revision_candidate_unlocked(
                    payload,
                    category=category,
                    run_id=run_id,
                    bundle_hash=bundle_hash,
                )
        except Exception:
            for path in reversed(created):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                artifacts_dir.rmdir()
            except OSError:
                pass
            raise
        result = dict(payload)
        result["receipt_path"] = receipt_path.relative_to(self.root).as_posix()
        result["path"] = result["receipt_path"]
        result["receipt_sha256"] = receipt_hash
        return result

    def _verify_revision_feedback_snapshot_unlocked(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        category = _scope_text(payload.get("category"), "revision.category")
        run_id = _scope_text(payload.get("run_id"), "revision.run_id")
        bundle_hash = _require_sha256(payload.get("bundle_hash"), "revision.bundle_hash")
        persisted = payload.get("feedback_snapshot")
        if not isinstance(persisted, Mapping):
            raise ValueError("revision feedback snapshot is missing")
        persisted = dict(persisted)
        recorded_hash = _require_sha256(
            persisted.get("sha256"), "revision feedback snapshot sha256"
        )
        unsigned = dict(persisted)
        unsigned.pop("sha256", None)
        if _canonical_sha256(unsigned) != recorded_hash:
            raise ValueError("revision feedback snapshot hash is invalid")
        if persisted.get("category") != category or persisted.get("model_version") != run_id:
            raise ValueError("revision feedback snapshot scope does not match revision")
        if persisted.get("bundle_hash") != bundle_hash:
            raise ValueError("revision feedback snapshot bundle hash does not match revision")
        # Validate shape and duplicate IDs before comparing with live records.
        normalised = self._normalise_feedback_bindings(persisted)
        if normalised["training"] != persisted.get("training", []) or normalised["validation"] != persisted.get("validation", []):
            raise ValueError("revision feedback snapshot bindings are invalid")
        current = self._current_revision_snapshot_unlocked(category, run_id, bundle_hash)
        # A revision binds the evidence that existed when it was proposed, but
        # operators must still be able to collect new feedback after activation.
        # Require every bound row to remain byte-for-byte represented while
        # allowing additional, later rows in the live store.
        current_bindings = self._normalise_feedback_bindings(current)
        for role in ("training", "validation"):
            current_by_id = {
                row["feedback_id"]: row for row in current_bindings[role]
            }
            for bound in normalised[role]:
                if current_by_id.get(bound["feedback_id"]) != bound:
                    raise ValueError(
                        "revision feedback snapshot does not match current records"
                    )
        return dict(persisted)

    @staticmethod
    def _metric_value(metrics: Mapping[str, Any], key: str) -> float:
        return _finite_float(metrics.get(key), f"metrics.{key}")

    def _verify_revision_validation_unlocked(
        self, payload: Mapping[str, Any]
    ) -> None:
        validation = payload.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("revision independent validation receipt is missing")
        if validation.get("passed") is not True:
            raise ValueError(
                f"revision independent validation failed: {validation.get('reason', 'unknown')}"
            )
        category = str(payload["category"])
        run_id = str(payload["run_id"])
        rows = [
            row
            for row in self._records_unlocked(category, "validation", run_id)
            if str(row.get("optimization_target", "appearance")) != "audit_only"
        ]
        by_id = {str(row.get("feedback_id")): row for row in rows}
        bindings = validation.get("samples", validation.get("sample_rows", []))
        if not isinstance(bindings, list):
            raise TypeError("revision validation samples must be a list")
        declared_ids = validation.get("sample_ids")
        if declared_ids is not None:
            if not isinstance(declared_ids, list):
                raise TypeError("revision validation sample_ids must be a list")
            if {str(value) for value in declared_ids} != {str(item.get("feedback_id")) for item in bindings if isinstance(item, Mapping)}:
                raise ValueError("revision validation sample_ids do not match sample rows")
        if len(bindings) < 4:
            raise ValueError("revision validation set is too small")
        labels: list[int] = []
        parent_scores: list[float] = []
        candidate_scores: list[float] = []
        seen: set[str] = set()
        for item in bindings:
            if not isinstance(item, Mapping):
                raise TypeError("revision validation sample row must be a mapping")
            feedback_id = _scope_text(str(item.get("feedback_id", "")), "validation feedback_id")
            if feedback_id in seen or feedback_id not in by_id:
                raise ValueError("revision validation sample IDs do not match current records")
            seen.add(feedback_id)
            row = by_id[feedback_id]
            labels.append(int(row.get("operator_label", -1)))
            parent_scores.append(_finite_float(item.get("parent_score"), "parent_score"))
            candidate_scores.append(_finite_float(item.get("candidate_score"), "candidate_score"))
            expected_image = _require_sha256(row.get("image_sha256"), "validation image_sha256")
            if _require_sha256(item.get("image_sha256"), "revision validation image_sha256") != expected_image:
                raise ValueError("revision validation image hash does not match record")
            expected_record = _record_hash(row)
            if _require_sha256(item.get("record_sha256"), "revision validation record_sha256") != expected_record:
                raise ValueError("revision validation record hash does not match record")
        if len(labels) < 4 or labels.count(0) < 2 or labels.count(1) < 2:
            raise ValueError("independent validation requires at least two normal and two defect samples")
        threshold = _finite_float(payload.get("threshold"), "revision threshold")
        parent_threshold = _finite_float(
            validation.get("parent_threshold", payload.get("parent_threshold", threshold)),
            "parent_threshold",
        )
        expected_parent = classification_metrics(np.asarray(labels), np.asarray(parent_scores), parent_threshold)
        expected_candidate = classification_metrics(np.asarray(labels), np.asarray(candidate_scores), threshold)
        recorded_parent = validation.get("parent_metrics", validation.get("base_metrics"))
        recorded_candidate = validation.get("candidate_metrics")
        if not isinstance(recorded_parent, Mapping) or not isinstance(recorded_candidate, Mapping):
            raise ValueError("revision validation metrics are missing")
        if recorded_parent != expected_parent:
            raise ValueError("revision parent validation metrics do not match scores")
        if recorded_candidate != expected_candidate:
            raise ValueError("revision candidate validation metrics do not match scores")
        minimum_recall = _finite_float(validation.get("minimum_recall", 0.90), "minimum_recall")
        acceptance = _revision_validation_acceptance(
            expected_parent,
            expected_candidate,
            minimum_recall,
        )
        if not acceptance["passed"]:
            raise ValueError(
                "revision candidate neither meets the absolute recall/F1 gate "
                "nor safely improves independent validation"
            )
        declared_mode = validation.get("acceptance_mode")
        if acceptance["acceptance_mode"] == "relative_improvement":
            required_relative_fields = {
                "acceptance_policy",
                "acceptance_mode",
                "absolute_gate_passed",
                "relative_improvement_passed",
                "recall_delta",
                "f1_delta",
                "reason",
            }
            missing_relative_fields = sorted(
                required_relative_fields.difference(validation)
            )
            if missing_relative_fields:
                raise ValueError(
                    "relative validation improvement receipt is missing auditable "
                    f"fields: {', '.join(missing_relative_fields)}"
                )
        declared_policy = validation.get("acceptance_policy")
        if (
            declared_policy is not None
            and declared_policy != acceptance["acceptance_policy"]
        ):
            raise ValueError("revision validation acceptance policy is unsupported")
        if declared_mode is not None and declared_mode != acceptance["acceptance_mode"]:
            raise ValueError("revision validation acceptance mode does not match metrics")
        for key in (
            "absolute_gate_passed",
            "relative_improvement_passed",
        ):
            if key in validation and validation.get(key) is not acceptance[key]:
                raise ValueError(
                    f"revision validation {key} does not match metrics"
                )
        for key in ("recall_delta", "f1_delta"):
            if key in validation and not math.isclose(
                _finite_float(validation.get(key), f"validation.{key}"),
                float(acceptance[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"revision validation {key} does not match metrics"
                )
        declared_reason = validation.get("reason")
        if (
            acceptance["acceptance_mode"] == "relative_improvement"
            and declared_reason != acceptance["reason"]
        ):
            raise ValueError("revision validation reason does not match metrics")
        if declared_mode is not None and declared_reason not in {
            None,
            acceptance["reason"],
        }:
            raise ValueError("revision validation reason does not match metrics")

    def _verify_revision_candidate_unlocked(
        self,
        payload: Mapping[str, Any],
        *,
        category: str | None = None,
        run_id: str | None = None,
        bundle_hash: str | None = None,
    ) -> dict[str, Any]:
        if int(payload.get("schema_version", 0)) < 2:
            raise ValueError("revision receipt schema is unsupported")
        expected_category = _scope_text(category or payload.get("category"), "category")
        expected_run = _scope_text(run_id or payload.get("run_id"), "run_id")
        expected_bundle = _require_sha256(bundle_hash or payload.get("bundle_hash"), "bundle_hash")
        if payload.get("category") != expected_category or payload.get("run_id") != expected_run:
            raise ValueError("revision scope does not match requested model")
        if payload.get("bundle_hash") != expected_bundle:
            raise ValueError("revision bundle hash does not match requested model")
        if payload.get("scope_key") != self.revision_scope_key(expected_category, expected_run, expected_bundle):
            raise ValueError("revision scope key is invalid")
        _scope_text(payload.get("revision_id"), "revision_id")
        parent_id = payload.get("parent_revision_id")
        if parent_id not in {None, ""}:
            parent_id = _scope_text(parent_id, "parent_revision_id")
        self._verify_revision_artifacts(payload)
        snapshot = self._verify_revision_feedback_snapshot_unlocked(payload)

        # Training bindings are part of the immutable receipt.  Recompute the
        # parent/candidate metrics from those bindings so a caller cannot
        # manufacture a passing receipt by changing only summary numbers.
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("revision training receipt is missing")
        samples = training.get("samples", training.get("sample_rows"))
        if not isinstance(samples, list) or not samples:
            raise ValueError("revision training sample list is empty")
        rows = self._records_unlocked(expected_category, "training", expected_run)
        by_id = {str(row.get("feedback_id")): row for row in rows}
        eligible_rows = {
            key: row
            for key, row in by_id.items()
            if str(row.get("optimization_target", "appearance")) != "audit_only"
        }
        seen_training: set[str] = set()
        labels: list[int] = []
        parent_scores: list[float] = []
        candidate_scores: list[float] = []
        for item in samples:
            if not isinstance(item, Mapping):
                raise TypeError("revision training sample row must be a mapping")
            feedback_id = _scope_text(str(item.get("feedback_id", "")), "training feedback_id")
            if feedback_id in seen_training or feedback_id not in eligible_rows:
                raise ValueError("revision training sample IDs do not match records")
            seen_training.add(feedback_id)
            row = eligible_rows[feedback_id]
            expected_image = _require_sha256(row.get("image_sha256"), "training image_sha256")
            if _require_sha256(item.get("image_sha256"), "revision training image_sha256") != expected_image:
                raise ValueError("revision training image hash does not match record")
            if _require_sha256(item.get("record_sha256"), "revision training record_sha256") != _record_hash(row):
                raise ValueError("revision training record hash does not match record")
            labels.append(int(row.get("operator_label", -1)))
            parent_scores.append(_finite_float(item.get("parent_score"), "training parent_score"))
            candidate_scores.append(_finite_float(item.get("candidate_score"), "training candidate_score"))
        if len(samples) < 6 or labels.count(0) < 2 or labels.count(1) < 2:
            raise ValueError("revision training requires at least six samples with two normal and two defect labels")
        declared_ids = training.get("sample_ids")
        if declared_ids is not None and {str(v) for v in declared_ids} != seen_training:
            raise ValueError("revision training sample_ids do not match sample rows")
        if training.get("sample_count") is not None and int(training["sample_count"]) != len(samples):
            raise ValueError("revision training sample count does not match")
        if training.get("normal_count") is not None and int(training["normal_count"]) != labels.count(0):
            raise ValueError("revision training normal count does not match")
        if training.get("defect_count") is not None and int(training["defect_count"]) != labels.count(1):
            raise ValueError("revision training defect count does not match")
        threshold = _finite_float(payload.get("threshold"), "revision threshold")
        parent_threshold = _finite_float(
            payload.get("parent_threshold", training.get("parent_threshold", threshold)),
            "parent_threshold",
        )
        expected_parent_training = classification_metrics(
            np.asarray(labels), np.asarray(parent_scores), parent_threshold
        )
        expected_candidate_training = classification_metrics(
            np.asarray(labels), np.asarray(candidate_scores), threshold
        )
        recorded_parent_training = payload.get("base_metrics", training.get("base_metrics"))
        recorded_candidate_training = payload.get("candidate_metrics", training.get("candidate_metrics"))
        if isinstance(recorded_parent_training, Mapping) and dict(recorded_parent_training) != expected_parent_training:
            raise ValueError("revision parent training metrics do not match scores")
        if isinstance(recorded_candidate_training, Mapping) and dict(recorded_candidate_training) != expected_candidate_training:
            raise ValueError("revision candidate training metrics do not match scores")

        # A feature bank may contain only newly learned defect vectors, but all
        # IDs must still resolve to appearance defect training evidence.
        feature_meta = payload.get("feature_bank")
        if isinstance(feature_meta, Mapping):
            feature_ids = [str(v) for v in feature_meta.get("sample_ids", [])]
            if len(set(feature_ids)) != len(feature_ids):
                raise ValueError("revision feature bank sample IDs must be unique")
            defect_ids = {
                key for key, row in eligible_rows.items()
                if int(row.get("operator_label", -1)) == 1
            }
            if not set(feature_ids).issubset(defect_ids):
                raise ValueError("revision feature bank IDs are not defect training feedback")
            declared_hashes = feature_meta.get("sample_sha256")
            if declared_hashes is not None:
                expected_hashes = [str(eligible_rows[key].get("image_sha256")) for key in feature_ids]
                if [str(v) for v in declared_hashes] != expected_hashes:
                    raise ValueError("revision feature bank image hashes do not match records")
        training_images = {
            _require_sha256(item.get("image_sha256"), "revision training image_sha256")
            for item in samples
            if isinstance(item, Mapping)
        }
        validation_images = {
            _require_sha256(item.get("image_sha256"), "revision validation image_sha256")
            for item in payload.get("validation", {}).get("samples", [])
            if isinstance(item, Mapping)
        } if isinstance(payload.get("validation"), Mapping) else set()
        if training_images & validation_images:
            raise ValueError("revision training and validation image content overlaps")
        self._verify_revision_validation_unlocked(payload)
        fingerprint = payload.get("head_parameter_fingerprint")
        if not isinstance(fingerprint, Mapping):
            raise ValueError("revision head parameter fingerprint is missing")
        if [str(v) for v in fingerprint.get("trainable_parameters", [])] != ["weight2", "bias2"]:
            raise ValueError("revision head trainable parameter set is invalid")
        if int(fingerprint.get("updated_parameter_count", -1)) != 33:
            raise ValueError("revision head must update exactly 33 parameters")
        parent_fp = fingerprint.get("parent")
        candidate_fp = fingerprint.get("candidate")
        if not isinstance(parent_fp, Mapping) or not isinstance(candidate_fp, Mapping):
            raise ValueError("revision head parent/candidate fingerprints are missing")
        for name in ("feature_mean", "feature_scale", "weight1", "bias1"):
            parent_value = _require_sha256(parent_fp.get(name), f"revision parent {name}")
            candidate_value = _require_sha256(candidate_fp.get(name), f"revision candidate {name}")
            if parent_value != candidate_value:
                raise ValueError(f"revision frozen head tensor changed: {name}")
        artifacts = payload.get("artifacts")
        head_path = artifacts.get("head", {}).get("path") if isinstance(artifacts, Mapping) and isinstance(artifacts.get("head"), Mapping) else None
        if head_path is None:
            raise ValueError("revision head overlay artifact is missing")
        overlay = torch.load(
            self._safe_revision_path(head_path, must_exist=True),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(overlay, Mapping):
            raise ValueError("revision head overlay is malformed")
        if overlay.get("parent_parameter_fingerprint") != dict(parent_fp) or overlay.get("candidate_parameter_fingerprint") != dict(candidate_fp):
            raise ValueError("revision head receipt fingerprints do not match overlay")
        return snapshot

    def active_revision(
        self,
        category: str,
        run_id: str | None = None,
        bundle_hash: str | None = None,
        *,
        model_version: str | None = None,
        verify: bool = True,
    ) -> dict[str, Any] | None:
        category = _scope_text(category, "category")
        run_id = _scope_text(run_id or model_version, "run_id")
        requested_bundle = (
            None if bundle_hash is None else _require_sha256(bundle_hash, "bundle_hash")
        )
        with self._records_guard():
            registry = self._revision_registry()
            active = registry.get("active", {})
            if not isinstance(active, Mapping):
                raise TypeError("revision registry active field is invalid")
            entry: Mapping[str, Any] | None = None
            scope: str | None = None
            if requested_bundle is not None:
                scope = self.revision_scope_key(category, run_id, requested_bundle)
                raw = active.get(scope)
                if raw is not None and not isinstance(raw, Mapping):
                    raise TypeError("active revision entry is invalid")
                entry = raw
            else:
                matches = [
                    (str(key), value)
                    for key, value in active.items()
                    if isinstance(value, Mapping)
                    and value.get("category") == category
                    and value.get("run_id") == run_id
                ]
                if len(matches) == 1:
                    scope, entry = matches[0]
                elif len(matches) > 1:
                    raise ValueError("bundle_hash is required when multiple revisions share a run_id")
            if entry is None:
                return None
            result = dict(entry)
            entry_bundle = _require_sha256(result.get("bundle_hash"), "active revision bundle_hash")
            if result.get("category") != category or result.get("run_id") != run_id:
                raise ValueError("active revision scope does not match registry lookup")
            expected_scope = self.revision_scope_key(category, run_id, entry_bundle)
            if scope is not None and scope != expected_scope:
                raise ValueError("active revision scope key is invalid")
            if requested_bundle is not None and entry_bundle != requested_bundle:
                raise ValueError("active revision bundle hash does not match lookup")
            receipt = result.get("receipt")
            if not isinstance(receipt, Mapping):
                raise ValueError("active revision has no receipt reference")
            payload, path, receipt_hash = self._load_revision_receipt(
                {
                    "revision_id": result.get("revision_id"),
                    "receipt_path": receipt.get("path"),
                    "receipt_sha256": receipt.get("sha256"),
                }
            )
            if receipt.get("path") != path.relative_to(self.root).as_posix() or receipt.get("sha256") != receipt_hash:
                raise ValueError("active revision receipt reference does not match")
            if payload.get("category") != category or payload.get("run_id") != run_id or payload.get("bundle_hash") != entry_bundle:
                raise ValueError("active revision receipt scope does not match registry entry")
            if verify:
                self._verify_revision_candidate_unlocked(
                    payload,
                    category=category,
                    run_id=run_id,
                    bundle_hash=entry_bundle,
                )
            result["receipt"] = {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": receipt_hash,
            }
            result["payload"] = payload
            return result

    def activate_revision(
        self,
        candidate: Mapping[str, Any] | str,
        *,
        category: str | None = None,
        run_id: str | None = None,
        bundle_hash: str | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        # Resolve once outside the lock for early, useful path errors, then
        # resolve again while holding the records lock.  The second read makes
        # receipt replacement during activation a hard failure (TOCTOU guard).
        payload, receipt_path, receipt_hash = self._load_revision_receipt(candidate)
        requested_category = category or payload.get("category")
        requested_run = run_id or model_version or payload.get("run_id")
        requested_bundle = bundle_hash or payload.get("bundle_hash")
        requested_category = _scope_text(requested_category, "category")
        requested_run = _scope_text(requested_run, "run_id")
        requested_bundle = _require_sha256(requested_bundle, "bundle_hash")
        with self._records_guard():
            current_payload, current_receipt_path, current_receipt_hash = self._load_revision_receipt(candidate)
            if current_receipt_hash != receipt_hash or current_payload != payload or current_receipt_path != receipt_path:
                raise ValueError("revision receipt changed during activation")
            self._verify_revision_candidate_unlocked(
                current_payload,
                category=requested_category,
                run_id=requested_run,
                bundle_hash=requested_bundle,
            )
            registry = self._revision_registry()
            scope = self.revision_scope_key(requested_category, requested_run, requested_bundle)
            current = registry["active"].get(scope)
            if current is not None:
                if not isinstance(current, Mapping):
                    raise TypeError("active revision entry is invalid")
                if current.get("category") != requested_category or current.get("run_id") != requested_run or _require_sha256(current.get("bundle_hash"), "active revision bundle_hash") != requested_bundle:
                    raise ValueError("active revision scope does not match registry key")
                # A malformed current pointer must never be silently replaced;
                # this also protects the rollback chain from forged metadata.
                self._verify_revision_entry_unlocked(current)
            if current is not None and current.get("revision_id") == payload.get("revision_id"):
                return dict(current)
            parent_id = payload.get("parent_revision_id")
            current_id = None if current is None else current.get("revision_id")
            if current_id is not None and current_id != parent_id:
                raise ValueError("revision parent is not the current active revision")
            entry = {
                "revision_id": str(payload["revision_id"]),
                "category": requested_category,
                "run_id": requested_run,
                "bundle_hash": requested_bundle,
                "scope_key": scope,
                "threshold": _finite_float(payload.get("threshold"), "revision threshold"),
                "artifacts": payload.get("artifacts"),
                "feedback_snapshot": payload.get("feedback_snapshot"),
                "receipt": {
                    "path": receipt_path.relative_to(self.root).as_posix(),
                    "sha256": receipt_hash,
                },
                "activated_at": _now(),
                "previous": current,
                "parent_revision_id": parent_id,
            }
            registry["active"][scope] = entry
            registry["history"].append({"action": "activate", **entry})
            _write_yaml_atomic(self.revision_registry_path, registry)
            return entry

    def rollback_revision(
        self,
        category: str,
        run_id: str | None = None,
        bundle_hash: str | None = None,
        *,
        model_version: str | None = None,
    ) -> dict[str, Any] | None:
        category = _scope_text(category, "category")
        run_id = _scope_text(run_id or model_version, "run_id")
        if bundle_hash is None:
            # Resolve an unambiguous active scope for convenience.
            registry = self._revision_registry()
            matches = [
                (key, value)
                for key, value in registry.get("active", {}).items()
                if isinstance(value, Mapping)
                and value.get("category") == category
                and value.get("run_id") == run_id
            ]
            if len(matches) != 1:
                if not matches:
                    return None
                raise ValueError("bundle_hash is required when multiple active scopes exist")
            scope, current = matches[0]
            bundle_hash = str(current.get("bundle_hash"))
        bundle_hash = _require_sha256(bundle_hash, "bundle_hash")
        with self._records_guard():
            registry = self._revision_registry()
            scope = self.revision_scope_key(category, run_id, bundle_hash)
            current = registry["active"].get(scope)
            if current is None:
                return None
            if not isinstance(current, Mapping):
                raise TypeError("active revision entry is invalid")
            if (
                current.get("category") != category
                or current.get("run_id") != run_id
                or _require_sha256(current.get("bundle_hash"), "active revision bundle_hash") != bundle_hash
            ):
                raise ValueError("active revision scope does not match registry key")
            # Validate the complete current state before changing the pointer.
            self._verify_revision_entry_unlocked(current)
            previous = current.get("previous")
            if previous is not None:
                if not isinstance(previous, Mapping):
                    raise ValueError("revision rollback pointer is invalid")
                if (
                    previous.get("category") != category
                    or previous.get("run_id") != run_id
                    or _require_sha256(previous.get("bundle_hash"), "rollback bundle_hash") != bundle_hash
                ):
                    raise ValueError("revision rollback scope does not match active model")
                # Verify the previous receipt/artifacts before exposing it.
                self._verify_revision_entry_unlocked(previous)
                registry["active"][scope] = dict(previous)
            else:
                registry["active"].pop(scope, None)
            event = {
                "action": "rollback",
                "category": category,
                "run_id": run_id,
                "bundle_hash": bundle_hash,
                "from_revision_id": current.get("revision_id"),
                "to_revision_id": None if previous is None else previous.get("revision_id"),
                "rolled_back_at": _now(),
            }
            registry["history"].append(event)
            _write_yaml_atomic(self.revision_registry_path, registry)
            return None if previous is None else dict(previous)

    def _verify_revision_entry_unlocked(self, entry: Mapping[str, Any]) -> None:
        receipt = entry.get("receipt")
        if not isinstance(receipt, Mapping):
            raise ValueError("revision rollback entry has no receipt")
        payload, _path, _hash = self._load_revision_receipt(
            {
                "revision_id": entry.get("revision_id"),
                "receipt_path": receipt.get("path"),
                "receipt_sha256": receipt.get("sha256"),
            }
        )
        self._verify_revision_candidate_unlocked(
            payload,
            category=str(entry.get("category")),
            run_id=str(entry.get("run_id")),
            bundle_hash=str(entry.get("bundle_hash")),
        )

    # Explicit aliases make the public registry vocabulary unambiguous.
    save_adaptive_revision_candidate = save_revision_candidate
    activate_adaptive_revision = activate_revision
    rollback_adaptive_revision = rollback_revision

    def rollback_feature_bank(
        self, category: str, model_version: str
    ) -> dict[str, Any] | None:
        """Restore the previous feature-bank pointer (legacy registry API)."""
        category = _scope_text(category, "category")
        model_version = _scope_text(model_version, "model_version")
        with self._records_guard():
            registry = self.feature_registry()
            scope = _scope_key(category, model_version)
            current = registry["active"].get(scope)
            if current is None:
                return None
            previous = current.get("previous")
            if previous is None:
                registry["active"].pop(scope, None)
            else:
                if not isinstance(previous, Mapping):
                    raise ValueError("feature-bank rollback pointer is invalid")
                # Verify the referenced file before committing the pointer.
                raw = previous.get("path")
                path = self._safe_revision_path(raw, must_exist=True)
                if _sha256_file(path) != _require_sha256(previous.get("file_sha256"), "feature bank file_sha256"):
                    raise ValueError("feature-bank rollback artifact hash mismatch")
                registry["active"][scope] = dict(previous)
            registry["history"].append(
                {
                    "action": "rollback",
                    "category": category,
                    "model_version": model_version,
                    "from_version": current.get("version"),
                    "to_version": None if previous is None else previous.get("version"),
                    "rolled_back_at": _now(),
                }
            )
            _write_yaml_atomic(self.feature_registry_path, registry)
            return None if previous is None else dict(previous)
