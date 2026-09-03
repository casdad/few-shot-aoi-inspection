from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .service import FeedbackStore


@dataclass(frozen=True)
class FeedbackRuntimeState:
    enabled: bool
    category: str
    model_version: str
    store_root: str | None
    threshold: float | None
    threshold_source: str
    feature_version: str | None
    feature_sample_count: int
    revision_id: str | None = None
    head_version: str | None = None
    head_parameter_count: int = 0
    threshold_version: str | None = None
    bundle_hash: str | None = None

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


def runtime_model_version(runtime: Any) -> str:
    bundle = getattr(runtime, "bundle", None)
    if not isinstance(bundle, Mapping):
        raise TypeError("runtime does not expose model-bundle metadata")
    value = str(bundle.get("run_id", "")).strip()
    if value:
        return value
    bundle_path = Path(getattr(runtime, "bundle_path", ""))
    if bundle_path.name:
        return bundle_path.parent.name
    raise ValueError("model bundle has no stable run_id")


def set_runtime_feature_bank(
    runtime: Any,
    embeddings: np.ndarray | None,
    *,
    feature_version: str | None,
) -> None:
    dino = getattr(runtime, "dino", None)
    if dino is None or getattr(dino, "defect_global_bank", None) is None:
        if embeddings is None:
            # A head-only/legacy runtime has no mutable DINO bank to restore.
            return
        raise RuntimeError("runtime DINO defect feature bank is unavailable")
    if not hasattr(runtime, "_feedback_base_defect_bank"):
        runtime._feedback_base_defect_bank = (
            dino.defect_global_bank.detach().clone()
        )
        runtime._feedback_feature_version = None
    if embeddings is not None and runtime._feedback_feature_version == feature_version:
        return

    base = runtime._feedback_base_defect_bank.to(dino.device)
    if embeddings is None:
        # Always assign the frozen snapshot, even when the version marker is
        # already None.  This closes the stale-marker case after a failed or
        # externally interrupted activation.
        dino.defect_global_bank = base.detach().clone()
    else:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or not len(values):
            raise ValueError("feedback embeddings must be a non-empty rank-2 matrix")
        if values.shape[1] != base.shape[1]:
            raise ValueError(
                "feedback embedding dimension does not match the active model"
            )
        feedback = F.normalize(torch.from_numpy(values).to(dino.device), dim=1)
        dino.defect_global_bank = torch.cat((base, feedback), dim=0)
    runtime._feedback_feature_version = feature_version


def _runtime_lock(runtime: Any) -> threading.RLock:
    lock = getattr(runtime, "_feedback_lock", None)
    if lock is None:
        lock = threading.RLock()
        runtime._feedback_lock = lock
    return lock


def _clone_head_value(head: Any) -> Any:
    clone = getattr(head, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(head)


def _bundle_hash(runtime: Any) -> str | None:
    bundle = getattr(runtime, "bundle", None)
    if isinstance(bundle, Mapping):
        value = bundle.get("bundle_hash")
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
        # Older bundles did not persist a bundle hash.  They still need a
        # deterministic scope; otherwise a revision can be accidentally
        # restored into another bundle with the same run ID.
        unsigned = {key: item for key, item in bundle.items() if key != "bundle_hash"}
        try:
            encoded = json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(encoded).hexdigest()
    return None


def _safe_runtime_artifact_path(store: FeedbackStore, raw: object) -> Path:
    """Resolve a revision artifact without permitting traversal or links."""
    checker = getattr(store, "_safe_revision_path", None)
    if callable(checker):
        return checker(raw, must_exist=True)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("active revision artifact path is missing")
    value = raw.replace("\\", "/")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or any(
        part in {"", "."} for part in candidate.parts
    ):
        raise PermissionError("active revision artifact path must be relative and contained")
    cursor = store.root
    for part in candidate.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PermissionError("active revision artifact path must not contain a symbolic link")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(store.root.resolve(strict=True))
    except ValueError as error:
        raise PermissionError("active revision artifact path escapes feedback root") from error
    return path


def _ensure_base_state(runtime: Any) -> None:
    if not hasattr(runtime, "_feedback_base_defect_bank"):
        dino = getattr(runtime, "dino", None)
        bank = None if dino is None else getattr(dino, "defect_global_bank", None)
        runtime._feedback_base_defect_bank = None if bank is None else bank.detach().clone()
        runtime._feedback_feature_version = None
    if not hasattr(runtime, "_feedback_base_head") and hasattr(runtime, "head"):
        runtime._feedback_base_head = _clone_head_value(runtime.head)
        runtime._feedback_head_version = None
    if not hasattr(runtime, "_feedback_threshold"):
        runtime._feedback_threshold = None


def _load_revision_head(runtime: Any, store: FeedbackStore, payload: Mapping[str, Any]) -> Any:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("active revision has no artifacts")
    head_entry = artifacts.get("head")
    if not isinstance(head_entry, Mapping):
        raise ValueError("active revision has no head artifact")
    path = _safe_runtime_artifact_path(store, head_entry.get("path"))
    expected = str(head_entry.get("sha256", "")).lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("active revision head artifact SHA-256 is invalid")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("active revision head artifact SHA-256 mismatch")
    current = getattr(runtime, "head", None)
    try:
        serialized = torch.load(
            io.BytesIO(data),
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError("active revision head artifact is malformed") from error
    if isinstance(serialized, Mapping) and serialized.get("artifact_type") == (
        "dinov2_tiled_mlp_feedback_output_layer_overlay"
    ):
        # Every overlay is applied to a fresh clone of the frozen bundle head,
        # never to the runtime's currently active revision.  This makes process
        # restarts and direct rollback restoration deterministic.
        parent = getattr(runtime, "_feedback_base_head", current)
        if parent is None:
            raise RuntimeError("runtime has no parent head for feedback overlay")
        candidate = _clone_head_value(parent)
        applier = getattr(candidate, "apply_output_layer_overlay", None)
        if not callable(applier):
            raise TypeError("runtime head does not support feedback output-layer overlays")
        applier(dict(serialized))
        receipt_fingerprint = payload.get("head_parameter_fingerprint")
        candidate_fingerprint = (
            receipt_fingerprint.get("candidate")
            if isinstance(receipt_fingerprint, Mapping)
            else None
        )
        actual_fingerprint = getattr(candidate, "parameter_fingerprint", None)
        if isinstance(candidate_fingerprint, Mapping) and callable(actual_fingerprint):
            actual = actual_fingerprint()
            for name in ("feature_mean", "feature_scale", "weight1", "bias1", "weight2", "bias2", "overall_sha256"):
                if name in candidate_fingerprint and candidate_fingerprint[name] != actual.get(name):
                    raise ValueError(
                        f"active revision overlay does not match receipt fingerprint: {name}"
                    )
        return candidate
    cls = type(current) if current is not None else None
    if cls is not None and hasattr(cls, "load"):
        try:
            return cls.load(path, device=str(getattr(runtime, "device", "cpu")))
        except TypeError:
            return cls.load(path)
    loaded = serialized
    if current is not None and isinstance(loaded, Mapping) and hasattr(current, "apply_output_layer_state"):
        candidate = _clone_head_value(current)
        state = loaded.get("output_layer", loaded)
        candidate.apply_output_layer_state(dict(state))
        return candidate
    return loaded


def _load_revision_feature_bank(
    runtime: Any, store: FeedbackStore, payload: Mapping[str, Any]
) -> tuple[np.ndarray | None, str | None, int]:
    artifacts = payload.get("artifacts")
    entry = artifacts.get("feature_bank") if isinstance(artifacts, Mapping) else None
    feature_meta = payload.get("feature_bank")
    if entry is None:
        return None, None, 0
    if not isinstance(entry, Mapping):
        raise TypeError("active revision feature artifact is invalid")
    path = _safe_runtime_artifact_path(store, entry.get("path"))
    expected = str(entry.get("sha256", "")).lower()
    import hashlib

    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("active revision feature artifact SHA-256 is invalid")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("active revision feature artifact SHA-256 mismatch")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            if "embeddings" not in loaded or "sample_ids" not in loaded:
                raise ValueError("active revision feature artifact has no sample IDs")
            values = np.asarray(loaded["embeddings"], dtype=np.float32)
            ids = [str(value) for value in loaded["sample_ids"].tolist()]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("active revision feature artifact is malformed") from error
    if values.ndim != 2 or len(values) != len(ids) or not np.all(np.isfinite(values)):
        raise ValueError("active revision feature artifact is malformed")
    if len(set(ids)) != len(ids) or np.any(np.linalg.norm(values, axis=1) <= 0):
        raise ValueError("active revision feature artifact is malformed")
    if isinstance(feature_meta, Mapping):
        expected_count = feature_meta.get("sample_count")
        if expected_count is not None and int(expected_count) != len(values):
            raise ValueError("active revision feature count does not match receipt")
        expected_ids = feature_meta.get("sample_ids")
        if expected_ids is not None and [str(value) for value in expected_ids] != ids:
            raise ValueError("active revision feature IDs do not match receipt")
        expected_dimension = feature_meta.get("embedding_dimension")
        if expected_dimension is not None and int(expected_dimension) != int(values.shape[1]):
            raise ValueError("active revision feature dimension does not match receipt")
    version = str(payload.get("revision_id"))
    return values, version, len(values)


def apply_active_feedback(
    runtime: Any,
    store: FeedbackStore,
    category: str,
    *,
    enabled: bool = True,
    model_version: str | None = None,
) -> FeedbackRuntimeState:
    version = model_version or runtime_model_version(runtime)
    lock = _runtime_lock(runtime)
    with lock:
        _ensure_base_state(runtime)
        dino = getattr(runtime, "dino", None)
        old_head = getattr(runtime, "head", None)
        old_bank = None
        if dino is not None and getattr(dino, "defect_global_bank", None) is not None:
            old_bank = dino.defect_global_bank.detach().clone()
        old_feature_version = getattr(runtime, "_feedback_feature_version", None)
        old_head_version = getattr(runtime, "_feedback_head_version", None)
        old_threshold = getattr(runtime, "_feedback_threshold", None)

        def restore_mutable_state() -> None:
            if hasattr(runtime, "head"):
                runtime.head = old_head
            if dino is not None and old_bank is not None:
                dino.defect_global_bank = old_bank
            runtime._feedback_feature_version = old_feature_version
            runtime._feedback_head_version = old_head_version
            runtime._feedback_threshold = old_threshold

        # A real FeedbackStore uses the unified registry exclusively.  A small
        # compatibility fallback remains for older embedders that only expose
        # active_threshold/active_feature_bank (and for the historical tests).
        revision_registry = getattr(store, "revision_registry_path", None)
        unified = callable(getattr(store, "active_revision", None)) and (
            revision_registry is None or Path(revision_registry).is_file()
        )
        active_revision = None
        if enabled and unified:
            runtime_bundle_hash = _bundle_hash(runtime)
            if runtime_bundle_hash is not None:
                active_revision = store.active_revision(
                    category, version, runtime_bundle_hash, verify=True
                )
            else:
                active_revision = store.active_revision(category, version, verify=True)
        if not enabled or (unified and active_revision is None):
            # Restore every mutable tensor from the initial frozen snapshot.
            if hasattr(runtime, "_feedback_base_head") and hasattr(runtime, "head"):
                runtime.head = _clone_head_value(runtime._feedback_base_head)
                runtime._feedback_head_version = None
            set_runtime_feature_bank(runtime, None, feature_version=None)
            runtime._feedback_threshold = None
            return FeedbackRuntimeState(
                enabled=False if not enabled else True,
                category=category,
                model_version=version,
                store_root=None if not enabled else str(store.root),
                threshold=None,
                threshold_source="model_bundle",
                feature_version=None,
                feature_sample_count=0,
                bundle_hash=_bundle_hash(runtime),
            )

        if unified and active_revision is not None:
            payload = active_revision.get("payload")
            if not isinstance(payload, Mapping):
                raise ValueError("active revision payload is unavailable")
            try:
                # Validate/load all artifacts before mutating runtime state.
                new_head = _load_revision_head(runtime, store, payload)
                embeddings, feature_version, feature_count = _load_revision_feature_bank(runtime, store, payload)
                threshold = float(payload["threshold"])
                if not np.isfinite(threshold):
                    raise ValueError("active revision threshold must be finite")
                # Commit as one lock-protected state transition.  Any failure
                # in the second component restores the exact prior tensors.
                runtime.head = new_head
                set_runtime_feature_bank(runtime, embeddings, feature_version=feature_version)
                revision_id = str(payload.get("revision_id"))
                runtime._feedback_head_version = revision_id
                runtime._feedback_threshold = threshold
            except Exception:
                restore_mutable_state()
                raise
            fingerprint = payload.get("head_parameter_fingerprint")
            if isinstance(fingerprint, Mapping):
                updated_count = int(fingerprint.get("updated_parameter_count", 0))
            else:
                updated_count = int(payload.get("updated_parameter_count", 0))
            return FeedbackRuntimeState(
                enabled=True,
                category=category,
                model_version=version,
                store_root=str(store.root),
                threshold=threshold,
                threshold_source="active_operator_feedback",
                feature_version=feature_version,
                feature_sample_count=feature_count,
                revision_id=revision_id,
                head_version=revision_id,
                head_parameter_count=updated_count,
                threshold_version=revision_id,
                bundle_hash=_bundle_hash(runtime),
            )

        # Legacy store fallback.
        try:
            active = store.active_feature_bank(category, version)
            if active is None:
                embeddings = None
                feature_version = None
                feature_sample_count = 0
            else:
                embeddings, entry = active
                feature_version = str(entry["version"])
                feature_sample_count = int(entry["sample_count"])
            set_runtime_feature_bank(runtime, embeddings, feature_version=feature_version)
            threshold = store.active_threshold(category, version)
            runtime._feedback_threshold = threshold
            return FeedbackRuntimeState(
                enabled=True,
                category=category,
                model_version=version,
                store_root=str(store.root),
                threshold=threshold,
                threshold_source=("active_operator_feedback" if threshold is not None else "model_bundle"),
                feature_version=feature_version,
                feature_sample_count=feature_sample_count,
                bundle_hash=_bundle_hash(runtime),
            )
        except Exception:
            restore_mutable_state()
            raise
