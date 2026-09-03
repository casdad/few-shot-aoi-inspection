"""End-to-end human-feedback supervised incremental revision proposals.

This module intentionally owns the *proposal* transaction.  Submitting a
feedback record only writes evidence; this code is the only path that may
turn selected evidence into a candidate head/feature/threshold revision.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch
import yaml

from aoi.competition.models.known_defect_heads.dinov2_tiled_mlp import (
    DinoTiledMLPKnownHead,
)
from aoi.paths import resolve_artifact
from aoi.runtime.fusion import nine_tiles
from aoi.training.adaptation import nine_tile_defect

from .runtime import apply_active_feedback, runtime_model_version
from .service import (
    FeedbackStore,
    _canonical_sha256,
    _finite_float,
    _require_sha256,
    _revision_validation_acceptance,
    classification_metrics,
)


DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_L2_SP = 1e-3
DEFAULT_MINIMUM_RECALL = 0.90


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _bundle_hash(runtime: Any) -> str:
    bundle = getattr(runtime, "bundle", None)
    if isinstance(bundle, Mapping):
        value = bundle.get("bundle_hash")
        if isinstance(value, str) and len(value) == 64:
            try:
                return _require_sha256(value, "bundle_hash")
            except (TypeError, ValueError):
                pass
        unsigned = {key: item for key, item in bundle.items() if key != "bundle_hash"}
        return _canonical_sha256(_jsonable(unsigned))
    integrity = getattr(runtime, "integrity", None)
    if isinstance(integrity, Mapping):
        value = integrity.get("bundle_hash") or integrity.get("bundle_sha256")
        if isinstance(value, str) and len(value) == 64:
            return _require_sha256(value, "bundle_hash")
    path = getattr(runtime, "bundle_path", None)
    if path is not None:
        try:
            return _sha256_bytes(Path(path).read_bytes())
        except OSError:
            pass
    # Fake/lightweight runtimes still need a deterministic scope.  Hashing a
    # stable class identifier avoids accidentally sharing candidates between
    # unrelated runtime objects.
    return _canonical_sha256({"runtime": f"{type(runtime).__module__}.{type(runtime).__qualname__}"})


def _record_target(row: Mapping[str, Any]) -> str:
    target = str(row.get("optimization_target", "appearance")).strip().lower()
    return target or "appearance"


def _eligible_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    role: str,
    selected_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    selected = None if selected_ids is None else {str(value) for value in selected_ids}
    result: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if selected is not None and str(row.get("feedback_id")) not in selected:
            continue
        # Old audit-only rows are retained for record compatibility and never
        # participate in model optimization.
        if _record_target(row) == "audit_only":
            continue
        if role and str(row.get("dataset_role", role)) != role:
            continue
        result.append(row)
    result.sort(key=lambda item: str(item.get("feedback_id", "")))
    return result


def _read_feedback_images(store: FeedbackStore, rows: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for row in rows:
        reference = row.get("image")
        if not isinstance(reference, str) or Path(reference).is_absolute() or ".." in Path(reference).parts:
            raise PermissionError("feedback image path must be a contained relative path")
        raw_parts = reference.replace("\\", "/").split("/")
        if any(part in {"", ".", ".."} for part in raw_parts):
            raise PermissionError("feedback image path must be a contained relative path")
        cursor = store.root
        for part in raw_parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PermissionError("feedback image path must not contain a symbolic link")
        path = cursor.resolve(strict=True)
        try:
            path.relative_to(store.images.resolve(strict=True))
        except ValueError as error:
            raise PermissionError("feedback image path escapes the managed image directory") from error
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != str(row.get("image_sha256", "")).lower():
            raise ValueError(f"feedback image SHA-256 mismatch for {row.get('feedback_id')}")

        decode_error: Exception | None = None
        try:
            import cv2

            # ``cv2.imread(str(path))`` cannot reliably open Windows paths
            # containing non-ASCII characters.  Decode the already verified
            # bytes instead so Chinese project/product names remain valid.
            encoded = np.frombuffer(payload, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("decoder returned no image")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            continue
        except Exception as error:
            decode_error = error

        try:
            from PIL import Image

            with Image.open(io.BytesIO(payload)) as image:
                images.append(np.array(image.convert("RGB"), copy=True))
            continue
        except Exception as error:
            decode_error = error
            # Lightweight tests often store arbitrary bytes and provide a fake
            # runtime.  Let that runtime consume the bytes only when it has an
            # explicit decoder hook; real runtimes still fail clearly.
            decoder = getattr(store, "decode_image", None)
            if callable(decoder):
                images.append(np.asarray(decoder(payload)))
            else:
                raise ValueError(f"unable to decode feedback image {path}") from decode_error
    return images


def _feedback_roi_masks(
    rows: Sequence[Mapping[str, Any]],
    images: Sequence[Any],
) -> list[np.ndarray]:
    """Convert persisted ROI rectangles into image-sized binary masks.

    The head consumes a patch-grid mask, while the operator store persists a
    strict ``[left, top, right, bottom]`` ROI.  Keeping this conversion here
    avoids treating the four rectangle coordinates as a four-pixel mask.
    """
    masks: list[np.ndarray] = []
    for row, image in zip(rows, images, strict=True):
        region = row.get("defect_region")
        if isinstance(region, np.ndarray) and region.ndim >= 2:
            masks.append(np.asarray(region, dtype=np.uint8))
            continue
        if isinstance(region, (list, tuple)) and len(region) == 4:
            values = [int(value) for value in region]
            if isinstance(image, np.ndarray) and image.ndim >= 2:
                height, width = image.shape[:2]
            else:
                declared = row.get("image_size")
                if isinstance(declared, (list, tuple)) and len(declared) == 2:
                    width, height = int(declared[0]), int(declared[1])
                else:
                    # Byte-oriented test/runtime hooks have no decoder.  A
                    # conservative normalized canvas still preserves the ROI
                    # ordering and yields a valid patch supervision mask.
                    width = max(values[2], 1)
                    height = max(values[3], 1)
            mask = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint8)
            left, top, right, bottom = values
            mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 1
            masks.append(mask)
            continue
        # Image-level defect feedback has no localization; all patches are
        # valid positive supervision.
        if isinstance(image, np.ndarray) and image.ndim >= 2:
            masks.append(np.ones(image.shape[:2], dtype=np.uint8))
        else:
            masks.append(np.ones((32, 32), dtype=np.uint8))
    return masks


def _load_raw_feedback_bytes(store: FeedbackStore, rows: Sequence[Mapping[str, Any]]) -> list[bytes]:
    values: list[bytes] = []
    for row in rows:
        reference = row.get("image")
        if not isinstance(reference, str):
            raise ValueError("feedback image reference is missing")
        raw_parts = reference.replace("\\", "/").split("/")
        if any(part in {"", ".", ".."} for part in raw_parts):
            raise PermissionError("feedback image path must be a contained relative path")
        cursor = store.root
        for part in raw_parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PermissionError("feedback image path must not contain a symbolic link")
        path = cursor.resolve(strict=True)
        values.append(path.read_bytes())
    return values


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["feedback_id"]) for row in rows]


def _image_hashes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [_require_sha256(row.get("image_sha256"), "feedback image_sha256") for row in rows]


def _record_hashes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get("record_sha256")
        if value is None:
            unsigned = dict(row)
            unsigned.pop("record_sha256", None)
            value = _canonical_sha256(unsigned)
        values.append(_require_sha256(value, "feedback record_sha256"))
    return values


def _revision_feedback_snapshot(
    *,
    category: str,
    run_id: str,
    bundle_hash: str,
    training: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind only the evidence consumed by one immutable revision.

    Proposal creation separately checks a whole-scope start/end snapshot for
    concurrent edits.  An active revision must instead retain just the rows
    that trained/validated it or supplied its cumulative feature bank, so new
    feedback and unrelated audit records remain usable after activation.
    """

    def bindings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in sorted(rows, key=lambda item: str(item.get("feedback_id", ""))):
            feedback_id = str(row.get("feedback_id", "")).strip()
            if not feedback_id:
                raise ValueError("revision feedback binding has no feedback_id")
            if feedback_id in seen:
                continue
            seen.add(feedback_id)
            result.append(
                {
                    "feedback_id": feedback_id,
                    "image_sha256": _require_sha256(
                        row.get("image_sha256"), "revision image_sha256"
                    ),
                    "record_sha256": _record_hashes([row])[0],
                }
            )
        return result

    snapshot: dict[str, Any] = {
        "category": category,
        "model_version": run_id,
        "bundle_hash": bundle_hash,
        "training": bindings(training),
        "validation": bindings(validation),
    }
    snapshot["sha256"] = _canonical_sha256(snapshot)
    return snapshot


def _prediction_rows(result: Any, sample_id: str) -> dict[str, Any]:
    if hasattr(result, "__dict__") and not isinstance(result, Mapping):
        result = vars(result)
    if not isinstance(result, Mapping):
        result = {"fusion_score": float(result)}
    row = {str(key): _jsonable(value) for key, value in result.items() if not str(key).startswith("_")}
    score = row.get("fusion_score", row.get("score", row.get("decision_score")))
    if score is None:
        raise ValueError("runtime prediction has no fusion score")
    row["fusion_score"] = _finite_float(score, "runtime fusion_score")
    row["decision"] = bool(row.get("decision", row["fusion_score"] >= float(row.get("fusion_threshold", 0.5))))
    row["sample_id"] = sample_id
    return row


def _call_predict(runtime: Any, images: Sequence[Any], ids: Sequence[str]) -> list[dict[str, Any]]:
    byte_predictor = getattr(runtime, "predict_feedback_bytes", None)
    if callable(byte_predictor) and images and isinstance(images[0], (bytes, bytearray)):
        values = byte_predictor(images, ids)
        if isinstance(values, Mapping) or not isinstance(values, Sequence):
            values = [values]
        if len(values) != len(images):
            raise ValueError("runtime prediction count does not match feedback images")
        return [_prediction_rows(value, sample_id) for value, sample_id in zip(values, ids, strict=True)]
    if hasattr(runtime, "predict_batch"):
        try:
            values = runtime.predict_batch(images, ids)
        except TypeError:
            try:
                values = runtime.predict_batch(images, sample_ids=ids)
            except TypeError:
                values = [runtime.predict(image, sample_id) for image, sample_id in zip(images, ids, strict=True)]
    elif hasattr(runtime, "predict"):
        values = [runtime.predict(image, sample_id) for image, sample_id in zip(images, ids, strict=True)]
    else:
        raise TypeError("runtime must expose predict_batch or predict")
    if isinstance(values, Mapping) or not isinstance(values, Sequence):
        values = [values]
    if len(values) != len(images):
        raise ValueError("runtime prediction count does not match feedback images")
    return [_prediction_rows(value, sample_id) for value, sample_id in zip(values, ids, strict=True)]


@contextmanager
def _runtime_override(
    runtime: Any,
    *,
    head: Any | None = None,
    feature_embeddings: np.ndarray | None = None,
    threshold: float | None = None,
):
    """Temporarily apply candidate tensors while preserving runtime state."""
    lock = getattr(runtime, "_feedback_lock", None) or getattr(runtime, "_lock", None)
    acquired = False
    if lock is not None:
        lock.acquire()
        acquired = True
    saved: list[tuple[Any, str, Any]] = []
    try:
        if head is not None and hasattr(runtime, "head"):
            saved.append((runtime, "head", runtime.head))
            runtime.head = head
        dino = getattr(runtime, "dino", None)
        if feature_embeddings is not None and dino is not None and hasattr(dino, "defect_global_bank"):
            saved.append((dino, "defect_global_bank", dino.defect_global_bank))
            base = getattr(runtime, "_feedback_base_defect_bank", None)
            if base is None:
                base = dino.defect_global_bank.detach().clone()
            values = torch.as_tensor(feature_embeddings, device=getattr(dino, "device", "cpu"), dtype=torch.float32)
            values = torch.nn.functional.normalize(values, dim=1)
            base = base.to(values.device)
            dino.defect_global_bank = torch.cat((base, values), dim=0)
        # ``apply_threshold`` is usually performed by the caller; runtimes
        # that expose a mutable feedback threshold can still be overridden.
        if threshold is not None and hasattr(runtime, "feedback_threshold"):
            saved.append((runtime, "feedback_threshold", runtime.feedback_threshold))
            runtime.feedback_threshold = float(threshold)
        yield
    finally:
        for target, name, value in reversed(saved):
            setattr(target, name, value)
        if acquired:
            lock.release()


def _predict_with_candidate(
    runtime: Any,
    images: Sequence[Any],
    ids: Sequence[str],
    *,
    head: Any | None = None,
    feature_embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    # Prefer an explicit override-aware API on test/dedicated runtimes.
    predictor = getattr(runtime, "predict_with_feedback", None)
    if callable(predictor):
        try:
            values = predictor(images, ids, head=head, feature_bank=feature_embeddings)
            return [_prediction_rows(value, sample_id) for value, sample_id in zip(values, ids, strict=True)]
        except TypeError:
            pass
    with _runtime_override(runtime, head=head, feature_embeddings=feature_embeddings):
        return _call_predict(runtime, images, ids)


def _head_features(runtime: Any, images: Sequence[np.ndarray]) -> torch.Tensor | None:
    dino = getattr(runtime, "dino", None)
    extractor = getattr(dino, "_features", None)
    if not callable(extractor):
        return None
    tiled: list[Any] = []
    for image in images:
        tiled.extend(nine_tiles(np.asarray(image)))
    features = extractor(tiled)
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 3 or len(features) % 9:
        raise ValueError("DINO tile extractor must return [N*9,P,D] features")
    return features.reshape(len(images), 9, *features.shape[1:])


def _defect_head_features(
    runtime: Any,
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> tuple[torch.Tensor | None, list[np.ndarray]]:
    """Extract the one ROI-bearing tile used by the original head trainer."""
    dino = getattr(runtime, "dino", None)
    extractor = getattr(dino, "_features", None)
    if not callable(extractor):
        return None, []
    selected = [
        nine_tile_defect(np.asarray(image), np.asarray(mask, dtype=np.uint8))
        for image, mask in zip(images, masks, strict=True)
    ]
    features = extractor([item[0] for item in selected])
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 3 or len(features) != len(images):
        raise ValueError("DINO defect tile extractor must return [N,P,D] features")
    return features, [item[1] for item in selected]


def _defect_feature_crops(
    rows: Sequence[Mapping[str, Any]],
    images: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Use the annotated defect region for the incremental global feature bank."""
    crops: list[np.ndarray] = []
    for row, image in zip(rows, images, strict=True):
        value = np.asarray(image)
        region = row.get("defect_region")
        if isinstance(region, (list, tuple)) and len(region) == 4:
            left, top, right, bottom = [int(item) for item in region]
            crop = value[top:bottom, left:right]
            if crop.size == 0:
                raise ValueError(
                    f"feedback defect ROI is empty for {row.get('feedback_id')}"
                )
            crops.append(crop.copy())
        else:
            crops.append(value.copy())
    return crops


def _global_features(runtime: Any, images: Sequence[np.ndarray]) -> np.ndarray | None:
    dino = getattr(runtime, "dino", None)
    extractor = getattr(dino, "_features", None)
    if not callable(extractor):
        return None
    features = extractor(images)
    if isinstance(features, torch.Tensor):
        values = torch.nn.functional.normalize(features.mean(dim=1), dim=1).detach().cpu().numpy()
    else:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 3:
            values = values.mean(axis=1)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, 1e-12)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("DINO global feedback features are invalid")
    return np.asarray(values, dtype=np.float32)


def _select_threshold(labels: np.ndarray, scores: np.ndarray, minimum_recall: float) -> tuple[float, dict[str, Any]]:
    unique = np.unique(scores)
    if not len(unique):
        raise ValueError("at least one training score is required")
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2.0, [unique[-1], np.nextafter(unique[-1], np.inf)])
    )
    feasible: list[tuple[float, dict[str, Any]]] = []
    for value in candidates:
        if not math.isfinite(float(value)):
            continue
        metrics = classification_metrics(labels, scores, float(value))
        if metrics["recall"] >= minimum_recall:
            feasible.append((float(value), metrics))
    if not feasible:
        raise ValueError("no candidate threshold satisfies minimum recall")
    return min(feasible, key=lambda item: (float(item[1]["fpr"]), -float(item[1]["f1"]), -item[0]))


def _score_records(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(predictions):
        raise ValueError("feedback rows and prediction rows have different lengths")
    labels = np.asarray([int(row.get("operator_label", -1)) for row in rows], dtype=np.uint8)
    scores = np.asarray([_finite_float(prediction.get("fusion_score"), "fusion_score") for prediction in predictions], dtype=np.float64)
    if set(labels.tolist()) - {0, 1} or not np.all(np.isfinite(scores)):
        raise ValueError("feedback labels/scores are invalid")
    metrics = classification_metrics(labels, scores, threshold)
    result: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        item = {
            "feedback_id": str(row["feedback_id"]),
            "image_sha256": _require_sha256(row.get("image_sha256"), "image_sha256"),
            "record_sha256": _require_sha256(row.get("record_sha256"), "record_sha256"),
            "label": int(row["operator_label"]),
            "fusion_score": _finite_float(prediction["fusion_score"], "fusion_score"),
            "decision": bool(float(prediction["fusion_score"]) >= threshold),
            "fusion_threshold": float(threshold),
            "branch_margins": _jsonable(prediction.get("branch_margins", {})),
        }
        result.append(item)
    return result, metrics


def _serialize_head(
    head: Any,
    *,
    parent_fingerprint: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize only a feedback overlay for the production tiled-MLP head.

    A full ``.save()`` artifact would permit a candidate to replace frozen
    normalisation/hidden-layer tensors.  Duck-typed fallbacks remain for small
    test embedders, but the actual competition head always emits this narrow
    overlay format.
    """
    if isinstance(head, (bytes, bytearray)):
        return bytes(head)
    buffer = io.BytesIO()
    overlay_exporter = getattr(head, "output_layer_overlay", None)
    if callable(overlay_exporter):
        overlay = overlay_exporter(parent_fingerprint=dict(parent_fingerprint or {}))
        torch.save(overlay, buffer)
        return buffer.getvalue()
    saver = getattr(head, "save", None)
    if callable(saver):
        with tempfile.TemporaryDirectory(prefix="aoi-feedback-head-") as directory:
            path = Path(directory) / "head.pt"
            try:
                saver(path)
                return path.read_bytes()
            except (AttributeError, KeyError, RuntimeError, TypeError):
                # Minimal/fake heads may not carry calibration-only fields;
                # their tensor overlay is still serializable and auditable.
                pass
    state = getattr(head, "state_dict", None)
    if callable(state):
        torch.save(state(), buffer)
        return buffer.getvalue()
    torch.save(head, buffer)
    return buffer.getvalue()


def _head_fingerprint(head: Any) -> dict[str, Any]:
    method = getattr(head, "parameter_fingerprint", None)
    if callable(method):
        value = method()
        return _jsonable(value) if isinstance(value, Mapping) else {"overall_sha256": str(value)}
    values: dict[str, Any] = {}
    for name in ("feature_mean", "feature_scale", "weight1", "bias1", "weight2", "bias2"):
        tensor = getattr(head, name, None)
        if tensor is None:
            continue
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        values[name] = hashlib.sha256(raw).hexdigest()
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    values["overall_sha256"] = hashlib.sha256(encoded).hexdigest()
    return values


def _clone_head(head: Any) -> Any:
    clone = getattr(head, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(head)


def _revision_feature_values(
    store: FeedbackStore, revision: Mapping[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """Read a parent revision's feature artifact with its recorded hash."""
    payload = revision.get("payload") if isinstance(revision.get("payload"), Mapping) else revision
    artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else None
    entry = artifacts.get("feature_bank") if isinstance(artifacts, Mapping) else None
    if not isinstance(entry, Mapping):
        return np.empty((0, 0), dtype=np.float32), []
    checker = getattr(store, "_safe_revision_path", None)
    if callable(checker):
        path = checker(entry.get("path"), must_exist=True)
    else:
        raw = entry.get("path")
        if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise PermissionError("revision feature path must be relative")
        path = (store.root / raw).resolve(strict=True)
        try:
            path.relative_to(store.root.resolve(strict=True))
        except ValueError as error:
            raise PermissionError("revision feature path escapes feedback root") from error
    expected = str(entry.get("sha256", "")).lower()
    actual = _sha256_bytes(path.read_bytes())
    if len(expected) != 64 or actual != expected:
        raise ValueError("parent revision feature artifact SHA-256 mismatch")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            values = np.asarray(loaded["embeddings"], dtype=np.float32)
            ids = [str(value) for value in loaded["sample_ids"].tolist()]
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError("parent revision feature artifact is malformed") from error
    if values.ndim != 2 or len(values) != len(ids) or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("parent revision feature artifact is malformed")
    return values, ids


def propose_adaptive_revision(
    runtime: Any,
    store: FeedbackStore,
    category: str,
    *,
    training_feedback_ids: Sequence[str] | None = None,
    validation_feedback_ids: Sequence[str] | None = None,
    run_id: str | None = None,
    bundle_hash: str | None = None,
    minimum_recall: float = DEFAULT_MINIMUM_RECALL,
    minimum_training_samples: int = 6,
    minimum_validation_samples: int = 4,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gradient_clip: float = DEFAULT_GRADIENT_CLIP,
    l2_sp: float = DEFAULT_L2_SP,
    seed: int | None = 20260830,
    head: Any | None = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a complete, independently validated adaptive revision.

    ``runtime`` and ``store`` are deliberately duck-typed so the same flow is
    usable by the desktop UI, CLI, and deterministic unit-test runtimes.
    """
    category = str(category).strip()
    if not category:
        raise ValueError("category is required")
    run_id = str(run_id or runtime_model_version(runtime)).strip()
    if not run_id:
        raise ValueError("run_id is required")
    bundle_hash = _require_sha256(bundle_hash or _bundle_hash(runtime), "bundle_hash")
    minimum_recall = _finite_float(minimum_recall, "minimum_recall")
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")

    # Proposal always starts from the complete state currently authorized by
    # the registry.  This is deliberately separate from feedback submission:
    # merely writing a record cannot mutate a runtime or create a revision.
    if callable(getattr(store, "active_revision", None)):
        try:
            apply_active_feedback(
                runtime,
                store,
                category,
                enabled=True,
                model_version=run_id,
            )
        except TypeError:
            # Lightweight compatibility stores may expose an older method
            # signature; their caller is responsible for preparing state.
            pass

    all_training = store.records(category, "training", run_id)
    all_validation = store.records(category, "validation", run_id)
    expected_bundle = bundle_hash
    for row in [*all_training, *all_validation]:
        declared_bundle = row.get("bundle_hash")
        if declared_bundle not in {None, "", expected_bundle}:
            raise PermissionError(
                f"feedback {row.get('feedback_id')} belongs to a different model bundle"
            )
    training = _eligible_rows(all_training, role="training", selected_ids=training_feedback_ids)
    validation = _eligible_rows(all_validation, role="validation", selected_ids=validation_feedback_ids)
    if len(training) < int(minimum_training_samples):
        raise ValueError(f"at least {minimum_training_samples} appearance training feedback samples are required")
    if len(validation) < int(minimum_validation_samples):
        raise ValueError(f"at least {minimum_validation_samples} independent validation samples are required")
    if sum(int(row.get("operator_label", -1)) == 0 for row in training) < 2 or sum(int(row.get("operator_label", -1)) == 1 for row in training) < 2:
        raise ValueError("training feedback requires at least two normal and two defect samples")
    if sum(int(row.get("operator_label", -1)) == 0 for row in validation) < 2 or sum(int(row.get("operator_label", -1)) == 1 for row in validation) < 2:
        raise ValueError("independent validation requires at least two normal and two defect samples")
    training_hashes = _image_hashes(training)
    validation_hashes = _image_hashes(validation)
    if set(training_hashes) & set(validation_hashes):
        raise PermissionError("FEEDBACK_TRAIN_VALIDATION_LEAKAGE: training and validation image content overlaps")
    training_records = _record_hashes(training)
    validation_records = _record_hashes(validation)

    # Lock the binding before decoding/reasoning.  A second snapshot is checked
    # by save/activation, protecting against TOCTOU edits during a long GPU run.
    with store._records_guard():
        start_snapshot = store._feedback_snapshot_unlocked(category, run_id, bundle_hash)

    raw_training = _load_raw_feedback_bytes(store, training)
    raw_validation = _load_raw_feedback_bytes(store, validation)
    try:
        training_images = _read_feedback_images(store, training)
        validation_images = _read_feedback_images(store, validation)
    except ValueError:
        # If the runtime supplies a byte-oriented prediction hook, preserve the
        # bytes; otherwise surface the decoder error.
        if not callable(getattr(runtime, "predict_feedback_bytes", None)):
            raise
        training_images = raw_training  # type: ignore[assignment]
        validation_images = raw_validation  # type: ignore[assignment]

    parent_predictions = _call_predict(runtime, training_images + validation_images, _ids(training) + _ids(validation))
    parent_training_predictions = parent_predictions[: len(training)]
    parent_validation_predictions = parent_predictions[len(training) :]

    parent_revision = None
    if callable(getattr(store, "active_revision", None)):
        try:
            parent_revision = store.active_revision(category, run_id, bundle_hash, verify=True)
        except FileNotFoundError:
            parent_revision = None
    parent_revision_id = None if parent_revision is None else str(parent_revision.get("revision_id"))
    parent_threshold = None
    if parent_revision is not None:
        parent_threshold = _finite_float(parent_revision.get("threshold"), "parent revision threshold")
    if parent_threshold is None:
        values = [prediction.get("fusion_threshold") for prediction in parent_predictions if prediction.get("fusion_threshold") is not None]
        parent_threshold = _finite_float(values[0] if values else 0.5, "parent threshold")

    labels_train = np.asarray([int(row["operator_label"]) for row in training], dtype=np.uint8)
    parent_train_scores = np.asarray([float(row["fusion_score"]) for row in parent_training_predictions], dtype=np.float64)
    parent_val_scores = np.asarray([float(row["fusion_score"]) for row in parent_validation_predictions], dtype=np.float64)
    parent_train_items, parent_train_metrics = _score_records(training, parent_training_predictions, threshold=parent_threshold)
    parent_val_items, parent_val_metrics = _score_records(validation, parent_validation_predictions, threshold=parent_threshold)

    parent_head = head or getattr(runtime, "head", None)
    if parent_head is None:
        raise RuntimeError("runtime does not expose a DinoTiledMLPKnownHead")
    candidate_head = _clone_head(parent_head)
    before_fingerprint = _head_fingerprint(parent_head)
    candidate_defect_rows = [row for row in training if int(row.get("operator_label", -1)) == 1]
    defect_indices = [training.index(row) for row in candidate_defect_rows]
    normal_indices = [index for index, row in enumerate(training) if int(row.get("operator_label", -1)) == 0]
    normal_images = [training_images[index] for index in normal_indices]
    defect_images = [training_images[index] for index in defect_indices]
    full_defect_masks = _feedback_roi_masks(candidate_defect_rows, defect_images)
    defect_tile_masks: list[np.ndarray] = []
    try:
        normal_features = _head_features(runtime, normal_images)
        defect_features, defect_tile_masks = _defect_head_features(
            runtime, defect_images, full_defect_masks
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        normal_features = None
        defect_features = None
    if normal_features is None or defect_features is None:
        # A fake head may implement its own feature-independent fine-tuning.
        tuner = getattr(candidate_head, "fine_tune_feedback_output_layer", None)
        if not callable(tuner):
            raise RuntimeError("runtime cannot extract DINO tile features for feedback tuning")
        tuning = tuner(labels=labels_train, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, gradient_clip=gradient_clip, l2_sp=l2_sp, seed=seed)
    else:
        tuning = candidate_head.fine_tune_feedback_output_layer(
            normal_features,
            defect_features,
            masks=defect_tile_masks,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            l2_sp=l2_sp,
            seed=seed,
        )
    after_fingerprint = _head_fingerprint(candidate_head)
    changed_count = int(tuning.get("updated_parameter_count", 0)) if isinstance(tuning, Mapping) else 0
    if changed_count != 33:
        raise ValueError(f"feedback head must update exactly 33 output parameters; got {changed_count}")
    for name in ("feature_mean", "feature_scale", "weight1", "bias1"):
        if before_fingerprint.get(name) != after_fingerprint.get(name):
            raise ValueError(f"frozen head tensor changed during feedback tuning: {name}")

    try:
        candidate_global = _global_features(
            runtime, _defect_feature_crops(candidate_defect_rows, defect_images)
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        candidate_global = None
    if candidate_global is None:
        # A runtime can expose a precomputed feature hook for lightweight tests.
        hook = getattr(runtime, "feedback_feature_bank", None)
        candidate_global = None if not callable(hook) else np.asarray(hook(candidate_defect_rows), dtype=np.float32)
    if candidate_global is None:
        raise RuntimeError("runtime cannot extract DINO global feedback features")
    if candidate_global.ndim != 2 or len(candidate_global) != len(candidate_defect_rows):
        raise ValueError("feedback feature bank count does not match defect feedback")

    # Revisions carry the cumulative defect bank.  This keeps a second update
    # from silently dropping features learned by its parent while still making
    # rollback a single-pointer operation.
    candidate_feature_ids = _ids(candidate_defect_rows)
    if parent_revision is not None:
        parent_values, parent_ids = _revision_feature_values(store, parent_revision)
        if len(parent_ids):
            if parent_values.shape[1] != candidate_global.shape[1]:
                raise ValueError("parent and candidate feature dimensions do not match")
            merged_values: list[np.ndarray] = [parent_values]
            merged_ids: list[str] = list(parent_ids)
            existing = set(parent_ids)
            new_rows = [index for index, value in enumerate(candidate_feature_ids) if value not in existing]
            if new_rows:
                merged_values.append(candidate_global[new_rows])
                merged_ids.extend(candidate_feature_ids[index] for index in new_rows)
            candidate_global = np.concatenate(merged_values, axis=0)
            candidate_feature_ids = merged_ids

    candidate_predictions = _predict_with_candidate(
        runtime,
        training_images + validation_images,
        _ids(training) + _ids(validation),
        head=candidate_head,
        feature_embeddings=candidate_global,
    )
    candidate_training_predictions = candidate_predictions[: len(training)]
    candidate_validation_predictions = candidate_predictions[len(training) :]
    candidate_train_scores = np.asarray([float(row["fusion_score"]) for row in candidate_training_predictions], dtype=np.float64)
    candidate_threshold, candidate_train_metrics = _select_threshold(labels_train, candidate_train_scores, minimum_recall)
    candidate_train_items, _ = _score_records(training, candidate_training_predictions, threshold=candidate_threshold)
    candidate_val_items, candidate_val_metrics = _score_records(validation, candidate_validation_predictions, threshold=candidate_threshold)
    validation_acceptance = _revision_validation_acceptance(
        parent_val_metrics,
        candidate_val_metrics,
        minimum_recall,
    )
    if not validation_acceptance["passed"]:
        recall_delta = float(validation_acceptance["recall_delta"]) * 100.0
        f1_delta = float(validation_acceptance["f1_delta"]) * 100.0
        raise ValueError(
            "候选独立验证未改善："
            f"召回率 {float(parent_val_metrics['recall']):.1%} → "
            f"{float(candidate_val_metrics['recall']):.1%}（{recall_delta:+.1f} 个百分点）；"
            f"F1 {float(parent_val_metrics['f1']):.1%} → "
            f"{float(candidate_val_metrics['f1']):.1%}（{f1_delta:+.1f} 个百分点）。"
            f"成功条件为达到 {minimum_recall:.1%} 召回率目标且 F1 不下降，"
            "或召回率/F1 均不下降且至少一项提升。"
        )

    # Add parent/candidate values to each immutable sample row.  These values
    # are recomputed at proposal time and never copied from stored feedback.
    training_samples: list[dict[str, Any]] = []
    validation_samples: list[dict[str, Any]] = []
    for parent_item, candidate_item in zip(parent_train_items, candidate_train_items, strict=True):
        training_samples.append({
            "feedback_id": parent_item["feedback_id"],
            "image_sha256": parent_item["image_sha256"],
            "record_sha256": parent_item["record_sha256"],
            "label": parent_item["label"],
            "parent_score": parent_item["fusion_score"],
            "candidate_score": candidate_item["fusion_score"],
            "parent_decision": parent_item["decision"],
            "candidate_decision": candidate_item["decision"],
            "parent_branch_margins": parent_item.get("branch_margins", {}),
            "candidate_branch_margins": candidate_item.get("branch_margins", {}),
        })
    for parent_item, candidate_item in zip(parent_val_items, candidate_val_items, strict=True):
        validation_samples.append({
            "feedback_id": parent_item["feedback_id"],
            "image_sha256": parent_item["image_sha256"],
            "record_sha256": parent_item["record_sha256"],
            "label": parent_item["label"],
            "parent_score": parent_item["fusion_score"],
            "candidate_score": candidate_item["fusion_score"],
            "parent_decision": parent_item["decision"],
            "candidate_decision": candidate_item["decision"],
            "parent_branch_margins": parent_item.get("branch_margins", {}),
            "candidate_branch_margins": candidate_item.get("branch_margins", {}),
        })
    # Cumulative feature banks may reference defect evidence from the parent
    # revision even when this proposal selects only newly added rows.
    row_by_id = {
        str(row.get("feedback_id")): row
        for row in all_training
        if str(row.get("optimization_target", "appearance")) != "audit_only"
    }
    feature_sample_hashes = [
        _require_sha256(row_by_id[feature_id].get("image_sha256"), "feature image_sha256")
        for feature_id in candidate_feature_ids
        if feature_id in row_by_id
    ]
    if len(feature_sample_hashes) != len(candidate_feature_ids):
        raise ValueError("feature bank IDs do not resolve to training feedback")
    bound_training_by_id = {
        str(row["feedback_id"]): row for row in training
    }
    for feature_id in candidate_feature_ids:
        bound_training_by_id.setdefault(feature_id, row_by_id[feature_id])
    revision_snapshot = _revision_feedback_snapshot(
        category=category,
        run_id=run_id,
        bundle_hash=bundle_hash,
        training=list(bound_training_by_id.values()),
        validation=validation,
    )
    candidate_payload: dict[str, Any] = {
        "revision_id": f"revision-{uuid4().hex}",
        "category": category,
        "run_id": run_id,
        "bundle_hash": bundle_hash,
        "parent_revision_id": parent_revision_id,
        "threshold": float(candidate_threshold),
        "candidate_threshold": float(candidate_threshold),
        "parent_threshold": float(parent_threshold),
        "feedback_snapshot": revision_snapshot,
        "base_metrics": parent_train_metrics,
        "candidate_metrics": candidate_train_metrics,
        "training": {
            "sample_count": len(training),
            "normal_count": int(np.sum(labels_train == 0)),
            "defect_count": int(np.sum(labels_train == 1)),
            "sample_ids": _ids(training),
            "sample_sha256": training_hashes,
            "sample_record_sha256": training_records,
            "samples": training_samples,
        },
        "validation": {
            **validation_acceptance,
            "minimum_recall": float(minimum_recall),
            "parent_threshold": float(parent_threshold),
            "sample_count": len(validation),
            "normal_count": sum(int(row["operator_label"]) == 0 for row in validation),
            "defect_count": sum(int(row["operator_label"]) == 1 for row in validation),
            "sample_ids": _ids(validation),
            "sample_sha256": validation_hashes,
            "sample_record_sha256": validation_records,
            "parent_metrics": parent_val_metrics,
            "base_metrics": parent_val_metrics,
            "candidate_metrics": candidate_val_metrics,
            "samples": validation_samples,
        },
        "head_parameter_fingerprint": {
            "parent": before_fingerprint,
            "candidate": after_fingerprint,
            "updated_parameter_count": changed_count,
            "trainable_parameters": ["weight2", "bias2"],
        },
        "feature_bank": {
            "sample_ids": candidate_feature_ids,
            "sample_sha256": feature_sample_hashes,
            "embedding_dimension": int(candidate_global.shape[1]),
            "sample_count": int(len(candidate_global)),
        },
        "optimization": {
            "method": "human_feedback_supervised_incremental_output_layer_finetune",
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip": float(gradient_clip),
            "l2_sp": float(l2_sp),
            "seed": seed,
            "historical_fusion_scores_used": False,
        },
        "metadata": {
            "parent_training_metrics": parent_train_metrics,
            "candidate_training_metrics": candidate_train_metrics,
            "parent_validation_metrics": parent_val_metrics,
            "candidate_validation_metrics": candidate_val_metrics,
        },
    }
    if dry_run:
        candidate_payload["head_bytes"] = _serialize_head(
            candidate_head, parent_fingerprint=before_fingerprint
        )
        candidate_payload["feature_embeddings"] = candidate_global
        candidate_payload["feature_sample_ids"] = candidate_feature_ids
        return candidate_payload

    # Recheck the exact evidence binding immediately before writing artifacts.
    with store._records_guard():
        end_snapshot = store._feedback_snapshot_unlocked(category, run_id, bundle_hash)
    if end_snapshot != start_snapshot:
        raise RuntimeError("feedback records changed while generating revision candidate")
    result = store.save_revision_candidate(
        candidate_payload,
        head_artifact=_serialize_head(
            candidate_head, parent_fingerprint=before_fingerprint
        ),
        feature_embeddings=candidate_global,
        feature_sample_ids=candidate_feature_ids,
        feedback_snapshot=revision_snapshot,
        validation=candidate_payload["validation"],
        training=candidate_payload["training"],
        threshold=float(candidate_threshold),
        metadata={
            **candidate_payload["metadata"],
            "optimization": candidate_payload["optimization"],
            "head_parameter_fingerprint": candidate_payload["head_parameter_fingerprint"],
            "feature_bank": candidate_payload["feature_bank"],
        },
    )
    # Keep the high-signal fields at the top level for CLI/UI consumers.
    result.update(
        {
            "candidate_threshold": float(candidate_threshold),
            "base_metrics": parent_train_metrics,
            "candidate_metrics": candidate_train_metrics,
            "validation_metrics": candidate_val_metrics,
            "parent_validation_metrics": parent_val_metrics,
            "validation_acceptance": validation_acceptance,
            "updated_parameter_count": changed_count,
            "parameter_fingerprint": candidate_payload["head_parameter_fingerprint"],
            "feature_count": int(len(candidate_global)),
            "historical_fusion_scores_used": False,
        }
    )
    return result


class FeedbackOptimizer:
    """Small stateful facade used by the desktop UI and embedders."""

    def __init__(self, runtime: Any, store: FeedbackStore, category: str, **defaults: Any) -> None:
        self.runtime = runtime
        self.store = store
        self.category = category
        self.defaults = dict(defaults)

    def propose(self, **kwargs: Any) -> dict[str, Any]:
        options = {**self.defaults, **kwargs}
        return propose_adaptive_revision(self.runtime, self.store, self.category, **options)

    def activate(self, candidate: Mapping[str, Any] | str) -> dict[str, Any]:
        return self.store.activate_revision(candidate)

    def rollback(self, **kwargs: Any) -> dict[str, Any] | None:
        return self.store.rollback_revision(self.category, **kwargs)

    def status(self, **kwargs: Any) -> dict[str, Any] | None:
        run_id = kwargs.get("run_id") or runtime_model_version(self.runtime)
        bundle_hash = kwargs.get("bundle_hash") or _bundle_hash(self.runtime)
        return self.store.active_revision(self.category, run_id, bundle_hash)


# Public aliases used by integrations that prefer verb-first naming.
propose_feedback_revision = propose_adaptive_revision
generate_revision_candidate = propose_adaptive_revision
