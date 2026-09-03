"""Auditable active-learning sample queries for operator feedback.

The query stage is deliberately label-free and read-only with respect to the
model. It ranks an unlabeled image pool, writes a manifest for the operator,
and leaves training/validation/activation to the existing revision workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import yaml

from aoi.images import read_rgb


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_QUERY_BUDGET = 20
DEFAULT_QUERY_WEIGHTS = {
    "uncertainty": 0.55,
    "branch_disagreement": 0.20,
    "rarity": 0.25,
}


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (RuntimeError, ValueError):
            pass
    return value


def _normalise_embeddings(value: object, expected_count: int) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=1)
    if array.ndim != 2 or len(array) != expected_count:
        raise ValueError("candidate embeddings must have shape [N, D]")
    if not np.all(np.isfinite(array)):
        raise ValueError("candidate embeddings must be finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("candidate embeddings must be non-zero")
    return array / norms


def _branch_disagreement(row: Mapping[str, Any]) -> float:
    decisions = row.get("branch_decisions")
    if isinstance(decisions, Mapping) and decisions:
        values = np.asarray([bool(value) for value in decisions.values()], dtype=np.float32)
        fraction = float(values.mean())
        return float(np.clip(4.0 * fraction * (1.0 - fraction), 0.0, 1.0))
    margins = row.get("branch_margins")
    if not isinstance(margins, Mapping):
        return 0.0
    values: list[float] = []
    for value in margins.values():
        try:
            item = _finite(value, "branch margin")
        except ValueError:
            continue
        values.append(item)
    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    spread = float(np.std(array))
    scale = float(np.mean(np.abs(array))) + 1e-12
    return float(np.clip(spread / (spread + scale), 0.0, 1.0))


def _weights(value: Mapping[str, Any] | None) -> dict[str, float]:
    result = dict(DEFAULT_QUERY_WEIGHTS)
    if value is not None:
        for key in result:
            if key in value:
                result[key] = _finite(value[key], f"query weight {key}")
    if any(item < 0 for item in result.values()) or sum(result.values()) <= 0:
        raise ValueError("query weights must be non-negative with a positive sum")
    total = sum(result.values())
    return {key: item / total for key, item in result.items()}


def rank_candidates(
    predictions: Sequence[Mapping[str, Any]],
    *,
    budget: int = DEFAULT_QUERY_BUDGET,
    threshold: float | None = None,
    candidate_embeddings: object = None,
    existing_embeddings: object = None,
    weights: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rank unlabeled predictions and select a diverse top-budget batch.

    ``predictions`` may contain runtime fields such as ``fusion_score``,
    ``fusion_threshold``, ``branch_margins``, ``path`` and ``image_sha256``.
    The returned rows include component scores and a human-readable reason.
    """
    if isinstance(budget, bool) or int(budget) <= 0:
        raise ValueError("query budget must be a positive integer")
    rows = [dict(row) for row in predictions]
    if not rows:
        return []
    scores = np.asarray(
        [_finite(row.get("fusion_score", row.get("score")), "fusion_score") for row in rows],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        [
            _finite(
                threshold if threshold is not None else row.get("fusion_threshold", 0.5),
                "fusion_threshold",
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    gaps = np.abs(scores - thresholds)
    scale = max(float(np.ptp(scores)), float(np.median(gaps)), 1e-6)
    uncertainty = np.exp(-gaps / scale)
    branch = np.asarray([_branch_disagreement(row) for row in rows], dtype=np.float64)
    candidates = _normalise_embeddings(candidate_embeddings, len(rows))
    existing = None
    if existing_embeddings is not None:
        shape = getattr(existing_embeddings, "shape", None)
        expected_existing = int(shape[0]) if shape is not None else len(existing_embeddings)
        existing = _normalise_embeddings(existing_embeddings, expected_existing)
        if existing is not None and candidates is not None and existing.shape[1] != candidates.shape[1]:
            raise ValueError("candidate and existing embedding dimensions differ")
    rarity = np.zeros(len(rows), dtype=np.float64)
    if candidates is not None:
        if existing is not None and len(existing):
            rarity = 1.0 - np.max(candidates @ existing.T, axis=1)
        elif len(candidates) > 1:
            similarity = candidates @ candidates.T
            np.fill_diagonal(similarity, -1.0)
            rarity = 1.0 - np.max(similarity, axis=1)
        rarity = np.clip(rarity, 0.0, 1.0)
    normalised_weights = _weights(weights)
    base = (
        normalised_weights["uncertainty"] * uncertainty
        + normalised_weights["branch_disagreement"] * branch
        + normalised_weights["rarity"] * rarity
    )
    for index, row in enumerate(rows):
        row["uncertainty_score"] = float(uncertainty[index])
        row["branch_disagreement_score"] = float(branch[index])
        row["rarity_score"] = float(rarity[index])
        row["query_score"] = float(base[index])
        reasons: list[str] = []
        if uncertainty[index] >= 0.5:
            reasons.append("threshold_near")
        if branch[index] >= 0.25:
            reasons.append("branch_disagreement")
        if rarity[index] >= 0.5:
            reasons.append("feature_rare")
        row["query_reason"] = reasons or ["highest_composite_score"]
        row["query_status"] = "pending_label"

    order = sorted(range(len(rows)), key=lambda index: (-base[index], str(rows[index].get("sample_id", index))))
    selected: list[int] = []
    budget = min(int(budget), len(rows))
    while len(selected) < budget:
        remaining = [index for index in order if index not in selected]
        if not remaining:
            break
        best = remaining[0]
        best_value = -math.inf
        for index in remaining:
            diversity = 0.0
            if candidates is not None and selected:
                diversity = float(np.min(1.0 - candidates[index] @ candidates[selected].T))
                diversity = float(np.clip(diversity, 0.0, 1.0))
            value = 0.75 * float(base[index]) + 0.25 * diversity
            if value > best_value:
                best = index
                best_value = value
        rows[best]["selection_score"] = float(best_value)
        selected.append(best)
    result = [rows[index] for index in selected]
    for rank, row in enumerate(result, start=1):
        row["query_rank"] = rank
    return result


def _pool_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().casefold(),
    )


def _existing_image_hashes(store: Any, category: str, model_version: str) -> set[str]:
    hashes: set[str] = set()
    for row in store.records(category, model_version=model_version):
        value = row.get("image_sha256")
        if isinstance(value, str) and len(value) == 64:
            hashes.add(value.lower())
    return hashes


def _save_query_manifest(store: Any, payload: Mapping[str, Any], output: str | Path | None) -> Path:
    root = Path(store.root).resolve()
    if output is None:
        path = root / "queries" / f"{payload['query_id']}.yaml"
    else:
        path = Path(output)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PermissionError("active-learning query output must be inside feedback root") from error
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"active-learning query manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    unsigned["manifest_sha256"] = _canonical_sha256(unsigned)
    text = yaml.safe_dump(unsigned, sort_keys=False, allow_unicode=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def query_unlabeled_pool(
    runtime: Any,
    store: Any,
    category: str,
    input_dir: str | Path,
    *,
    budget: int = DEFAULT_QUERY_BUDGET,
    model_version: str | None = None,
    output: str | Path | None = None,
    weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score an unlabeled image directory and persist a query manifest."""
    version = model_version or str(getattr(runtime, "bundle", {}).get("run_id", ""))
    if not version:
        version = Path(getattr(runtime, "bundle_path", "model")).parent.name
    paths = _pool_paths(input_dir)
    known_hashes = _existing_image_hashes(store, category, version)
    seen_pool_hashes: set[str] = set()
    images: list[np.ndarray] = []
    candidates: list[dict[str, Any]] = []
    for path in paths:
        data = path.read_bytes()
        image_hash = _sha256_bytes(data)
        if image_hash in known_hashes or image_hash in seen_pool_hashes:
            continue
        seen_pool_hashes.add(image_hash)
        images.append(read_rgb(path))
        candidates.append(
            {
                "sample_id": f"{category}/{image_hash[:16]}/{path.name}",
                "path": str(path),
                "original_name": path.name,
                "image_sha256": image_hash,
            }
        )
    if not candidates:
        raise ValueError("unlabeled query pool is empty after excluding recorded image content")
    ids = [str(row["sample_id"]) for row in candidates]
    predictions = runtime.predict_batch(images, ids)
    if len(predictions) != len(candidates):
        raise ValueError("runtime prediction count does not match query pool")
    for candidate, prediction in zip(candidates, predictions, strict=True):
        if hasattr(prediction, "__dict__") and not isinstance(prediction, Mapping):
            prediction = vars(prediction)
        if not isinstance(prediction, Mapping):
            prediction = {"fusion_score": prediction}
        candidate.update(
            {
                str(key): _jsonable(value)
                for key, value in prediction.items()
                if not str(key).startswith("_")
            }
        )
    embeddings = None
    dino = getattr(runtime, "dino", None)
    extractor = getattr(dino, "_features", None)
    if callable(extractor):
        embeddings = _normalise_embeddings(extractor(images), len(images))
    existing_embeddings = None
    if dino is not None:
        banks: list[np.ndarray] = []
        for name in ("normal_global_bank", "defect_global_bank"):
            value = getattr(dino, name, None)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 2 and len(array):
                banks.append(array)
        if banks:
            existing_embeddings = np.concatenate(banks, axis=0)
    selected = rank_candidates(
        candidates,
        budget=budget,
        candidate_embeddings=embeddings,
        existing_embeddings=existing_embeddings,
        weights=weights,
    )
    bundle_hash = None
    bundle = getattr(runtime, "bundle", None)
    if isinstance(bundle, Mapping) and isinstance(bundle.get("bundle_hash"), str):
        bundle_hash = bundle.get("bundle_hash")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "query_id": f"query-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid4().hex[:8]}",
        "query_type": "operator_active_learning_pool",
        "category": category,
        "run_id": version,
        "bundle_hash": bundle_hash,
        "input_dir": str(Path(input_dir).resolve()),
        "pool_count": len(candidates),
        "selected_count": len(selected),
        "budget": int(budget),
        "weights": _weights(weights),
        "selection_policy": "uncertainty + branch disagreement + feature rarity + greedy cosine diversity",
        "candidates": selected,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _save_query_manifest(store, payload, output)
    result = dict(payload)
    result["manifest_path"] = path.relative_to(Path(store.root).resolve()).as_posix()
    result["manifest_sha256"] = _canonical_sha256(payload)
    return result


__all__ = ["DEFAULT_QUERY_BUDGET", "DEFAULT_QUERY_WEIGHTS", "IMAGE_SUFFIXES", "query_unlabeled_pool", "rank_candidates"]
