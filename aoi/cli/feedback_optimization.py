"""Command-line entry point for the complete operator-feedback revision flow.

The command intentionally keeps feedback submission out of this module.  A
submission only writes evidence; ``propose`` is the explicit operation that
re-runs the selected images and persists a complete head/feature/threshold
candidate.  ``activate`` and ``rollback`` are registry pointer operations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from aoi.feedback import FeedbackOptimizer, FeedbackStore, apply_active_feedback
from aoi.feedback.active_learning import query_unlabeled_pool
from aoi.feedback.optimization import _bundle_hash, propose_adaptive_revision
from aoi.paths import PROJECT_ROOT, resolve_artifact
from aoi.runtime.config import resolve_device
from aoi.runtime.runtime import AdaptedRuntime


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a YAML mapping")
    return dict(value)


def _config_payload(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve(strict=True)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config_path, _mapping(payload, "feedback config")


def _resolve_config_path(config_path: Path, value: object, context: str, *, strict: bool = True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError(f"{context} must be a non-empty path")
    raw = Path(value)
    if raw.is_absolute():
        candidates = [raw]
    else:
        # Repository configs live in ``<project>/configs``; ad-hoc callers
        # commonly place a config beside their temporary project.  Prefer the
        # inferred project root, then the config directory for explicit local
        # references.
        project_base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        candidates = [project_base / raw, config_path.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() or not strict:
            if strict and not resolved.exists():
                continue
            return resolved
    raise FileNotFoundError(f"{context} does not exist: {value}")


def _scope(config_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    root_value = payload.get("feedback_root", payload.get("store_root", "outputs/feedback"))
    feedback_root = _resolve_config_path(config_path, root_value, "feedback_root", strict=False)
    pointer_value = payload.get(
        "active_models_pointer",
        payload.get("models_pointer", payload.get("pointer")),
    )
    pointer_path: Path | None = None
    pointer: dict[str, Any] = {}
    if pointer_value not in {None, ""}:
        pointer_path = _resolve_config_path(config_path, pointer_value, "active_models_pointer")
        pointer = _mapping(
            yaml.safe_load(pointer_path.read_text(encoding="utf-8")) or {},
            "active_models pointer",
        )
    models = pointer.get("models", {})
    if not isinstance(models, Mapping):
        raise TypeError("active_models_pointer.models must be a mapping")
    category_value = payload.get("category")
    if category_value in {None, ""}:
        if len(models) != 1:
            raise ValueError("category is required when the model pointer has multiple categories")
        category = str(next(iter(models)))
    else:
        category = str(category_value).strip()
        if not category:
            raise ValueError("category must be non-empty")
    bundle_path: Path | None = None
    bundle: dict[str, Any] = {}
    if category in models:
        base = pointer_path.parent if pointer_path is not None else PROJECT_ROOT
        bundle_path = resolve_artifact(models[category], base=base)
        bundle = _mapping(
            yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {},
            "model bundle",
        )
    raw_run = payload.get("run_id", payload.get("model_version", bundle.get("run_id")))
    if raw_run in {None, ""} and bundle_path is not None:
        raw_run = bundle_path.parent.name
    run_id = str(raw_run or "").strip()
    if not run_id:
        raise ValueError("run_id/model_version is required")
    raw_bundle_hash = payload.get("bundle_hash", bundle.get("bundle_hash"))
    if raw_bundle_hash not in {None, "", "pending"}:
        bundle_hash = str(raw_bundle_hash).strip().lower()
    else:
        bundle_hash = None
    return {
        "feedback_root": feedback_root,
        "pointer_path": pointer_path,
        "pointer": pointer,
        "category": category,
        "run_id": run_id,
        "bundle_hash": bundle_hash,
        "bundle_path": bundle_path,
        "bundle": bundle,
        "device": str(payload.get("device", "auto")),
    }


def _bundle_scope(runtime: Any, scope: dict[str, Any]) -> str:
    value = scope.get("bundle_hash")
    if value:
        if len(str(value)) != 64 or any(ch not in "0123456789abcdef" for ch in str(value).lower()):
            raise ValueError("bundle_hash must be a SHA-256 string")
        return str(value).lower()
    return _bundle_hash(runtime)


def _ids(value: object) -> list[str] | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return values
        return [item.strip() for item in value.split(",") if item.strip()]
    raise TypeError("feedback ID selection must be a list, comma-separated string, or text file")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _print(value: object) -> None:
    print(yaml.safe_dump(_jsonable(value), sort_keys=False, allow_unicode=True))


def _candidate_reference(store: FeedbackStore, value: object) -> Mapping[str, Any] | str:
    if isinstance(value, Mapping):
        return value
    if value in {None, ""}:
        raise ValueError("activate requires a candidate revision ID or receipt path")
    text = str(value)
    path = Path(text)
    if path.is_file():
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(store.root.resolve(strict=True)).as_posix()
        except ValueError as error:
            raise PermissionError("candidate receipt must be inside feedback store") from error
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise TypeError("candidate receipt must be a YAML mapping")
        revision_id = str(payload.get("revision_id", "")).strip()
        if not revision_id:
            raise ValueError("candidate receipt has no revision_id")
        import hashlib

        return {
            "revision_id": revision_id,
            "receipt_path": relative,
            "receipt_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }
    return text


def _run_status(config_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    scope = _scope(config_path, payload)
    store = FeedbackStore(scope["feedback_root"])
    bundle_hash = scope.get("bundle_hash")
    active = store.active_revision(
        scope["category"], scope["run_id"], bundle_hash, verify=True
    ) if bundle_hash else store.active_revision(scope["category"], scope["run_id"], verify=True)
    return {
        "command": "status",
        "category": scope["category"],
        "run_id": scope["run_id"],
        "bundle_hash": bundle_hash,
        "feedback_root": str(scope["feedback_root"]),
        "active": active is not None,
        "active_revision": active,
    }


def _run_propose(config_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    scope = _scope(config_path, payload)
    if scope["bundle_path"] is None:
        raise ValueError("propose requires active_models_pointer/model bundle")
    device = resolve_device(scope["device"])
    runtime = AdaptedRuntime(scope["bundle_path"], device=device)
    store = FeedbackStore(scope["feedback_root"])
    options: dict[str, Any] = {}
    nested = payload.get("optimization", payload.get("optimizer", {}))
    if isinstance(nested, Mapping):
        options.update(nested)
    for key in (
        "training_feedback_ids", "validation_feedback_ids", "minimum_recall",
        "minimum_training_samples", "minimum_validation_samples", "epochs",
        "learning_rate", "weight_decay", "gradient_clip", "l2_sp", "seed", "dry_run",
    ):
        if key in payload:
            options[key] = payload[key]
    options["training_feedback_ids"] = _ids(options.get("training_feedback_ids"))
    options["validation_feedback_ids"] = _ids(options.get("validation_feedback_ids"))
    result = propose_adaptive_revision(
        runtime,
        store,
        scope["category"],
        run_id=scope["run_id"],
        bundle_hash=_bundle_scope(runtime, scope),
        **options,
    )
    return result


def _run_activate(config_path: Path, payload: Mapping[str, Any], candidate_arg: object) -> dict[str, Any]:
    scope = _scope(config_path, payload)
    store = FeedbackStore(scope["feedback_root"])
    value = candidate_arg if candidate_arg not in {None, ""} else payload.get("candidate", payload.get("revision_id"))
    reference = _candidate_reference(store, value)
    return store.activate_revision(
        reference,
        category=scope["category"],
        run_id=scope["run_id"],
        bundle_hash=scope.get("bundle_hash") or payload.get("bundle_hash"),
    )


def _run_rollback(config_path: Path, payload: Mapping[str, Any]) -> object:
    scope = _scope(config_path, payload)
    store = FeedbackStore(scope["feedback_root"])
    return store.rollback_revision(
        scope["category"],
        scope["run_id"],
        scope.get("bundle_hash"),
    )


def _run_query(
    config_path: Path,
    payload: Mapping[str, Any],
    *,
    input_dir_arg: str | None = None,
    budget_arg: int | None = None,
    output_arg: str | None = None,
) -> dict[str, Any]:
    """Generate a label-free active-learning query manifest."""
    scope = _scope(config_path, payload)
    if scope["bundle_path"] is None:
        raise ValueError("query requires active_models_pointer/model bundle")
    runtime = AdaptedRuntime(
        scope["bundle_path"], device=resolve_device(scope["device"])
    )
    store = FeedbackStore(scope["feedback_root"])
    apply_active_feedback(
        runtime,
        store,
        scope["category"],
        model_version=scope["run_id"],
    )
    input_value = input_dir_arg or payload.get("query_input_dir")
    if input_value in {None, ""}:
        raise ValueError("query requires --input-dir or query_input_dir in the config")
    budget_value = budget_arg
    if budget_value is None:
        budget_value = payload.get("query_budget", payload.get("budget", 20))
    output_value = output_arg or payload.get("query_output")
    nested = payload.get("active_learning", payload.get("query", {}))
    weights = nested.get("weights") if isinstance(nested, Mapping) else None
    return query_unlabeled_pool(
        runtime,
        store,
        scope["category"],
        _resolve_config_path(config_path, input_value, "query_input_dir"),
        budget=int(budget_value),
        model_version=scope["run_id"],
        output=(
            None
            if output_value in {None, ""}
            else Path(str(output_value))
        ),
        weights=weights,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage complete AOI feedback revisions.")
    parser.add_argument(
        "command",
        choices=("status", "propose", "generate", "activate", "rollback", "query"),
    )
    parser.add_argument("candidate", nargs="?", help="revision ID or receipt path for activate")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "feedback.yaml")
    parser.add_argument("--category")
    parser.add_argument("--run-id")
    parser.add_argument("--bundle-hash")
    parser.add_argument("--input-dir", help="unlabeled image pool for active-learning query")
    parser.add_argument("--budget", type=int, help="number of samples to request")
    parser.add_argument("--output", help="query manifest path inside feedback_root")
    args = parser.parse_args(argv)
    config_path, payload = _config_payload(args.config)
    payload = dict(payload)
    for key, value in (("category", args.category), ("run_id", args.run_id), ("bundle_hash", args.bundle_hash)):
        if value is not None:
            payload[key] = value
    command = "propose" if args.command == "generate" else args.command
    if command == "status":
        result = _run_status(config_path, payload)
    elif command == "propose":
        result = _run_propose(config_path, payload)
    elif command == "query":
        result = _run_query(
            config_path,
            payload,
            input_dir_arg=args.input_dir,
            budget_arg=args.budget,
            output_arg=args.output,
        )
    elif command == "activate":
        result = _run_activate(config_path, payload, args.candidate)
    else:
        result = _run_rollback(config_path, payload)
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
