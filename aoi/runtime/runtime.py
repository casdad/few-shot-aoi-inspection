from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Mapping

import yaml

from aoi.competition.baselines import DeepPatchCoreLite
from aoi.competition.deployment import SharedTorchMapBackend
from aoi.competition.models.anomaly_detectors import DINOv2SpatialPrototype, MultiScalePatchMemory
from aoi.competition.models.known_defect_heads import DinoTiledMLPKnownHead

from .fusion import score_images
from aoi.paths import resolve_artifact


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(payload: object) -> str:
    encoded = yaml.safe_dump(
        payload, sort_keys=True, allow_unicode=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_bundle_integrity(
    bundle: Mapping[str, Any], bundle_path: Path
) -> dict[str, Any]:
    """Verify immutable metadata and all hashes recorded by new bundles."""

    expected_bundle_hash = bundle.get("bundle_hash")
    bundle_verified = False
    if expected_bundle_hash not in {None, "pending"}:
        actual_bundle_hash = _payload_hash(
            {key: value for key, value in bundle.items() if key != "bundle_hash"}
        )
        if actual_bundle_hash != expected_bundle_hash:
            raise ValueError(
                "model bundle metadata SHA-256 mismatch: "
                f"expected {expected_bundle_hash}, got {actual_bundle_hash}"
            )
        bundle_verified = True

    recorded = bundle.get("artifact_sha256")
    if recorded is None:
        return {
            "bundle_hash_verified": bundle_verified,
            "artifact_hashes_verified": False,
            "legacy_bundle_without_artifact_hashes": True,
        }
    if not isinstance(recorded, Mapping):
        raise TypeError("model bundle artifact_sha256 must be a mapping")
    recorded_models = recorded.get("models")
    if not isinstance(recorded_models, Mapping):
        raise TypeError("model bundle artifact_sha256.models must be a mapping")
    models = bundle.get("models")
    if not isinstance(models, Mapping):
        raise TypeError("model bundle models must be a mapping")
    if set(recorded_models) != set(models):
        raise ValueError(
            "model bundle artifact hashes must cover every model exactly"
        )
    base = bundle_path.parent
    paths: dict[str, Path] = {
        "source_model": resolve_artifact(bundle["source_model"], base=base),
        "calibration": resolve_artifact(bundle["calibration"], base=base),
        "support_receipt": resolve_artifact(bundle["support_receipt"], base=base),
    }
    expected: dict[str, str] = {
        "source_model": str(recorded.get("source_model", "")),
        "calibration": str(recorded.get("calibration", "")),
        "support_receipt": str(recorded.get("support_receipt", "")),
    }
    for name, value in models.items():
        key = f"models.{name}"
        paths[key] = resolve_artifact(value, base=base)
        expected[key] = str(recorded_models[name])
    verified: dict[str, str] = {}
    for name, path in paths.items():
        expected_hash = expected[name].lower()
        if len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash
        ):
            raise ValueError(f"invalid recorded SHA-256 for {name}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"model artifact SHA-256 mismatch for {name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        verified[name] = actual_hash
    return {
        "bundle_hash_verified": bundle_verified,
        "artifact_hashes_verified": True,
        "legacy_bundle_without_artifact_hashes": False,
        "verified_artifacts": verified,
    }


class AdaptedRuntime:
    def __init__(self, bundle_path: str | Path, *, device: str):
        self.bundle_path = resolve_artifact(bundle_path)
        self.bundle = yaml.safe_load(self.bundle_path.read_text(encoding="utf-8"))
        if not isinstance(self.bundle, Mapping):
            raise TypeError("model bundle must be a YAML mapping")
        self.integrity = _verify_bundle_integrity(
            self.bundle, self.bundle_path
        )
        self.device = device
        # Feedback activation and inference share this lock so a batch cannot
        # observe a head from one revision and a feature bank/threshold from
        # another.
        self._feedback_lock = threading.RLock()
        bundle_dir = self.bundle_path.parent
        calibration_path = resolve_artifact(self.bundle["calibration"], base=bundle_dir)
        self.calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
        self.source = DeepPatchCoreLite.load(
            resolve_artifact(self.bundle["source_model"], base=bundle_dir),
            map_location=device,
        )
        self.adapted = DeepPatchCoreLite.load(
            resolve_artifact(self.bundle["models"]["adapted"], base=bundle_dir),
            map_location=device,
        )
        self.primary = MultiScalePatchMemory.load(
            resolve_artifact(self.bundle["models"]["primary"], base=bundle_dir),
            device=device,
        )
        self.dino = DINOv2SpatialPrototype.load(
            resolve_artifact(self.bundle["models"]["dino"], base=bundle_dir),
            device=device,
        )
        self.head = DinoTiledMLPKnownHead.load(
            resolve_artifact(self.bundle["models"]["head"], base=bundle_dir),
            device=device,
        )
        # Adaptation copies the frozen source extractor exactly, so both memory
        # banks can be scored from one ResNet forward.
        self.g2_backend = SharedTorchMapBackend(
            self.source, self.adapted, device=device
        )

    def predict_batch(self, images, ids, *, include_anomaly_map=False):
        with self._feedback_lock:
            rows = score_images(
                images=images, ids=ids, source=self.source, adapted=self.adapted,
                primary=self.primary, dino=self.dino, head=self.head,
                calibration=self.calibration, device=self.device,
                g2_backend=self.g2_backend,
                include_anomaly_map=include_anomaly_map,
            )
            # The feedback threshold is part of the same revision as the head
            # overlay and DINO bank.  Keep it inside the inference lock so a
            # batch cannot expose a mixture of revision components.
            threshold = getattr(self, "_feedback_threshold", None)
            if threshold is None:
                return rows
            from aoi.feedback.service import apply_threshold

            return [apply_threshold(dict(row), threshold) for row in rows]

    def apply_feedback(self, store, category: str, *, enabled: bool = True):
        """Atomically apply the active feedback revision to this runtime."""
        from aoi.feedback.runtime import apply_active_feedback

        return apply_active_feedback(self, store, category, enabled=enabled)

    def predict(self, image, sample_id="image"):
        return self.predict_batch([image], [sample_id])[0]

    def predict_explain(self, image, sample_id="image"):
        return self.predict_batch(
            [image], [sample_id], include_anomaly_map=True
        )[0]
