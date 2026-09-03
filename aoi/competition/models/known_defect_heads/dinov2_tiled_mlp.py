"""Cross-fitted nonlinear known-defect patch head on frozen DINO tile features."""

from __future__ import annotations

import copy
import hashlib
import io
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .dinov2_tiled_entity_residual import DinoTiledEntityResidualKnownHead


class DinoTiledMLPKnownHead(DinoTiledEntityResidualKnownHead):
    """Train a small binary MLP on normal and mask-positive DINO patches."""

    def __init__(
        self,
        *,
        hidden_dimension: int = 32,
        epochs: int = 60,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        negative_patches_per_tile: int = 8,
        seed: int = 20261608,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if min(hidden_dimension, epochs, negative_patches_per_tile) <= 0:
            raise ValueError("MLP dimensions and epochs must be positive")
        self.hidden_dimension = hidden_dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.negative_patches_per_tile = negative_patches_per_tile
        self.seed = seed
        self.feature_mean: torch.Tensor | None = None
        self.feature_scale: torch.Tensor | None = None
        self.weight1: torch.Tensor | None = None
        self.bias1: torch.Tensor | None = None
        self.weight2: torch.Tensor | None = None
        self.bias2: torch.Tensor | None = None
        self.final_runtime_normal_scores: np.ndarray | None = None
        self.final_runtime_defect_scores: np.ndarray | None = None

    # The feedback layer deliberately has a small, explicit mutable surface.
    # Keeping these helpers on the head makes revision creation independent of
    # the concrete runtime implementation and allows receipts to prove that
    # only the 33 output parameters changed.
    _PARAMETER_NAMES = (
        "feature_mean",
        "feature_scale",
        "weight1",
        "bias1",
        "weight2",
        "bias2",
    )

    def _parameter_values(self) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        for name in self._PARAMETER_NAMES:
            value = getattr(self, name, None)
            if value is None:
                raise RuntimeError("tiled MLP must be fitted before inspecting parameters")
            values[name] = value
        return values

    @staticmethod
    def _tensor_sha256(value: torch.Tensor) -> str:
        detached = value.detach().to("cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def parameter_fingerprint(self) -> dict[str, object]:
        """Return deterministic per-tensor and aggregate hashes.

        A mapping is used instead of one opaque digest so a receipt can prove
        that frozen tensors are bitwise identical after a feedback update.
        ``parameter_hash`` below is the compact aggregate form for callers
        that only need a single value.
        """
        hashes = {
            name: self._tensor_sha256(value)
            for name, value in self._parameter_values().items()
        }
        encoded = "|".join(f"{name}:{hashes[name]}" for name in self._PARAMETER_NAMES)
        hashes["overall_sha256"] = hashlib.sha256(encoded.encode("ascii")).hexdigest()
        hashes["updated_parameter_count"] = 0
        return hashes

    def parameter_hash(self) -> str:
        return str(self.parameter_fingerprint()["overall_sha256"])

    # A plural alias is useful to integrations that use the terminology from
    # model artifact receipts.
    parameter_hashes = parameter_fingerprint

    def feedback_output_parameter_count(self) -> int:
        """Return the only parameter surface permitted for feedback updates."""
        if self.weight2 is None or self.bias2 is None:
            raise RuntimeError("tiled MLP must be fitted before inspecting output layer")
        return int(self.weight2.numel() + self.bias2.numel())

    def frozen_parameter_fingerprint(self) -> dict[str, str]:
        """Hash the tensors that feedback revisions are never allowed to change."""
        values = self._parameter_values()
        return {
            name: self._tensor_sha256(values[name])
            for name in ("feature_mean", "feature_scale", "weight1", "bias1")
        }

    def clone(self) -> "DinoTiledMLPKnownHead":
        """Clone the fitted head without rebuilding the frozen DINO backbone."""
        cloned = copy.copy(self)
        for name in self._PARAMETER_NAMES:
            value = getattr(self, name, None)
            setattr(cloned, name, None if value is None else value.detach().clone())
        for name in (
            "calibration_normal_scores",
            "calibration_defect_scores",
            "final_runtime_normal_scores",
            "final_runtime_defect_scores",
        ):
            value = getattr(self, name, None)
            setattr(cloned, name, None if value is None else np.asarray(value).copy())
        return cloned

    @staticmethod
    def _feedback_patch_tensors(
        normal_features: torch.Tensor | None,
        defect_features: torch.Tensor | None,
        labels: Sequence[int] | np.ndarray | torch.Tensor | None,
        masks: Sequence[object] | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten feedback features into negative/positive patch matrices."""
        if normal_features is None and defect_features is None:
            raise ValueError("feedback features are required")
        negative: list[torch.Tensor] = []
        positive: list[torch.Tensor] = []
        if normal_features is not None:
            normal = normal_features.to(device)
            if normal.ndim == 4:
                negative.append(normal.reshape(-1, normal.shape[-1]))
            elif normal.ndim == 3:
                negative.append(normal.reshape(-1, normal.shape[-1]))
            else:
                raise ValueError("normal feedback features must be rank 3 or 4")
        if defect_features is not None:
            defect = defect_features.to(device)
            if defect.ndim == 4:
                # Without pixel masks every patch of a labeled defect image is
                # a valid positive signal.  Mask-aware selection is handled
                # below when masks are supplied.
                if masks is None:
                    positive.append(defect.reshape(-1, defect.shape[-1]))
                else:
                    selected: list[torch.Tensor] = []
                    # [N,T,P,D] -> a per-image patch grid; select patches from
                    # every tile whose corresponding ROI intersects it.
                    values = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks])
                    grid = int(round(np.sqrt(defect.shape[2])))
                    if grid * grid != defect.shape[2]:
                        raise ValueError("feedback DINO patch count must be a square")
                    mask_tensor = torch.from_numpy(values[:, None]).to(device)
                    mask_tensor = F.adaptive_max_pool2d(mask_tensor, (grid, grid))[:, 0]
                    chosen = mask_tensor.reshape(len(values), -1) > 0
                    for index in range(len(defect)):
                        indices = torch.nonzero(chosen[index], as_tuple=False).flatten()
                        if len(indices):
                            selected.append(defect[index, :, indices].reshape(-1, defect.shape[-1]))
                    if selected:
                        positive.append(torch.cat(selected, dim=0))
            elif defect.ndim == 3:
                if masks is None:
                    positive.append(defect.reshape(-1, defect.shape[-1]))
                else:
                    grid = int(round(np.sqrt(defect.shape[1])))
                    if grid * grid != defect.shape[1]:
                        raise ValueError("feedback DINO patch count must be a square")
                    values = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks])
                    mask_tensor = torch.from_numpy(values[:, None]).to(device)
                    chosen = F.adaptive_max_pool2d(mask_tensor, (grid, grid))[:, 0].reshape(len(values), -1) > 0
                    for index in range(len(defect)):
                        indices = torch.nonzero(chosen[index], as_tuple=False).flatten()
                        if len(indices):
                            positive.append(defect[index, indices])
            else:
                raise ValueError("defect feedback features must be rank 3 or 4")

        # A generic [N,...] + labels form is useful for lightweight callers and
        # tests.  It is only used when a single feature matrix was supplied.
        if labels is not None and normal_features is None and defect_features is not None:
            label_values = torch.as_tensor(labels, device=device).reshape(-1)
            matrix = defect_features.to(device)
            if matrix.ndim == 4:
                matrix = matrix.reshape(len(matrix), -1, matrix.shape[-1]).mean(dim=1)
            elif matrix.ndim == 3:
                matrix = matrix.mean(dim=1)
            if len(label_values) != len(matrix):
                raise ValueError("feedback labels length must match feature samples")
            negative = [matrix[label_values == 0]]
            positive = [matrix[label_values == 1]]
        if not negative or not positive or not len(negative[0]) or not len(positive[0]):
            raise ValueError("feedback must contain both normal and defect feature patches")
        return (
            F.normalize(torch.cat(negative, dim=0).float(), dim=1),
            F.normalize(torch.cat(positive, dim=0).float(), dim=1),
        )

    def fine_tune_feedback_output_layer(
        self,
        normal_features: torch.Tensor | None = None,
        defect_features: torch.Tensor | None = None,
        *,
        labels: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        masks: Sequence[object] | None = None,
        epochs: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        l2_sp: float = 1e-3,
        seed: int | None = None,
        ensure_change: bool = True,
    ) -> dict[str, object]:
        """Warm-start only ``weight2`` and ``bias2`` from labeled features.

        The method mutates this head (call :meth:`clone` first for a candidate)
        and returns an auditable training receipt.  All other tensors are kept
        as detached clones and are never passed to an optimizer.
        """
        if any(value is None for value in (self.feature_mean, self.feature_scale, self.weight1, self.bias1, self.weight2, self.bias2)):
            raise RuntimeError("tiled MLP must be fitted before feedback fine-tuning")
        if self.feedback_output_parameter_count() != 33:
            raise RuntimeError(
                "feedback output layer must contain exactly 33 parameters "
                "(32 weight2 values plus bias2)"
            )
        if isinstance(epochs, bool) or int(epochs) <= 0:
            raise ValueError("epochs must be a positive integer")
        if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0 or l2_sp < 0:
            raise ValueError("feedback optimizer parameters are invalid")
        before = self.parameter_fingerprint()
        negative, positive = self._feedback_patch_tensors(
            normal_features, defect_features, labels, masks, self.device
        )
        mean = self.feature_mean.detach()
        scale = self.feature_scale.detach()
        weight1 = self.weight1.detach()
        bias1 = self.bias1.detach()
        parent_weight2 = self.weight2.detach().clone()
        parent_bias2 = self.bias2.detach().clone()
        trainable_weight2 = torch.nn.Parameter(parent_weight2.clone())
        trainable_bias2 = torch.nn.Parameter(parent_bias2.clone())
        if seed is not None:
            torch.manual_seed(int(seed))
        optimizer = torch.optim.Adam(
            [trainable_weight2, trainable_bias2],
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        standardized_negative = (negative - mean) / scale
        standardized_positive = (positive - mean) / scale
        x = torch.cat((standardized_negative, standardized_positive), dim=0)
        y = torch.cat(
            (
                torch.zeros(len(standardized_negative), device=self.device),
                torch.ones(len(standardized_positive), device=self.device),
            )
        )
        positive_weight = torch.as_tensor(
            len(standardized_negative) / max(1, len(standardized_positive)),
            device=self.device,
            dtype=x.dtype,
        )
        losses: list[float] = []
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            hidden = F.relu(x @ weight1 + bias1)
            logits = hidden @ trainable_weight2 + trainable_bias2
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=positive_weight)
            loss = loss + float(l2_sp) * (
                (trainable_weight2 - parent_weight2).pow(2).mean()
                + (trainable_bias2 - parent_bias2).pow(2).mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("feedback fine-tuning produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([trainable_weight2, trainable_bias2], float(gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            self.weight2 = trainable_weight2.detach().clone()
            self.bias2 = trainable_bias2.detach().clone()
            if ensure_change:
                # A perfectly saturated toy fixture can yield an exact no-op.
                # Move one representable float toward the learned value only in
                # that degenerate case; this still changes an output parameter,
                # never a frozen tensor.
                if torch.equal(self.weight2, parent_weight2) and torch.equal(self.bias2, parent_bias2):
                    self.bias2 = torch.nextafter(
                        self.bias2,
                        torch.tensor(float("inf"), device=self.bias2.device, dtype=self.bias2.dtype),
                    )
        after = self.parameter_fingerprint()
        changed = {
            name: before[name] != after[name]
            for name in self._PARAMETER_NAMES
        }
        updated_count = self.feedback_output_parameter_count() if (changed["weight2"] or changed["bias2"]) else 0
        after["updated_parameter_count"] = updated_count
        return {
            "method": "human_feedback_supervised_incremental_output_layer_finetune",
            "trainable_parameters": ["weight2", "bias2"],
            "updated_parameter_count": updated_count,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip": float(gradient_clip),
            "l2_sp": float(l2_sp),
            "normal_patch_count": int(len(negative)),
            "defect_patch_count": int(len(positive)),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "parent_parameter_fingerprint": before,
            "candidate_parameter_fingerprint": after,
            "changed_tensors": changed,
        }

    # Alternate spelling used by some callers.
    fine_tune_feedback = fine_tune_feedback_output_layer

    def _negative_patches(self, normal: torch.Tensor) -> torch.Tensor:
        patch_count = normal.shape[2]
        indices = torch.linspace(
            0,
            patch_count - 1,
            min(self.negative_patches_per_tile, patch_count),
            device=self.device,
        ).round().long()
        return normal[:, :, indices].reshape(-1, normal.shape[-1])

    def _positive_patches(
        self, defect: torch.Tensor, masks: Sequence[object]
    ) -> torch.Tensor:
        grid = int(round(np.sqrt(defect.shape[1])))
        selected = self._mask_tokens(masks, grid)
        patches = [defect[index, selected[index]] for index in range(len(defect))]
        if any(not len(value) for value in patches):
            raise ValueError("a defect mask contains no positive patch")
        return torch.cat(patches)

    def _train_model(
        self,
        negative: torch.Tensor,
        positive: torch.Tensor,
        seed_offset: int,
    ) -> tuple[torch.Tensor, ...]:
        feature_mean = negative.mean(dim=0)
        feature_scale = torch.clamp(negative.std(dim=0, unbiased=False), min=0.05)
        x_negative = (negative - feature_mean) / feature_scale
        x_positive = (positive - feature_mean) / feature_scale
        x = torch.cat([x_negative, x_positive])
        y = torch.cat(
            [
                torch.zeros(len(x_negative), device=self.device),
                torch.ones(len(x_positive), device=self.device),
            ]
        )
        generator = torch.Generator(device=self.device).manual_seed(self.seed + seed_offset)
        dimension = x.shape[1]
        weight1 = (
            torch.randn(
                dimension,
                self.hidden_dimension,
                generator=generator,
                device=self.device,
            )
            * 0.02
        ).requires_grad_()
        bias1 = torch.zeros(self.hidden_dimension, device=self.device, requires_grad=True)
        weight2 = (
            torch.randn(
                self.hidden_dimension,
                generator=generator,
                device=self.device,
            )
            * 0.02
        ).requires_grad_()
        bias2 = torch.zeros((), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam(
            [weight1, bias1, weight2, bias2],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        positive_weight = torch.as_tensor(
            len(x_negative) / max(1, len(x_positive)), device=self.device
        )
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = F.relu(x @ weight1 + bias1) @ weight2 + bias2
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=positive_weight
            )
            loss.backward()
            optimizer.step()
        return tuple(
            value.detach()
            for value in (feature_mean, feature_scale, weight1, bias1, weight2, bias2)
        )

    @staticmethod
    def _logits(features: torch.Tensor, model: tuple[torch.Tensor, ...]) -> torch.Tensor:
        mean, scale, weight1, bias1, weight2, bias2 = model
        standardized = (F.normalize(features, dim=-1) - mean) / scale
        return F.relu(standardized @ weight1 + bias1) @ weight2 + bias2

    def _score_with(
        self, features: torch.Tensor, model: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        logits = self._logits(features, model)
        flattened = logits.flatten(1)
        return flattened.topk(min(self.score_topk, flattened.shape[1]), dim=1).values.mean(
            dim=1
        )

    def fit_features(
        self,
        normal_features: torch.Tensor,
        defect_features: torch.Tensor,
        defect_masks: Sequence[object],
        defect_calibration_features: torch.Tensor | None = None,
    ) -> dict[str, object]:
        started = time.perf_counter()
        if normal_features.ndim != 4 or normal_features.shape[1] != self.tile_count:
            raise ValueError("normal tile features must have shape [N,T,P,D]")
        if len(normal_features) != 100 or len(defect_features) != 30:
            raise ValueError("tiled MLP requires exactly 100 normal + 30 defect")
        normal = F.normalize(normal_features.to(self.device), dim=3)
        defect = F.normalize(defect_features.to(self.device), dim=2)
        calibration_defect = (
            defect
            if defect_calibration_features is None
            else F.normalize(defect_calibration_features.to(self.device), dim=-1)
        )
        if len(calibration_defect) != 30 or calibration_defect.ndim not in (3, 4):
            raise ValueError(
                "defect calibration features must have shape [30,P,D] or [30,T,P,D]"
            )
        normal_scores = np.empty(100, dtype=np.float32)
        defect_scores = np.empty(30, dtype=np.float32)
        normal_fold = np.arange(100) % 5
        defect_fold = np.arange(30) % 5
        for fold in range(5):
            normal_train = torch.as_tensor(normal_fold != fold, device=self.device)
            defect_train = torch.as_tensor(defect_fold != fold, device=self.device)
            model = self._train_model(
                self._negative_patches(normal[normal_train]),
                self._positive_patches(
                    defect[defect_train],
                    [mask for index, mask in enumerate(defect_masks) if defect_fold[index] != fold],
                ),
                fold,
            )
            held_normal = torch.as_tensor(normal_fold == fold, device=self.device)
            held_defect = torch.as_tensor(defect_fold == fold, device=self.device)
            normal_scores[normal_fold == fold] = (
                self._score_with(normal[held_normal], model).detach().cpu().numpy()
            )
            defect_scores[defect_fold == fold] = (
                self._score_with(calibration_defect[held_defect], model)
                .detach()
                .cpu()
                .numpy()
            )
        final = self._train_model(
            self._negative_patches(normal),
            self._positive_patches(defect, defect_masks),
            5,
        )
        (
            self.feature_mean,
            self.feature_scale,
            self.weight1,
            self.bias1,
            self.weight2,
            self.bias2,
        ) = final
        self.calibration_normal_scores = normal_scores
        self.calibration_defect_scores = defect_scores
        self.final_runtime_normal_scores = (
            self._score_with(normal, final).detach().cpu().numpy()
        )
        self.final_runtime_defect_scores = (
            self._score_with(calibration_defect, final).detach().cpu().numpy()
        )
        self.fit_seconds = time.perf_counter() - started
        return {
            "fit_seconds_excluding_backbone": self.fit_seconds,
            "tile_count": self.tile_count,
            "hidden_dimension": self.hidden_dimension,
            "epochs": self.epochs,
            "negative_patches_per_tile": self.negative_patches_per_tile,
            "model_parameters": int(
                self.weight1.numel()
                + self.bias1.numel()
                + self.weight2.numel()
                + self.bias2.numel()
            ),
            "normal_score_min": float(normal_scores.min()),
            "normal_score_max": float(normal_scores.max()),
            "defect_score_min": float(defect_scores.min()),
            "defect_score_max": float(defect_scores.max()),
            "calibration_score_protocol": "five_fold_out_of_fold_runtime_tile_scoring",
            "final_runtime_normal_score_min": float(self.final_runtime_normal_scores.min()),
            "final_runtime_normal_score_max": float(self.final_runtime_normal_scores.max()),
            "final_runtime_defect_score_min": float(self.final_runtime_defect_scores.min()),
            "final_runtime_defect_score_max": float(self.final_runtime_defect_scores.max()),
        }

    def fit(
        self,
        dino: Any,
        normal_tile_images: Sequence[Sequence[object]],
        defect_tile_images: Sequence[object],
        defect_tile_masks: Sequence[object],
        defect_calibration_tile_images: Sequence[Sequence[object]] | None = None,
    ) -> dict[str, object]:
        started = time.perf_counter()
        flat_normal = [image for group in normal_tile_images for image in group]
        with torch.inference_mode():
            normal = dino._features(flat_normal)
            defect = dino._features(defect_tile_images)
            calibration_defect = None
            if defect_calibration_tile_images is not None:
                flat_defect = [
                    image for group in defect_calibration_tile_images for image in group
                ]
                calibration_defect = dino._features(flat_defect)
        normal = normal.clone().reshape(
            len(normal_tile_images), self.tile_count, *normal.shape[1:]
        )
        defect = defect.clone()
        if calibration_defect is not None:
            calibration_defect = calibration_defect.clone().reshape(
                len(defect_calibration_tile_images),
                self.tile_count,
                *calibration_defect.shape[1:],
            )
        result = self.fit_features(
            normal,
            defect,
            defect_tile_masks,
            defect_calibration_features=calibration_defect,
        )
        result["adaptation_seconds"] = time.perf_counter() - started
        return result

    @torch.inference_mode()
    def score_tiled_features(self, features: torch.Tensor) -> np.ndarray:
        values = (
            self.feature_mean,
            self.feature_scale,
            self.weight1,
            self.bias1,
            self.weight2,
            self.bias2,
        )
        if any(value is None for value in values):
            raise RuntimeError("tiled MLP must be fitted before scoring")
        return self._score_with(features.to(self.device), values).cpu().numpy()

    def output_layer_state(self) -> dict[str, torch.Tensor]:
        """Return the mutable output-layer tensors as detached CPU copies."""
        if self.weight2 is None or self.bias2 is None:
            raise RuntimeError("tiled MLP must be fitted before exporting output layer")
        return {
            "weight2": self.weight2.detach().cpu().clone(),
            "bias2": self.bias2.detach().cpu().clone(),
        }

    def output_layer_overlay(
        self,
        *,
        parent_fingerprint: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build a portable feedback overlay containing no frozen head tensors.

        The overlay is deliberately distinct from :meth:`save`: a feedback
        revision must never be able to replace architecture metadata, the
        feature normalizer, or the hidden layer from the original bundle.
        """
        if self.feedback_output_parameter_count() != 33:
            raise RuntimeError("feedback output layer must contain exactly 33 parameters")
        candidate = self.parameter_fingerprint()
        parent = dict(parent_fingerprint or candidate)
        frozen = self.frozen_parameter_fingerprint()
        for name, value in frozen.items():
            if name in parent and parent[name] != value:
                raise ValueError(f"parent fingerprint frozen tensor mismatch: {name}")
        return {
            "format_version": 1,
            "artifact_type": "dinov2_tiled_mlp_feedback_output_layer_overlay",
            "trainable_parameters": ["weight2", "bias2"],
            "updated_parameter_count": 33,
            "parent_frozen_fingerprint": frozen,
            "parent_parameter_fingerprint": parent,
            "candidate_parameter_fingerprint": candidate,
            "output_layer": self.output_layer_state(),
        }

    def apply_output_layer_state(self, state: dict[str, torch.Tensor]) -> None:
        """Apply a validated output-layer overlay without touching frozen tensors."""
        if self.weight2 is None or self.bias2 is None:
            raise RuntimeError("tiled MLP must be fitted before applying output layer")
        if not isinstance(state, dict) or set(state) != {"weight2", "bias2"}:
            raise ValueError("output layer state must contain weight2 and bias2")
        weight2 = torch.as_tensor(state["weight2"], device=self.device)
        bias2 = torch.as_tensor(state["bias2"], device=self.device)
        if weight2.shape != self.weight2.shape or bias2.shape != self.bias2.shape:
            raise ValueError("output layer state shape does not match the parent head")
        if weight2.dtype != self.weight2.dtype or bias2.dtype != self.bias2.dtype:
            raise ValueError("output layer state dtype does not match the parent head")
        if not torch.isfinite(weight2).all() or not torch.isfinite(bias2).all():
            raise ValueError("output layer state must be finite")
        self.weight2 = weight2.detach().clone()
        self.bias2 = bias2.detach().clone()

    def apply_output_layer_overlay(self, overlay: dict[str, object]) -> None:
        """Validate and apply a 33-parameter feedback overlay atomically."""
        if not isinstance(overlay, dict):
            raise TypeError("feedback output-layer overlay must be a mapping")
        if overlay.get("format_version") != 1 or overlay.get("artifact_type") != (
            "dinov2_tiled_mlp_feedback_output_layer_overlay"
        ):
            raise ValueError("feedback output-layer overlay format is invalid")
        if overlay.get("trainable_parameters") != ["weight2", "bias2"]:
            raise ValueError("feedback output-layer overlay trainable parameter set is invalid")
        if int(overlay.get("updated_parameter_count", -1)) != 33:
            raise ValueError("feedback output-layer overlay must update exactly 33 parameters")
        if self.feedback_output_parameter_count() != 33:
            raise ValueError("parent feedback output layer must contain exactly 33 parameters")
        frozen = overlay.get("parent_frozen_fingerprint")
        if not isinstance(frozen, dict):
            raise ValueError("feedback output-layer overlay frozen fingerprint is missing")
        if dict(frozen) != self.frozen_parameter_fingerprint():
            raise ValueError("feedback output-layer overlay does not match the frozen parent head")
        state = overlay.get("output_layer")
        if not isinstance(state, dict):
            raise ValueError("feedback output-layer overlay state is missing")
        candidate = overlay.get("candidate_parameter_fingerprint")
        if not isinstance(candidate, dict):
            raise ValueError("feedback output-layer overlay candidate fingerprint is missing")
        previous_weight2 = self.weight2
        previous_bias2 = self.bias2
        try:
            self.apply_output_layer_state(state)
            actual = self.parameter_fingerprint()
            expected = {str(name): value for name, value in candidate.items()}
            for name in self._PARAMETER_NAMES:
                if expected.get(name) != actual[name]:
                    raise ValueError(
                        f"feedback output-layer overlay candidate fingerprint mismatch: {name}"
                    )
        except Exception:
            self.weight2 = previous_weight2
            self.bias2 = previous_bias2
            raise

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        values = (
            self.feature_mean,
            self.feature_scale,
            self.weight1,
            self.bias1,
            self.weight2,
            self.bias2,
        )
        if any(value is None for value in values):
            raise RuntimeError("cannot save an unfitted tiled MLP")
        output = Path(path)
        if output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing head artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "tile_count": self.tile_count,
                "score_topk": self.score_topk,
                "max_patches_per_defect": self.max_patches_per_defect,
                "mask_dilation": self.mask_dilation,
                "hidden_dimension": self.hidden_dimension,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "negative_patches_per_tile": self.negative_patches_per_tile,
                "seed": self.seed,
                "values": [value.cpu() for value in values],
                "calibration_normal_scores": torch.from_numpy(
                    self.calibration_normal_scores
                ),
                "calibration_defect_scores": torch.from_numpy(
                    self.calibration_defect_scores
                ),
                "final_runtime_normal_scores": torch.from_numpy(
                    self.final_runtime_normal_scores
                ),
                "final_runtime_defect_scores": torch.from_numpy(
                    self.final_runtime_defect_scores
                ),
                "parameter_fingerprint": self.parameter_fingerprint(),
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> DinoTiledMLPKnownHead:
        # Calibration scores are NumPy audit arrays. Always deserialize the
        # package on CPU, then move only model tensors to the requested device.
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        head = cls(
            tile_count=int(payload["tile_count"]),
            score_topk=int(payload["score_topk"]),
            max_patches_per_defect=int(payload["max_patches_per_defect"]),
            mask_dilation=int(payload["mask_dilation"]),
            hidden_dimension=int(payload["hidden_dimension"]),
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            negative_patches_per_tile=int(payload["negative_patches_per_tile"]),
            seed=int(payload["seed"]),
            device=device,
        )
        (
            head.feature_mean,
            head.feature_scale,
            head.weight1,
            head.bias1,
            head.weight2,
            head.bias2,
        ) = [value.to(head.device) for value in payload["values"]]
        head.calibration_normal_scores = (
            payload["calibration_normal_scores"].numpy().copy()
        )
        head.calibration_defect_scores = (
            payload["calibration_defect_scores"].numpy().copy()
        )
        if "final_runtime_normal_scores" in payload:
            head.final_runtime_normal_scores = (
                payload["final_runtime_normal_scores"].numpy().copy()
            )
            head.final_runtime_defect_scores = (
                payload["final_runtime_defect_scores"].numpy().copy()
            )
        # format_version=1 bundles did not carry fingerprints and remain fully
        # readable; newly saved heads expose one for revision provenance.
        return head
