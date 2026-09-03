"""Deployment modules and backends for batched PatchCore anomaly maps."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PatchCoreMapModule(nn.Module):
    def __init__(self, extractor: nn.Module, memory: torch.Tensor) -> None:
        super().__init__()
        if memory.ndim != 2 or not len(memory):
            raise ValueError("memory must be a non-empty [items, channels] tensor")
        self.extractor = extractor
        self.register_buffer("memory", memory.detach().clone())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.extractor(images)
        batch, channels, grid_height, grid_width = feature_map.shape
        features = feature_map.permute(0, 2, 3, 1).reshape(batch, grid_height * grid_width, channels)
        memory = self.memory.unsqueeze(0).expand(batch, -1, -1)
        distance = torch.cdist(features, memory).amin(dim=2).reshape(batch, 1, grid_height, grid_width)
        return F.interpolate(distance, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


class SharedPatchCoreMapModule(nn.Module):
    """Score two PatchCore memories from one frozen extractor forward."""

    def __init__(
        self,
        extractor: nn.Module,
        source_memory: torch.Tensor,
        adapted_memory: torch.Tensor,
    ) -> None:
        super().__init__()
        for name, memory in (
            ("source_memory", source_memory),
            ("adapted_memory", adapted_memory),
        ):
            if memory.ndim != 2 or not len(memory):
                raise ValueError(f"{name} must be a non-empty [items, channels] tensor")
        if source_memory.shape[1] != adapted_memory.shape[1]:
            raise ValueError("PatchCore memories must use the same feature dimension")
        self.extractor = extractor
        self.register_buffer("source_memory", source_memory.detach().clone())
        self.register_buffer("adapted_memory", adapted_memory.detach().clone())

    @staticmethod
    def _map(
        features: torch.Tensor,
        memory: torch.Tensor,
        *,
        grid_height: int,
        grid_width: int,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch = features.shape[0]
        expanded_memory = memory.unsqueeze(0).expand(batch, -1, -1)
        distance = torch.cdist(features, expanded_memory).amin(dim=2)
        distance = distance.reshape(batch, 1, grid_height, grid_width)
        return F.interpolate(
            distance,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.extractor(images)
        batch, channels, grid_height, grid_width = feature_map.shape
        features = feature_map.permute(0, 2, 3, 1).reshape(
            batch, grid_height * grid_width, channels
        )
        output_size = (int(images.shape[-2]), int(images.shape[-1]))
        return (
            self._map(
                features,
                self.source_memory,
                grid_height=grid_height,
                grid_width=grid_width,
                output_size=output_size,
            ),
            self._map(
                features,
                self.adapted_memory,
                grid_height=grid_height,
                grid_width=grid_width,
                output_size=output_size,
            ),
        )


def images_to_nchw(images: list[np.ndarray]) -> np.ndarray:
    if not images or any(image.ndim != 3 or image.shape[2] != 3 for image in images):
        raise ValueError("non-empty RGB image list required")
    if len({image.shape for image in images}) != 1:
        raise ValueError("all batch images must have equal shape")
    batch = np.stack(images).astype(np.float32) / 255.0
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


class TorchMapBackend:
    def __init__(
        self,
        model,
        *,
        device: str,
        dtype: torch.dtype = torch.float32,
        channels_last: bool = False,
        cudnn_benchmark: bool = False,
    ) -> None:
        if model.memory is None:
            raise ValueError("fitted model memory is required")
        self.device = torch.device(device)
        self.dtype = dtype
        self.channels_last = channels_last
        self.cudnn_benchmark = cudnn_benchmark
        self.module = PatchCoreMapModule(model.extractor, model.memory).to(device=self.device, dtype=dtype).eval()
        if channels_last:
            self.module = self.module.to(memory_format=torch.channels_last)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict_maps_profiled(
        self,
        images: list[np.ndarray],
        *,
        pinned_memory: bool = False,
        non_blocking: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        timings: dict[str, float] = {}
        started = perf_counter_ns()
        array = images_to_nchw(images)
        timings["normalization_ms"] = (perf_counter_ns() - started) / 1_000_000.0

        started = perf_counter_ns()
        tensor = torch.from_numpy(array)
        if pinned_memory and self.device.type == "cuda":
            tensor = tensor.pin_memory()
        tensor = tensor.to(device=self.device, dtype=self.dtype, non_blocking=non_blocking)
        if self.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        self._synchronize()
        timings["host_to_device_ms"] = (perf_counter_ns() - started) / 1_000_000.0

        previous_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        try:
            started = perf_counter_ns()
            maps = self.module(tensor)
            self._synchronize()
            timings["model_forward_ms"] = (perf_counter_ns() - started) / 1_000_000.0
        finally:
            torch.backends.cudnn.benchmark = previous_benchmark

        started = perf_counter_ns()
        result = maps.float().cpu().numpy()
        self._synchronize()
        timings["device_to_host_ms"] = (perf_counter_ns() - started) / 1_000_000.0
        return result, timings

    @torch.inference_mode()
    def predict_maps(self, images: list[np.ndarray]) -> np.ndarray:
        return self.predict_maps_profiled(images)[0]


def _extractors_are_identical(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        left_state[name].shape == right_state[name].shape
        and torch.equal(left_state[name], right_state[name])
        for name in left_state
    )


class SharedTorchMapBackend:
    """Exact two-memory PatchCore backend for models with identical extractors."""

    def __init__(self, source_model, adapted_model, *, device: str) -> None:
        if source_model.memory is None or adapted_model.memory is None:
            raise ValueError("both PatchCore models must have fitted memories")
        if not _extractors_are_identical(
            source_model.extractor, adapted_model.extractor
        ):
            raise ValueError("shared PatchCore backend requires identical extractors")
        self.device = torch.device(device)
        self.module = SharedPatchCoreMapModule(
            source_model.extractor,
            source_model.memory,
            adapted_model.memory,
        ).to(self.device).eval()

    @torch.inference_mode()
    def predict_maps(
        self, images: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        array = images_to_nchw(images)
        tensor = torch.from_numpy(array).to(self.device)
        source_maps, adapted_maps = self.module(tensor)
        return (
            source_maps.float().cpu().numpy(),
            adapted_maps.float().cpu().numpy(),
        )


class OnnxMapBackend:
    def __init__(
        self,
        path: Path,
        *,
        providers: list[str] | None = None,
        enable_profiling: bool = False,
    ) -> None:
        import onnxruntime as ort

        requested = providers or ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.enable_profiling = enable_profiling
        self.session = ort.InferenceSession(str(path.resolve(strict=True)), sess_options=options, providers=requested)
        if self.session.get_providers() != requested:
            raise RuntimeError(f"unexpected ONNX Runtime providers: {self.session.get_providers()}; requested={requested}")

    def predict_maps_profiled(self, images: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
        started = perf_counter_ns()
        array = images_to_nchw(images)
        normalization_ms = (perf_counter_ns() - started) / 1_000_000.0
        started = perf_counter_ns()
        result = np.asarray(self.session.run(["anomaly_maps"], {"images": array})[0], dtype=np.float32)
        model_ms = (perf_counter_ns() - started) / 1_000_000.0
        return result, {
            "normalization_ms": normalization_ms,
            "host_to_device_ms": 0.0,
            "model_forward_ms": model_ms,
            "device_to_host_ms": 0.0,
        }

    def predict_maps(self, images: list[np.ndarray]) -> np.ndarray:
        return self.predict_maps_profiled(images)[0]


def anomaly_score(anomaly_map: np.ndarray) -> float:
    if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
        raise ValueError("finite 2D anomaly map required")
    return float(np.quantile(anomaly_map, 0.995))
