from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml

from aoi.paths import resolve_artifact
from aoi.runtime.config import load_yaml, resolve_device, resolve_path
from aoi.runtime.data import image_files, read_rgb
from aoi.runtime.runtime import AdaptedRuntime


BENCHMARK_CONFIG_SCHEMA_VERSION = 2
BENCHMARK_REPORT_SCHEMA_VERSION = 3
PERCENTILE_METHOD = "linear"
SUPPORTED_METRIC = "p50_ms"
STATUS_PASS = "PASS"
STATUS_FAIL_HARDWARE = "FAIL_HARDWARE_MISMATCH"
STATUS_FAIL_RESOLUTION = "FAIL_RESOLUTION_MISMATCH"
STATUS_FAIL_PROTOCOL = "FAIL_SAMPLE_PROTOCOL"
STATUS_FAIL_GPU_EXCLUSIVITY = "FAIL_GPU_NOT_EXCLUSIVE"
STATUS_FAIL_LATENCY = "FAIL_P50_LATENCY"
STATUS_NOT_MEASURED = "NOT_MEASURED"
MINIMUM_WARMUP = 20
MINIMUM_RUNS = 100


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_evidence(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


@dataclass(frozen=True)
class PerformanceTarget:
    target_id: str
    authority: str
    requirement: str
    metric: str
    comparator: str
    limit_ms: float
    required_resolution: tuple[int, int]
    required_device_type: str
    allowed_device_names: tuple[str, ...]
    metric_note: str
    target_hardware: str
    hardware_match_policy: str = "normalized_exact_device_name"
    diagnostic_only: bool = False
    minimum_warmup: int = MINIMUM_WARMUP
    minimum_runs: int = MINIMUM_RUNS
    batch_size: int = 1

    def as_definition(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "authority": self.authority,
            "requirement": self.requirement,
            "metric": self.metric,
            "comparator": self.comparator,
            "limit_ms": self.limit_ms,
            "required_resolution": list(self.required_resolution),
            "required_device_type": self.required_device_type,
            "allowed_device_names": list(self.allowed_device_names),
            "hardware_match_policy": self.hardware_match_policy,
            "target_hardware": self.target_hardware,
            "diagnostic_only": self.diagnostic_only,
            "claim_supported": not self.diagnostic_only,
            "minimum_warmup": self.minimum_warmup,
            "minimum_runs": self.minimum_runs,
            "batch_size": self.batch_size,
            "metric_note": self.metric_note,
        }


PERFORMANCE_TARGETS: Mapping[str, PerformanceTarget] = MappingProxyType(
    {
        "official_rtx2060": PerformanceTarget(
            target_id="official_rtx2060",
            authority="competition_official",
            requirement=(
                "For a 2500x2500 input image, runtime must be strictly below "
                "200 ms on NVIDIA RTX 2060 or lower."
            ),
            metric=SUPPORTED_METRIC,
            comparator="<",
            limit_ms=200.0,
            required_resolution=(2500, 2500),
            required_device_type="cuda",
            allowed_device_names=("NVIDIA GeForce RTX 2060",),
            metric_note=(
                "The official requirement specifies NVIDIA RTX 2060 or lower and "
                "runtime strictly below 200 ms, but does not define a percentile; "
                "this project uses synchronized single-image P50 as its statistical "
                "protocol. Exact RTX 2060 is automated; a lower-tier model needs a "
                "separately reviewed immutable standard."
            ),
            target_hardware="NVIDIA RTX 2060 or lower",
        ),
        "internal_rtx4070": PerformanceTarget(
            target_id="internal_rtx4070",
            authority="project_internal",
            requirement=(
                "For a 2500x2500 input image, synchronized single-image P50 "
                "latency must be strictly below 100 ms on NVIDIA GeForce RTX 4070."
            ),
            metric=SUPPORTED_METRIC,
            comparator="<",
            limit_ms=100.0,
            required_resolution=(2500, 2500),
            required_device_type="cuda",
            allowed_device_names=("NVIDIA GeForce RTX 4070",),
            metric_note=(
                "Internal engineering gate using at least 20 warmup runs and 100 "
                "synchronized timed runs."
            ),
            target_hardware="NVIDIA GeForce RTX 4070",
        ),
        "cpu_diagnostic": PerformanceTarget(
            target_id="cpu_diagnostic",
            authority="project_diagnostic",
            requirement=(
                "For a 2500x2500 input image, single-image CPU P50 latency "
                "must be strictly below 2000 ms. This diagnostic is not a "
                "competition hardware acceptance claim."
            ),
            metric=SUPPORTED_METRIC,
            comparator="<",
            limit_ms=2000.0,
            required_resolution=(2500, 2500),
            required_device_type="cpu",
            allowed_device_names=(),
            metric_note=(
                "CPU engineering diagnostic using at least 20 warmup runs and "
                "100 timed single-image runs. Passing does not establish RTX "
                "2060 or RTX 4070 compliance."
            ),
            target_hardware="Current CPU",
            hardware_match_policy="device_type_only",
            diagnostic_only=True,
        ),
    }
)


def _non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _normalize_device_name(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _resolution(value: Any, context: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{context} must contain [width, height]")
    values: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise TypeError(f"{context}[{index}] must be a positive integer")
        values.append(int(item))
    return values[0], values[1]


def load_performance_targets(config: Mapping[str, Any]) -> tuple[PerformanceTarget, ...]:
    """Return immutable targets and reject configurable target definitions."""

    if "performance_targets" in config:
        raise ValueError(
            "performance_targets are built into the benchmark implementation and "
            "cannot be defined or weakened in YAML"
        )
    return tuple(PERFORMANCE_TARGETS.values())


def get_selected_target(config: Mapping[str, Any]) -> PerformanceTarget:
    selected = _non_empty_string(
        config.get("benchmark_standard"), "benchmark_standard"
    )
    try:
        target = PERFORMANCE_TARGETS[selected]
    except KeyError as exc:
        choices = ", ".join(PERFORMANCE_TARGETS)
        raise ValueError(
            f"unknown benchmark_standard {selected!r}; choose one of: {choices}"
        ) from exc
    legacy_primary = config.get("primary_target_id")
    if legacy_primary is not None and legacy_primary != selected:
        raise ValueError(
            "primary_target_id cannot override benchmark_standard; "
            f"expected {selected!r}"
        )
    return target


def _positive_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    if value < 1:
        raise ValueError(f"{context} must be >= 1")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    if value < 0:
        raise ValueError(f"{context} must be >= 0")
    return value


def _allow_nonexclusive_measurement(config: Mapping[str, Any]) -> bool:
    value = config.get("allow_nonexclusive_measurement", False)
    if type(value) is not bool:
        raise TypeError("allow_nonexclusive_measurement must be a boolean")
    return value


def _reject_override(field: str, configured: Any, required: Any) -> None:
    raise ValueError(
        f"{field}={configured!r} cannot override benchmark_standard; "
        f"required value is {required!r}"
    )


def validate_benchmark_config(
    config: Mapping[str, Any], target: PerformanceTarget
) -> tuple[int, int]:
    """Validate legacy aliases without allowing them to weaken the standard."""

    if "performance_targets" in config:
        load_performance_targets(config)
    if config.get("enforce_primary_target") is False:
        raise ValueError("enforce_primary_target cannot be disabled")
    _allow_nonexclusive_measurement(config)

    if "target_size" in config:
        size = _positive_int(config["target_size"], "target_size")
        configured_resolution = (size, size)
        if configured_resolution != target.required_resolution:
            _reject_override(
                "target_size", configured_resolution, target.required_resolution
            )
    for field in ("target_resolution", "resolution"):
        if field in config:
            configured_resolution = _resolution(config[field], field)
            if configured_resolution != target.required_resolution:
                _reject_override(
                    field, configured_resolution, target.required_resolution
                )
    if "p95_limit_ms" in config:
        raise ValueError(
            "p95_limit_ms is not a supported target field; benchmark standards "
            "use immutable P50 limits"
        )
    for field in ("target_ms", "p50_limit_ms"):
        if field in config:
            configured_limit = _finite_number(config[field], field, minimum=0.0)
            if configured_limit != target.limit_ms:
                _reject_override(field, configured_limit, target.limit_ms)
    if "batch_size" in config:
        batch_size = _positive_int(config["batch_size"], "batch_size")
        if batch_size != target.batch_size:
            _reject_override("batch_size", batch_size, target.batch_size)
    if "expected_device_name" in config:
        expected_name = _non_empty_string(
            config["expected_device_name"], "expected_device_name"
        )
        allowed = {_normalize_device_name(name) for name in target.allowed_device_names}
        if (
            target.hardware_match_policy != "normalized_exact_device_name"
            or _normalize_device_name(expected_name) not in allowed
        ):
            _reject_override(
                "expected_device_name", expected_name, target.target_hardware
            )
    if "target_hardware" in config:
        hardware = _non_empty_string(config["target_hardware"], "target_hardware")
        if _normalize_device_name(hardware) != _normalize_device_name(
            target.target_hardware
        ):
            _reject_override("target_hardware", hardware, target.target_hardware)

    warmup = _nonnegative_int(
        config.get("warmup", target.minimum_warmup), "warmup"
    )
    runs = _nonnegative_int(config.get("runs", target.minimum_runs), "runs")
    return warmup, runs


def _device_type(device: str) -> str:
    normalized = str(device).strip().casefold()
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise ValueError(f"unsupported benchmark device {device!r}; expected cuda or cpu")


def evaluate_performance_targets(
    targets: Sequence[PerformanceTarget],
    *,
    primary_target_id: str,
    device: str,
    device_name: str,
    resolution: Sequence[int],
    warmup: int,
    runs: int,
    p50_ms: float | None,
    exclusive_gpu: bool | None = True,
) -> dict[str, Any]:
    """Evaluate claim eligibility without inferring GPU performance ordering."""

    observed_resolution = _resolution(list(resolution), "observed resolution")
    observed_device_type = _device_type(device)
    normalized_observed_name = _normalize_device_name(device_name)
    observed_warmup = _nonnegative_int(warmup, "warmup")
    observed_runs = _nonnegative_int(runs, "runs")
    observed_p50 = None if p50_ms is None else float(p50_ms)
    evaluations: list[dict[str, Any]] = []
    target_ids = {target.target_id for target in targets}
    if primary_target_id not in target_ids:
        raise ValueError(f"primary target {primary_target_id!r} is not defined")
    for target in targets:
        allowed_names = tuple(target.allowed_device_names)
        if target.hardware_match_policy == "device_type_only":
            hardware_match = observed_device_type == target.required_device_type
        elif target.hardware_match_policy == "normalized_exact_device_name":
            hardware_match = bool(
                observed_device_type == target.required_device_type
                and normalized_observed_name
                in {_normalize_device_name(name) for name in allowed_names}
            )
        else:
            raise ValueError(
                f"unsupported hardware match policy "
                f"{target.hardware_match_policy!r}"
            )
        resolution_match = observed_resolution == target.required_resolution
        warmup_sufficient = observed_warmup >= target.minimum_warmup
        runs_sufficient = observed_runs >= target.minimum_runs
        reasons: list[str] = []
        blockers: list[str] = []
        if not hardware_match:
            reasons.append("device_or_gpu_name_mismatch")
            blockers.append(
                "actual device does not match the standard hardware policy: "
                f"{device_name!r}"
            )
        if not resolution_match:
            reasons.append("resolution_mismatch")
            blockers.append(
                f"resolution must be {list(target.required_resolution)}, got "
                f"{list(observed_resolution)}"
            )
        if not warmup_sufficient:
            reasons.append("insufficient_warmup")
            blockers.append(
                f"warmup must be >= {target.minimum_warmup}, got {observed_warmup}"
            )
        if not runs_sufficient:
            reasons.append("insufficient_measured_runs")
            blockers.append(
                f"runs must be >= {target.minimum_runs}, got {observed_runs}"
            )
        finite_measurement = bool(
            observed_p50 is not None
            and np.isfinite(observed_p50)
            and observed_p50 >= 0.0
        )
        if observed_p50 is not None and not finite_measurement:
            reasons.append("non_finite_or_negative_p50")
            blockers.append("P50 latency must be finite and non-negative")
        requires_exclusive_gpu = target.required_device_type == "cuda"
        evaluated_exclusivity = exclusive_gpu if requires_exclusive_gpu else None
        if requires_exclusive_gpu and exclusive_gpu is False:
            reasons.append("concurrent_python_gpu_process_detected")
            blockers.append(
                "another Python GPU process was detected; benchmark requires "
                "exclusive GPU access"
            )
        elif requires_exclusive_gpu and exclusive_gpu is None:
            reasons.append("gpu_exclusivity_not_verified")
            blockers.append(
                "GPU exclusivity could not be verified with nvidia-smi"
            )

        metric_pass = (
            None if not finite_measurement else observed_p50 < target.limit_ms
        )
        if not hardware_match:
            status = STATUS_FAIL_HARDWARE
        elif not resolution_match:
            status = STATUS_FAIL_RESOLUTION
        elif not warmup_sufficient or not runs_sufficient:
            status = STATUS_FAIL_PROTOCOL
        elif requires_exclusive_gpu and exclusive_gpu is not True:
            status = STATUS_FAIL_GPU_EXCLUSIVITY
        elif observed_p50 is None:
            status = STATUS_NOT_MEASURED
            blockers.append("P50 latency has not been measured")
        elif not finite_measurement:
            status = STATUS_FAIL_LATENCY
        else:
            reasons.append("strict_less_than_limit")
            if metric_pass:
                status = STATUS_PASS
            else:
                status = STATUS_FAIL_LATENCY
                blockers.append(
                    f"P50 must be strictly < {target.limit_ms:g} ms, got "
                    f"{observed_p50:g} ms"
                )
        passed = status == STATUS_PASS
        claim_eligible = bool(passed and not target.diagnostic_only)
        evaluations.append(
            {
                "target_id": target.target_id,
                "authority": target.authority,
                "requirement": target.requirement,
                "metric": target.metric,
                "metric_note": target.metric_note or None,
                "comparator": target.comparator,
                "limit_ms": target.limit_ms,
                "observed_p50_ms": observed_p50,
                "required_resolution": list(target.required_resolution),
                "observed_resolution": list(observed_resolution),
                "required_device_type": target.required_device_type,
                "allowed_device_names": list(allowed_names),
                "hardware_match_policy": target.hardware_match_policy,
                "target_hardware": target.target_hardware,
                "diagnostic_only": target.diagnostic_only,
                "observed_device_type": observed_device_type,
                "observed_device_name": device_name,
                "hardware_match": hardware_match,
                "resolution_match": resolution_match,
                "minimum_warmup": target.minimum_warmup,
                "observed_warmup": observed_warmup,
                "warmup_sufficient": warmup_sufficient,
                "minimum_runs": target.minimum_runs,
                "observed_runs": observed_runs,
                "runs_sufficient": runs_sufficient,
                "metric_pass": metric_pass,
                "exclusive_gpu_verified": evaluated_exclusivity,
                "passed": passed,
                "claim_eligible": claim_eligible,
                "status": status,
                "reason_codes": reasons,
                "blockers": blockers,
            }
        )
    primary = next(
        evaluation
        for evaluation in evaluations
        if evaluation["target_id"] == primary_target_id
    )
    return {
        "primary_target_id": primary_target_id,
        "status": primary["status"],
        "passed": primary["passed"],
        "claim_eligible": primary["claim_eligible"],
        "diagnostic_only": primary["diagnostic_only"],
        "validation": {
            "hardware_match": primary["hardware_match"],
            "resolution_match": primary["resolution_match"],
            "warmup_sufficient": primary["warmup_sufficient"],
            "runs_sufficient": primary["runs_sufficient"],
            "latency_p50_strictly_under_target": primary["metric_pass"],
            "exclusive_gpu_verified": primary["exclusive_gpu_verified"],
            "passed": primary["passed"],
            "claim_eligible": primary["claim_eligible"],
            "status": primary["status"],
            "blockers": primary["blockers"],
        },
        "targets": evaluations,
    }


def _percentile(values: np.ndarray, percentile: float) -> float:
    try:
        return float(np.percentile(values, percentile, method=PERCENTILE_METHOD))
    except TypeError:  # pragma: no cover - compatibility with old NumPy
        return float(
            np.percentile(values, percentile, interpolation=PERCENTILE_METHOD)
        )


def _runtime_environment(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch_device = torch.device(device)
            index = torch_device.index
            if index is None:
                index = torch.cuda.current_device()
            payload["cuda_device_index"] = int(index)
            payload["cudnn_version"] = torch.backends.cudnn.version()
        except (AssertionError, RuntimeError, ValueError) as exc:
            payload["cuda_device_error"] = str(exc)
    return payload


def _gpu_exclusivity(device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return {
            "query_succeeded": True,
            "exclusive": True,
            "note": "GPU exclusivity is not applicable to a CPU run",
            "concurrent_python_processes": [],
        }
    torch_device = torch.device(device)
    index = 0 if torch_device.index is None else int(torch_device.index)
    base = ["nvidia-smi", f"--id={index}"]
    try:
        gpu_query = subprocess.run(
            [
                *base,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        process_query = subprocess.run(
            [
                *base,
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "query_succeeded": False,
            "exclusive": None,
            "error": str(exc),
            "concurrent_python_processes": [],
        }
    gpu_values = [value.strip() for value in gpu_query.stdout.strip().split(",")]
    processes: list[dict[str, Any]] = []
    for line in process_query.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        raw_pid, raw_name = line.split(",", 1)
        try:
            pid = int(raw_pid.strip())
        except ValueError:
            continue
        name = raw_name.strip()
        if pid != os.getpid() and "python" in Path(name).name.casefold():
            processes.append({"pid": pid, "process_name": name})
    return {
        "query_succeeded": True,
        "exclusive": not processes,
        "gpu_index": index,
        "utilization_percent_at_preflight": (
            float(gpu_values[0]) if len(gpu_values) >= 1 else None
        ),
        "memory_used_mib_at_preflight": (
            float(gpu_values[1]) if len(gpu_values) >= 2 else None
        ),
        "memory_total_mib": (
            float(gpu_values[2]) if len(gpu_values) >= 3 else None
        ),
        "concurrent_python_processes": processes,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(result), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _actual_device(config: Mapping[str, Any]) -> tuple[str, str, str | None]:
    requested = str(config.get("device", "auto"))
    try:
        device = resolve_device(requested)
    except RuntimeError as exc:
        return requested, "UNAVAILABLE", str(exc)
    if not device.startswith("cuda"):
        name = cpu_device_name() if device == "cpu" else f"NON-CUDA ({device})"
        return device, name, None
    try:
        name = torch.cuda.get_device_name(torch.device(device))
    except (AssertionError, RuntimeError, ValueError) as exc:
        return device, "UNAVAILABLE", str(exc)
    return device, name, None


def cpu_device_name() -> str:
    """Return a useful CPU model name without requiring an optional package."""

    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _value_type = winreg.QueryValueEx(key, "ProcessorNameString")
            name = " ".join(str(value).strip().split())
            if name:
                return name
        except (OSError, ImportError):
            pass
    for candidate in (platform.processor(), platform.uname().processor):
        name = " ".join(str(candidate).strip().split())
        if name:
            return name
    return "CPU"


def _measurement_definition(
    *, device: str, timing_scope: str, completed: bool
) -> dict[str, Any]:
    return {
        "completed": completed,
        "statistic": SUPPORTED_METRIC,
        "percentile": 50.0,
        "percentile_method": PERCENTILE_METHOD,
        "definition": (
            "50th percentile (median) of synchronized single-image wall-clock latency "
            "after warmup; NumPy linear interpolation; strict '<' limits"
        ),
        "synchronization": (
            "torch.cuda.synchronize before and after each timed run"
            if device.startswith("cuda")
            else "not required for CPU"
        ),
        "cold_start_included": False,
        "timing_scope": timing_scope,
        "input_decode_and_target_resize_excluded": True,
    }


def _base_result(
    config: Mapping[str, Any],
    *,
    target: PerformanceTarget,
    targets: Sequence[PerformanceTarget],
    device: str,
    device_name: str,
    resolution: tuple[int, int],
    warmup: int,
    runs: int,
    timing_scope: str,
    performance: Mapping[str, Any],
    hardware_detection_error: str | None,
    gpu_exclusivity: Mapping[str, Any],
) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    result: dict[str, Any] = {
        "report_schema": "aoi.benchmark.v3",
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "selected_standard": target.target_id,
        "standard": target.as_definition(),
        "available_standards": {
            item.target_id: item.as_definition() for item in targets
        },
        "category": str(config.get("category", "")).strip(),
        "resolution": list(resolution),
        "batch_size": target.batch_size,
        "device": device,
        "device_name": device_name,
        "measurement_note": config.get("measurement_note"),
        "measurement_policy": {
            "allow_nonexclusive_measurement": _allow_nonexclusive_measurement(
                config
            ),
            "exclusive_gpu_required_for_claim": (
                target.required_device_type == "cuda"
            ),
            "competition_claim_supported": (
                target.authority == "competition_official"
            ),
        },
        "runs": runs,
        "warmup_runs": warmup,
        "target_size": resolution[0],
        "target_hardware": target.target_hardware,
        "target_ms": target.limit_ms,
        "mean_ms": None,
        "p50_ms": None,
        "max_ms": None,
        "p50_under_target": None,
        "latency_samples_ms": [],
        "measurement": _measurement_definition(
            device=device, timing_scope=timing_scope, completed=False
        ),
        "environment": _runtime_environment(device),
        "gpu_exclusivity": dict(gpu_exclusivity),
        "performance": dict(performance),
        "status": performance["status"],
        "scope": timing_scope,
        "evidence": {
            "config": _file_evidence(Path(str(config["_config_path"]))),
            "implementation": {
                "benchmarking": _file_evidence(Path(__file__)),
                "runtime": _file_evidence(
                    package_root / "runtime" / "runtime.py"
                ),
            },
        },
    }
    if hardware_detection_error is not None:
        result["hardware_detection_error"] = hardware_detection_error
    return result


def _emit_report(output: Path, result: Mapping[str, Any]) -> None:
    _write_report(output, result)
    print(yaml.safe_dump(dict(result), sort_keys=False, allow_unicode=True))


def run(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    schema_version = config.get("schema_version")
    if schema_version != BENCHMARK_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "benchmark config schema_version must be "
            f"{BENCHMARK_CONFIG_SCHEMA_VERSION}; "
            f"got {schema_version!r}"
        )
    targets = load_performance_targets(config)
    target = get_selected_target(config)
    primary_target_id = target.target_id
    warmup, runs = validate_benchmark_config(config, target)
    allow_nonexclusive = _allow_nonexclusive_measurement(config)
    resolution = target.required_resolution
    width, height = resolution
    device, device_name, hardware_detection_error = _actual_device(config)
    gpu_exclusivity = _gpu_exclusivity(device)
    timing_scope = str(
        config.get(
            "timing_scope",
            "model/runtime and structured inspection; disk decode and output encoding excluded",
        )
    )
    output = resolve_path(
        config,
        str(
            config.get(
                "output", f"outputs/benchmark_{target.target_id}.yaml"
            )
        ),
    )
    preflight = evaluate_performance_targets(
        targets,
        primary_target_id=primary_target_id,
        device=device,
        device_name=device_name,
        resolution=resolution,
        warmup=warmup,
        runs=runs,
        p50_ms=None,
        exclusive_gpu=gpu_exclusivity["exclusive"],
    )
    result = _base_result(
        config,
        target=target,
        targets=targets,
        device=device,
        device_name=device_name,
        resolution=resolution,
        warmup=warmup,
        runs=runs,
        timing_scope=timing_scope,
        performance=preflight,
        hardware_detection_error=hardware_detection_error,
        gpu_exclusivity=gpu_exclusivity,
    )
    primary_validation = preflight["validation"]
    protocol_ready = all(
        bool(primary_validation[field])
        for field in (
            "hardware_match",
            "resolution_match",
            "warmup_sufficient",
            "runs_sufficient",
        )
    )
    exclusivity_allows_measurement = bool(
        target.required_device_type != "cuda"
        or primary_validation["exclusive_gpu_verified"]
        or (
            allow_nonexclusive
            and gpu_exclusivity["query_succeeded"]
            and gpu_exclusivity["exclusive"] is False
        )
    )
    eligible_to_measure = protocol_ready and exclusivity_allows_measurement
    if not eligible_to_measure:
        _emit_report(output, result)
        return 2

    if target.diagnostic_only:
        measurement_mode = "cpu_diagnostic"
    elif gpu_exclusivity["exclusive"] is True:
        measurement_mode = "claim_protocol"
    else:
        measurement_mode = "contended_diagnostic"
    result["measurement_mode"] = measurement_mode
    if measurement_mode == "cpu_diagnostic":
        result["measurement_warning"] = (
            "CPU P50 is evaluated against the project diagnostic limit of "
            "2000 ms. This result is not eligible for a competition GPU "
            "acceptance claim."
        )
    elif measurement_mode == "contended_diagnostic":
        result["measurement_warning"] = (
            "Measurement continued by explicit opt-in while another Python GPU "
            "process was active. Latency is diagnostic only and is not eligible "
            "for a competition or internal acceptance claim."
        )

    # Expensive loading starts after the immutable hardware/sample protocol is
    # valid and either exclusivity or explicit diagnostic opt-in is established.
    pointer_value = config.get("active_models_pointer")
    if not isinstance(pointer_value, str) or not pointer_value.strip():
        raise ValueError("active_models_pointer is required")
    pointer = resolve_path(config, pointer_value)
    payload = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
    result["evidence"]["active_models_pointer"] = _file_evidence(pointer)
    category = str(config.get("category", "")).strip()
    models = payload.get("models", {})
    if category not in models:
        raise KeyError(
            f"category {category!r} has no adapted model; available: {sorted(models)}"
        )
    bundle = resolve_artifact(models[category], base=pointer.parent)
    bundle_payload = yaml.safe_load(bundle.read_text(encoding="utf-8")) or {}
    bundle_evidence: dict[str, Any] = _file_evidence(bundle)
    bundle_evidence["bundle_hash"] = (
        bundle_payload.get("bundle_hash")
        if isinstance(bundle_payload, Mapping)
        else None
    )
    result["evidence"]["model_bundle"] = bundle_evidence
    image_dir_value = config.get("image_dir")
    if not isinstance(image_dir_value, str) or not image_dir_value.strip():
        raise ValueError("image_dir is required")
    image_dir = resolve_path(config, image_dir_value)
    images = image_files(image_dir)
    if not images:
        raise ValueError("no benchmark images")
    runtime = AdaptedRuntime(bundle, device=device)
    runtime_integrity = getattr(runtime, "integrity", None)
    if not isinstance(runtime_integrity, Mapping):
        raise TypeError("adapted runtime must expose an integrity receipt")
    bundle_evidence["runtime_integrity"] = dict(runtime_integrity)
    source = [read_rgb(p) for p in images[: min(8, len(images))]]
    prepared = [cv2.resize(img, (width, height)) for img in source]

    def predict(image: np.ndarray, sample_id: str) -> None:
        runtime.predict(image, sample_id)

    for i in range(warmup):
        predict(prepared[i % len(prepared)], f"warmup-{i}")
    values: list[float] = []
    for i in range(runs):
        image = prepared[i % len(prepared)]
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        started = time.perf_counter_ns()
        predict(image, f"run-{i}")
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    arr = np.asarray(values, dtype=np.float64)
    p50_ms = _percentile(arr, 50.0)
    target_evaluation = evaluate_performance_targets(
        targets,
        primary_target_id=primary_target_id,
        device=device,
        device_name=device_name,
        resolution=resolution,
        warmup=warmup,
        runs=runs,
        p50_ms=p50_ms,
        exclusive_gpu=gpu_exclusivity["exclusive"],
    )
    result.update(
        {
            "mean_ms": float(arr.mean()),
            "p50_ms": p50_ms,
            "max_ms": float(arr.max()),
            "latency_samples_ms": [float(value) for value in values],
            "measurement": _measurement_definition(
                device=device, timing_scope=timing_scope, completed=True
            ),
            "performance": target_evaluation,
            "status": target_evaluation["status"],
            "p50_under_target": target_evaluation["validation"][
                "latency_p50_strictly_under_target"
            ],
        }
    )
    _emit_report(output, result)
    return 0 if target_evaluation["passed"] else 2
