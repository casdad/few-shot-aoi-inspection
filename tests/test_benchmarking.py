from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from aoi.cli import benchmarking


def _file_evidence(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _evaluate(
    standard: str,
    *,
    device_name: str,
    p50_ms: float | None,
    resolution: tuple[int, int] = (2500, 2500),
    warmup: int = 20,
    runs: int = 100,
) -> dict:
    return benchmarking.evaluate_performance_targets(
        benchmarking.load_performance_targets({}),
        primary_target_id=standard,
        device="cuda:0",
        device_name=device_name,
        resolution=resolution,
        warmup=warmup,
        runs=runs,
        p50_ms=p50_ms,
    )


def test_builtin_standards_fix_hardware_resolution_and_latency() -> None:
    official = benchmarking.PERFORMANCE_TARGETS["official_rtx2060"]
    internal = benchmarking.PERFORMANCE_TARGETS["internal_rtx4070"]
    cpu = benchmarking.PERFORMANCE_TARGETS["cpu_diagnostic"]

    assert official.required_resolution == (2500, 2500)
    assert internal.required_resolution == (2500, 2500)
    assert official.allowed_device_names == ("NVIDIA GeForce RTX 2060",)
    assert internal.allowed_device_names == ("NVIDIA GeForce RTX 4070",)
    assert official.limit_ms == 200.0
    assert internal.limit_ms == 100.0
    assert official.metric == internal.metric == cpu.metric == "p50_ms"
    assert official.minimum_warmup == internal.minimum_warmup == 20
    assert official.minimum_runs == internal.minimum_runs == 100
    assert cpu.required_resolution == (2500, 2500)
    assert cpu.required_device_type == "cpu"
    assert cpu.hardware_match_policy == "device_type_only"
    assert cpu.limit_ms == 2000.0
    assert cpu.minimum_warmup == 20
    assert cpu.minimum_runs == 100
    assert cpu.diagnostic_only is True
    assert benchmarking.BENCHMARK_CONFIG_SCHEMA_VERSION == 2
    assert benchmarking.BENCHMARK_REPORT_SCHEMA_VERSION == 3

    with pytest.raises(TypeError):
        benchmarking.PERFORMANCE_TARGETS["weakened"] = internal  # type: ignore[index]


def test_p50_uses_the_median_not_a_tail_percentile() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 100.0], dtype=np.float64)

    assert benchmarking._percentile(values, 50.0) == pytest.approx(2.5)


def test_benchmark_config_v3_is_rejected_because_v3_is_report_only(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "benchmark_v3_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "benchmark_standard": "cpu_diagnostic",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="benchmark config schema_version must be 2"
    ):
        benchmarking.run(config_path)


@pytest.mark.parametrize(
    "override",
    [
        {"target_size": 224},
        {"target_ms": 101},
        {"p50_limit_ms": 200},
        {"p95_limit_ms": 200},
        {"expected_device_name": "NVIDIA GeForce RTX 4070 Ti"},
        {"target_hardware": "NVIDIA RTX 2060 or lower"},
        {"batch_size": 2},
        {"enforce_primary_target": False},
        {"performance_targets": [{"limit_ms": 200}]},
    ],
)
def test_config_cannot_weaken_internal_standard(override: dict) -> None:
    config = {"benchmark_standard": "internal_rtx4070", **override}
    target = benchmarking.get_selected_target(config)

    with pytest.raises((TypeError, ValueError)):
        benchmarking.validate_benchmark_config(config, target)


def test_internal_standard_uses_strict_p50_boundary() -> None:
    passing = _evaluate(
        "internal_rtx4070",
        device_name="NVIDIA GeForce RTX 4070",
        p50_ms=99.999,
    )
    boundary = _evaluate(
        "internal_rtx4070",
        device_name="NVIDIA GeForce RTX 4070",
        p50_ms=100.0,
    )

    assert passing["status"] == benchmarking.STATUS_PASS
    assert passing["claim_eligible"] is True
    assert boundary["status"] == benchmarking.STATUS_FAIL_LATENCY
    assert boundary["status"] == "FAIL_P50_LATENCY"
    assert boundary["claim_eligible"] is False
    assert boundary["validation"]["latency_p50_strictly_under_target"] is False
    evaluated_target = next(
        target
        for target in boundary["targets"]
        if target["target_id"] == "internal_rtx4070"
    )
    assert evaluated_target["observed_p50_ms"] == 100.0
    assert "observed_p95_ms" not in evaluated_target


@pytest.mark.parametrize(
    "device_name",
    [
        "NVIDIA GeForce RTX 4070 Ti",
        "NVIDIA GeForce RTX 4070 Laptop GPU",
        "NVIDIA GeForce RTX 4060",
        "NVIDIA GeForce RTX 2060",
    ],
)
def test_internal_standard_rejects_similar_or_other_gpu_names(
    device_name: str,
) -> None:
    result = _evaluate(
        "internal_rtx4070", device_name=device_name, p50_ms=50.0
    )

    assert result["status"] == benchmarking.STATUS_FAIL_HARDWARE
    assert result["validation"]["hardware_match"] is False
    assert result["passed"] is False


def test_device_name_matching_only_normalizes_case_and_whitespace() -> None:
    result = _evaluate(
        "internal_rtx4070",
        device_name="  nvidia   geforce RTX 4070  ",
        p50_ms=99.0,
    )

    assert result["status"] == benchmarking.STATUS_PASS


def test_rtx4070_result_cannot_claim_official_rtx2060_compliance() -> None:
    result = _evaluate(
        "official_rtx2060",
        device_name="NVIDIA GeForce RTX 4070",
        p50_ms=50.0,
    )

    assert result["status"] == benchmarking.STATUS_FAIL_HARDWARE
    assert result["claim_eligible"] is False
    assert result["validation"]["latency_p50_strictly_under_target"] is True
    assert result["validation"]["passed"] is False


def test_exact_rtx2060_can_pass_official_strict_limit() -> None:
    result = _evaluate(
        "official_rtx2060",
        device_name="NVIDIA GeForce RTX 2060",
        p50_ms=199.999,
    )

    assert result["status"] == benchmarking.STATUS_PASS
    assert result["claim_eligible"] is True


def test_cpu_diagnostic_enforces_two_second_p50_without_claim() -> None:
    targets = benchmarking.load_performance_targets({})
    passing = benchmarking.evaluate_performance_targets(
        targets,
        primary_target_id="cpu_diagnostic",
        device="cpu",
        device_name="12th Gen Intel Core i7-12700K",
        resolution=(2500, 2500),
        warmup=20,
        runs=100,
        p50_ms=1999.999,
    )
    boundary = benchmarking.evaluate_performance_targets(
        targets,
        primary_target_id="cpu_diagnostic",
        device="cpu",
        device_name="Any CPU model",
        resolution=(2500, 2500),
        warmup=20,
        runs=100,
        p50_ms=2000.0,
    )

    assert passing["status"] == benchmarking.STATUS_PASS
    assert passing["passed"] is True
    assert passing["claim_eligible"] is False
    assert passing["diagnostic_only"] is True
    assert passing["validation"]["hardware_match"] is True
    assert passing["validation"]["exclusive_gpu_verified"] is None
    assert passing["validation"]["latency_p50_strictly_under_target"] is True
    assert boundary["status"] == benchmarking.STATUS_FAIL_LATENCY
    assert boundary["passed"] is False
    assert boundary["claim_eligible"] is False
    assert boundary["validation"]["latency_p50_strictly_under_target"] is False


@pytest.mark.parametrize(
    ("warmup", "runs"),
    [(0, 100), (19, 100), (20, 0), (20, 99)],
)
def test_sample_protocol_is_enforced(warmup: int, runs: int) -> None:
    result = _evaluate(
        "internal_rtx4070",
        device_name="NVIDIA GeForce RTX 4070",
        p50_ms=50.0,
        warmup=warmup,
        runs=runs,
    )

    assert result["status"] == benchmarking.STATUS_FAIL_PROTOCOL
    assert result["passed"] is False


def test_hardware_mismatch_writes_failure_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    output = tmp_path / "official_failure.yaml"
    config_path = config_dir / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "benchmark_standard": "official_rtx2060",
                "device": "cuda:0",
                "category": "Class1",
                "warmup": 20,
                "runs": 100,
                "active_models_pointer": "missing/active_models.yaml",
                "output": str(output),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmarking, "resolve_device", lambda _value: "cuda:0")
    monkeypatch.setattr(
        benchmarking.torch.cuda,
        "get_device_name",
        lambda _device: "NVIDIA GeForce RTX 4070",
    )
    monkeypatch.setattr(
        benchmarking.torch.cuda, "is_available", lambda: False
    )
    monkeypatch.setattr(
        benchmarking,
        "_gpu_exclusivity",
        lambda _device: {
            "query_succeeded": True,
            "exclusive": True,
            "concurrent_python_processes": [],
        },
    )

    exit_code = benchmarking.run(config_path)
    report = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["report_schema"] == "aoi.benchmark.v3"
    assert report["schema_version"] == 3
    assert report["status"] == benchmarking.STATUS_FAIL_HARDWARE
    assert report["measurement"]["completed"] is False
    assert report["latency_samples_ms"] == []
    assert report["performance"]["claim_eligible"] is False
    assert report["measurement_policy"]["competition_claim_supported"] is True
    assert report["evidence"]["config"] == _file_evidence(config_path)
    assert report["evidence"]["implementation"] == {
        "benchmarking": _file_evidence(Path(benchmarking.__file__)),
        "runtime": _file_evidence(
            Path(benchmarking.__file__).resolve().parents[1]
            / "runtime"
            / "runtime.py"
        ),
    }
    assert "active_models_pointer" not in report["evidence"]
    assert "model_bundle" not in report["evidence"]


def test_concurrent_gpu_python_process_blocks_claim() -> None:
    result = benchmarking.evaluate_performance_targets(
        benchmarking.load_performance_targets({}),
        primary_target_id="internal_rtx4070",
        device="cuda:0",
        device_name="NVIDIA GeForce RTX 4070",
        resolution=(2500, 2500),
        warmup=20,
        runs=100,
        p50_ms=50.0,
        exclusive_gpu=False,
    )

    assert result["status"] == benchmarking.STATUS_FAIL_GPU_EXCLUSIVITY
    assert result["claim_eligible"] is False
    assert result["validation"]["exclusive_gpu_verified"] is False


def test_nonexclusive_measurement_config_requires_boolean() -> None:
    target = benchmarking.PERFORMANCE_TARGETS["internal_rtx4070"]

    with pytest.raises(TypeError, match="must be a boolean"):
        benchmarking.validate_benchmark_config(
            {
                "benchmark_standard": "internal_rtx4070",
                "allow_nonexclusive_measurement": "yes",
            },
            target,
        )


def test_explicit_nonexclusive_mode_measures_but_cannot_pass_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "contended.yaml"
    pointer = tmp_path / "active_models.yaml"
    bundle = tmp_path / "model_bundle.yaml"
    bundle_hash = "b" * 64
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    pointer.write_text(
        yaml.safe_dump({"models": {"Class1": str(bundle)}}), encoding="utf-8"
    )
    bundle.write_text(
        yaml.safe_dump({"bundle_hash": bundle_hash}), encoding="utf-8"
    )
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "benchmark_standard": "internal_rtx4070",
                "device": "cuda:0",
                "category": "Class1",
                "warmup": 20,
                "runs": 100,
                "allow_nonexclusive_measurement": True,
                "active_models_pointer": str(pointer),
                "image_dir": str(image_dir),
                "output": str(output),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeRuntime:
        def __init__(self, _bundle: Path, *, device: str) -> None:
            assert device == "cuda:0"
            self.integrity = {
                "bundle_hash_verified": True,
                "artifact_hashes_verified": True,
            }

        def predict(self, _image: np.ndarray, _sample_id: str) -> dict:
            return {"decision": False}

    monkeypatch.setattr(
        benchmarking,
        "_actual_device",
        lambda _config: ("cuda:0", "NVIDIA GeForce RTX 4070", None),
    )
    monkeypatch.setattr(
        benchmarking,
        "_gpu_exclusivity",
        lambda _device: {
            "query_succeeded": True,
            "exclusive": False,
            "concurrent_python_processes": [
                {"pid": 123, "process_name": "python"}
            ],
        },
    )
    monkeypatch.setattr(
        benchmarking, "resolve_artifact", lambda *_args, **_kwargs: bundle
    )
    monkeypatch.setattr(
        benchmarking, "image_files", lambda _path: [tmp_path / "one.png"]
    )
    monkeypatch.setattr(
        benchmarking,
        "read_rgb",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(benchmarking, "AdaptedRuntime", FakeRuntime)
    monkeypatch.setattr(
        benchmarking.torch.cuda, "synchronize", lambda _device: None
    )

    exit_code = benchmarking.run(config)
    report = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["report_schema"] == "aoi.benchmark.v3"
    assert report["schema_version"] == 3
    assert report["measurement"]["completed"] is True
    assert report["measurement_mode"] == "contended_diagnostic"
    assert report["p50_ms"] is not None
    assert report["p50_under_target"] is True
    assert "p95_ms" not in report
    assert "p99_ms" not in report
    assert "p95_under_target" not in report
    assert report["measurement"]["statistic"] == "p50_ms"
    assert report["measurement"]["percentile"] == 50.0
    assert report["status"] == benchmarking.STATUS_FAIL_GPU_EXCLUSIVITY
    assert report["performance"]["claim_eligible"] is False
    assert report["measurement_policy"]["competition_claim_supported"] is False
    evidence = report["evidence"]
    assert evidence["config"] == _file_evidence(config)
    assert evidence["active_models_pointer"] == _file_evidence(pointer)
    assert evidence["model_bundle"] == {
        **_file_evidence(bundle),
        "bundle_hash": bundle_hash,
        "runtime_integrity": {
            "bundle_hash_verified": True,
            "artifact_hashes_verified": True,
        },
    }
    assert evidence["implementation"] == {
        "benchmarking": _file_evidence(Path(benchmarking.__file__)),
        "runtime": _file_evidence(
            Path(benchmarking.__file__).resolve().parents[1]
            / "runtime"
            / "runtime.py"
        ),
    }


def test_cpu_diagnostic_run_measures_and_never_claims_competition_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cpu_diagnostic.yaml"
    pointer = tmp_path / "active_models.yaml"
    bundle = tmp_path / "model_bundle.yaml"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    pointer.write_text(
        yaml.safe_dump({"models": {"product": str(bundle)}}), encoding="utf-8"
    )
    bundle.write_text(yaml.safe_dump({"bundle_hash": "c" * 64}), encoding="utf-8")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "benchmark_standard": "cpu_diagnostic",
                "device": "cpu",
                "category": "product",
                "warmup": 20,
                "runs": 100,
                "active_models_pointer": str(pointer),
                "image_dir": str(image_dir),
                "output": str(output),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeRuntime:
        def __init__(self, _bundle: Path, *, device: str) -> None:
            assert device == "cpu"
            self.integrity = {
                "bundle_hash_verified": True,
                "artifact_hashes_verified": True,
            }

        def predict(self, _image: np.ndarray, _sample_id: str) -> dict:
            return {"decision": False}

    monkeypatch.setattr(
        benchmarking,
        "_actual_device",
        lambda _config: ("cpu", "Test CPU Model", None),
    )
    monkeypatch.setattr(
        benchmarking, "resolve_artifact", lambda *_args, **_kwargs: bundle
    )
    monkeypatch.setattr(
        benchmarking, "image_files", lambda _path: [tmp_path / "one.png"]
    )
    monkeypatch.setattr(
        benchmarking,
        "read_rgb",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(benchmarking, "AdaptedRuntime", FakeRuntime)

    exit_code = benchmarking.run(config)
    report = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["report_schema"] == "aoi.benchmark.v3"
    assert report["schema_version"] == 3
    assert report["selected_standard"] == "cpu_diagnostic"
    assert report["measurement_mode"] == "cpu_diagnostic"
    assert report["measurement"]["completed"] is True
    assert report["device"] == "cpu"
    assert report["device_name"] == "Test CPU Model"
    assert report["p50_ms"] is not None
    assert report["p50_ms"] < 2000.0
    assert report["p50_under_target"] is True
    assert "p95_ms" not in report
    assert "p99_ms" not in report
    assert "p95_under_target" not in report
    assert report["standard"]["metric"] == "p50_ms"
    assert report["measurement"]["statistic"] == "p50_ms"
    assert report["measurement"]["percentile"] == 50.0
    assert report["performance"]["passed"] is True
    assert report["performance"]["claim_eligible"] is False
    assert report["standard"]["diagnostic_only"] is True
    assert report["measurement_policy"]["competition_claim_supported"] is False
    assert "not eligible" in report["measurement_warning"]


@pytest.mark.parametrize(
    "config_name",
    [
        "benchmark.yaml",
        "benchmark_cpu_diagnostic.yaml",
        "benchmark_dagm_demo.yaml",
        "benchmark_internal_rtx4070.yaml",
        "benchmark_official_rtx2060.yaml",
    ],
)
def test_shipped_benchmark_configs_use_p50_v3_output_paths(
    config_name: str,
) -> None:
    config = benchmarking.load_yaml(
        Path(benchmarking.__file__).resolve().parents[2] / "configs" / config_name
    )
    target = benchmarking.get_selected_target(config)

    assert config["schema_version"] == benchmarking.BENCHMARK_CONFIG_SCHEMA_VERSION
    assert target.metric == "p50_ms"
    output = str(config["output"])
    assert "p50_v3" in output
    assert not output.endswith("benchmark_internal_rtx4070.yaml")
    assert not output.endswith("benchmark_cpu_diagnostic.yaml")
