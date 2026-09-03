from __future__ import annotations

from pathlib import Path

from feedback_ui import CPU_DIAGNOSTIC_STANDARD, FeedbackApp


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class FakeWidget:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def configure(self, **options: object) -> None:
        self.options.update(options)


def test_cpu_mode_exposes_p50_two_second_target_in_ui() -> None:
    app = FeedbackApp.__new__(FeedbackApp)
    app.benchmark_standard_var = FakeVar(CPU_DIAGNOSTIC_STANDARD)
    app.benchmark_target_device_var = FakeVar()
    app.benchmark_target_p50_var = FakeVar()
    app.benchmark_run_button = FakeWidget()
    app.benchmark_nonexclusive_check = FakeWidget()

    app._apply_benchmark_mode()

    assert app.benchmark_target_device_var.get() == "当前 CPU"
    assert app.benchmark_target_p50_var.get() == "< 2000 ms"
    assert app.benchmark_run_button.options["text"] == "开始 CPU 测试"
    assert app.benchmark_nonexclusive_check.options["state"] == "disabled"


def test_cpu_result_shows_latency_and_non_competition_disclaimer() -> None:
    app = FeedbackApp.__new__(FeedbackApp)
    app.busy = True
    app.benchmark_run_button = FakeWidget()
    app.open_benchmark_report_button = FakeWidget()
    app.benchmark_p50_var = FakeVar()
    app.benchmark_actual_device_var = FakeVar()
    app.benchmark_resolution_var = FakeVar()
    app.benchmark_hardware_var = FakeVar()
    app.benchmark_samples_var = FakeVar()
    app.benchmark_status_var = FakeVar()
    app.benchmark_hero_var = FakeVar()
    app.status_var = FakeVar()
    app.benchmark_validation_vars = {
        "hardware": FakeVar(),
        "resolution": FakeVar(),
        "latency": FakeVar(),
        "protocol": FakeVar(),
    }
    report = {
        "selected_standard": CPU_DIAGNOSTIC_STANDARD,
        "measurement_mode": "cpu_diagnostic",
        "device": "cpu",
        "device_name": "12th Gen Intel Core i7-12700K",
        "resolution": [2500, 2500],
        "warmup_runs": 20,
        "runs": 100,
        "p50_ms": 820.25,
        # Retained backend diagnostics must not be exposed or used by the UI.
        "p95_ms": 1250.5,
        "p50_under_target": True,
        "gpu_exclusivity": {
            "exclusive": True,
            "note": "GPU exclusivity is not applicable to a CPU run",
        },
        "measurement": {"completed": True},
        "performance": {
            "status": "PASS",
            "passed": True,
            "claim_eligible": False,
            "diagnostic_only": True,
            "validation": {
                "hardware_match": True,
                "resolution_match": True,
                "warmup_sufficient": True,
                "runs_sufficient": True,
                "latency_p50_strictly_under_target": True,
                "exclusive_gpu_verified": None,
            },
        },
    }
    path = Path("cpu_report.yaml")

    app._benchmark_complete(report, path, 0)

    assert app.busy is False
    assert app.last_benchmark_report == path
    assert app.benchmark_actual_device_var.get() == "12th Gen Intel Core i7-12700K"
    assert app.benchmark_p50_var.get() == "820.25 ms"
    assert app.benchmark_hardware_var.get() == "不适用（CPU）"
    assert app.benchmark_samples_var.get() == "100"
    assert app.benchmark_hero_var.get() == "PASS"
    assert app.benchmark_validation_vars["latency"].get() == "PASS"
    assert "P50 小于 2 秒：是" in app.benchmark_status_var.get()
    assert "1250.50" not in app.benchmark_status_var.get()
    assert "不作为竞赛 GPU 验收声明" in app.benchmark_status_var.get()
    assert "CPU 性能诊断达标" in app.status_var.get()


def test_p95_only_legacy_report_is_not_used_for_p50_acceptance() -> None:
    app = FeedbackApp.__new__(FeedbackApp)
    app.busy = True
    app.benchmark_run_button = FakeWidget()
    app.open_benchmark_report_button = FakeWidget()
    app.benchmark_p50_var = FakeVar()
    app.benchmark_actual_device_var = FakeVar()
    app.benchmark_resolution_var = FakeVar()
    app.benchmark_hardware_var = FakeVar()
    app.benchmark_samples_var = FakeVar()
    app.benchmark_status_var = FakeVar()
    app.benchmark_hero_var = FakeVar()
    app.status_var = FakeVar()
    app.benchmark_validation_vars = {
        "hardware": FakeVar(),
        "resolution": FakeVar(),
        "latency": FakeVar(),
        "protocol": FakeVar(),
    }
    report = {
        "selected_standard": CPU_DIAGNOSTIC_STANDARD,
        "measurement_mode": "cpu_diagnostic",
        "device_name": "Historical CPU",
        "resolution": [2500, 2500],
        "runs": 100,
        "p95_ms": 900.0,
        "p95_under_target": True,
        "standard": {"metric": "p95_ms", "limit_ms": 2000.0},
        "measurement": {"completed": True},
        "performance": {
            "status": "PASS",
            "claim_eligible": False,
            "validation": {
                "hardware_match": True,
                "resolution_match": True,
                "warmup_sufficient": True,
                "runs_sufficient": True,
                "latency_p95_strictly_under_target": True,
            },
        },
    }
    path = Path("legacy_p95_report.yaml")

    app._benchmark_complete(report, path, 0)

    assert app.benchmark_p50_var.get() == "历史性能报告"
    assert app.benchmark_hero_var.get() == "DIAGNOSTIC"
    assert app.benchmark_validation_vars["latency"].get() == "WARNING · 需重测"
    assert "不用于当前 P50 验收" in app.benchmark_status_var.get()
    assert "重新运行 P50 验收" in app.status_var.get()
