from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import feedback_ui
from aoi.feedback import FeedbackStore
from feedback_ui import (
    FINAL_INDEPENDENT_TEST_MINIMUM,
    FeedbackApp,
    _acceptance_thresholds_from_percent,
    _empty_product_pointer,
    _load_saved_product_pointer,
    _read_model_pointer,
    _validate_managed_product_task_root,
)
from aoi.training.config import load_config
from aoi.ui.widgets import COMPETITION_STEPS


def _write_pointer(root: Path, *, dataset: str, category: str = "product_a") -> Path:
    bundle = root / "bundle.yaml"
    bundle.write_text("schema_version: 3\n", encoding="utf-8")
    pointer = root / "active_models.yaml"
    pointer.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": dataset,
                "models": {category: bundle.name},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return pointer


def test_empty_product_pointer_has_no_benchmark_categories() -> None:
    pointer = _empty_product_pointer()

    assert pointer["dataset"] == "custom"
    assert pointer["models"] == {}


def test_competition_flow_contains_only_the_aoi_learning_loop() -> None:
    assert [step[0] for step in COMPETITION_STEPS] == [
        "adapt",
        "detect",
        "feedback",
        "optimize",
        "acceptance",
    ]


def test_desktop_custom_training_base_is_product_scoped(
    require_training_assets,
) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "custom_training_base.yaml"
    )

    assert config.dataset == "custom"
    assert config.minimum_holdout_per_category == 0
    assert config.minimum_pooled_test == 0


def test_saved_product_pointer_restores_only_custom_task(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    task_root.mkdir()
    pointer = _write_pointer(task_root, dataset="custom")
    state = tmp_path / "active_task.yaml"
    state.write_text(
        yaml.safe_dump({"active_models_pointer": str(pointer)}),
        encoding="utf-8",
    )

    resolved, payload = _load_saved_product_pointer(state)

    assert resolved == pointer.resolve()
    assert payload["models"] == {"product_a": "bundle.yaml"}


def test_saved_product_pointer_rejects_dagm_autoload(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    task_root.mkdir()
    pointer = _write_pointer(task_root, dataset="dagm2007", category="Class1")
    state = tmp_path / "active_task.yaml"
    state.write_text(
        yaml.safe_dump({"active_models_pointer": str(pointer)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="只接受 custom"):
        _load_saved_product_pointer(state)


def test_model_pointer_dataset_modes_are_explicit(tmp_path: Path) -> None:
    pointer = _write_pointer(tmp_path, dataset="custom")

    resolved, payload = _read_model_pointer(pointer, expected_dataset="custom")

    assert resolved == pointer.resolve()
    assert payload["dataset"] == "custom"
    with pytest.raises(ValueError, match="只接受 dagm2007"):
        _read_model_pointer(pointer, expected_dataset="dagm2007")


def test_small_manifest_can_be_confirmed_once_as_diagnostic(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "holdout.yaml"
    manifest.write_text("schema_version: 1\n", encoding="utf-8")
    app = object.__new__(FeedbackApp)
    app._small_manifest_approval = None
    prompts: list[tuple[str, str, dict[str, object]]] = []

    def approve(title: str, text: str, **options: object) -> bool:
        prompts.append((title, text, options))
        return True

    app._askyesno = approve

    assert app._confirm_small_test_manifest(manifest, 80) is True
    assert app._confirm_small_test_manifest(manifest, 80) is True
    assert len(prompts) == 1
    assert "少于1000张，是否继续" in prompts[0][0]
    assert "开发诊断" in prompts[0][1]
    assert prompts[0][2]["default"] == "no"
    assert app._confirm_small_test_manifest(
        manifest,
        FINAL_INDEPENDENT_TEST_MINIMUM,
    ) is True


def test_ui_acceptance_threshold_percentages_are_strict_and_normalized() -> None:
    assert _acceptance_thresholds_from_percent("95", "10%", 5.0) == {
        "recall_floor": 0.95,
        "stage1_fpr_strict_upper_bound": 0.10,
        "stage2_fpr_strict_upper_bound": 0.05,
    }
    with pytest.raises(ValueError, match="严格FPR"):
        _acceptance_thresholds_from_percent("95", "5", "10")
    with pytest.raises(ValueError, match="基础FPR"):
        _acceptance_thresholds_from_percent("95", "0", "0")
    with pytest.raises(ValueError, match="召回率"):
        _acceptance_thresholds_from_percent("101", "10", "5")


def test_close_is_idempotent_and_reaches_quit_and_destroy() -> None:
    class RootStub:
        def __init__(self) -> None:
            self.quit_calls = 0
            self.destroy_calls = 0

        def after_cancel(self, _after_id: str) -> None:
            raise AssertionError("no timer should be cancelled in this test")

        def quit(self) -> None:
            self.quit_calls += 1

        def destroy(self) -> None:
            self.destroy_calls += 1

    class EventStub:
        def __init__(self) -> None:
            self.set_calls = 0

        def set(self) -> None:
            self.set_calls += 1

    root = RootStub()
    event = EventStub()
    releases: list[bool] = []
    app = object.__new__(FeedbackApp)
    app.root = root
    app.closing = False
    app._preview_resize_after_id = None
    app.video_cancel_event = event
    app._release_video = lambda: releases.append(True)

    app._close()
    app._close()

    assert app.closing is True
    assert event.set_calls == 1
    assert releases == [True]
    assert root.quit_calls == 1
    assert root.destroy_calls == 1


class _FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _FakeWidget:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def configure(self, **options: object) -> None:
        self.options.update(options)


def _removal_app(pointer: Path, payload: dict[str, object]) -> FeedbackApp:
    category = next(iter(payload["models"]))
    app = FeedbackApp.__new__(FeedbackApp)
    app.root = object()
    app.busy = False
    app.pointer_path = pointer
    app.pointer = payload
    app.categories = [category]
    app.current_category = category
    app.category_var = _FakeVar(category)
    app.category_combo = _FakeWidget()
    app.runtimes = {("old", "cpu"): object()}
    app.detection = {"old": True}
    app.candidate = {"old": True}
    app.last_benchmark_report = Path("old-benchmark.yaml")
    app.last_active_learning_manifest = Path("old-query.yaml")
    app.open_benchmark_report_button = _FakeWidget()
    app.open_active_learning_button = _FakeWidget()
    app.active_learning_status_var = _FakeVar("old")
    app.status_var = _FakeVar("old")
    app._reset_result = lambda: None
    app._refresh_all = lambda: None
    return app


def test_managed_task_removal_accepts_only_one_direct_product_directory(
    tmp_path: Path,
) -> None:
    managed = tmp_path / "custom_tasks"
    task = managed / "product_a"
    task.mkdir(parents=True)
    pointer = _write_pointer(task, dataset="custom", category="product_a")
    payload = yaml.safe_load(pointer.read_text(encoding="utf-8"))

    assert _validate_managed_product_task_root(
        pointer, payload, "product_a", managed_root=managed
    ) == task.resolve()

    external = tmp_path / "external"
    external.mkdir()
    external_pointer = _write_pointer(
        external, dataset="custom", category="product_a"
    )
    external_payload = yaml.safe_load(external_pointer.read_text(encoding="utf-8"))
    with pytest.raises(PermissionError, match="外部加载任务"):
        _validate_managed_product_task_root(
            external_pointer,
            external_payload,
            "product_a",
            managed_root=managed,
        )

    payload["models"]["product_b"] = payload["models"]["product_a"]
    with pytest.raises(PermissionError, match="单产品"):
        _validate_managed_product_task_root(
            pointer, payload, "product_a", managed_root=managed
        )


def test_only_detach_keeps_task_files_and_clears_current_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = tmp_path / "custom_tasks"
    task = managed / "product_a"
    task.mkdir(parents=True)
    pointer = _write_pointer(task, dataset="custom", category="product_a")
    payload = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    state = managed / "active_task.yaml"
    state.write_text(
        yaml.safe_dump({"active_models_pointer": str(pointer.resolve())}),
        encoding="utf-8",
    )
    monkeypatch.setattr(feedback_ui, "ACTIVE_TASK_STATE", state)
    app = _removal_app(pointer, payload)
    prompts: list[tuple[str, str, dict[str, object]]] = []
    app._askyesno = lambda title, text, **options: (
        prompts.append((title, text, options)) or True
    )
    warnings: list[str] = []
    app._showwarning = lambda _title, text, **_options: warnings.append(text)

    app._delete_category()

    assert task.is_dir()
    assert pointer.is_file()
    assert not state.exists()
    assert app.pointer["models"] == {}
    assert app.category_var.get() == feedback_ui.EMPTY_PRODUCT_LABEL
    assert app.runtimes == {}
    assert app.last_benchmark_report is None
    assert app.last_active_learning_manifest is None
    assert warnings == []
    assert prompts[0][0] == "仅移除使用"
    assert prompts[0][2]["default"] == "no"


def test_permanent_removal_cancel_has_no_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = tmp_path / "custom_tasks"
    task = managed / "product_a"
    task.mkdir(parents=True)
    bundle_dir = task / "run-a"
    bundle_dir.mkdir()
    bundle = bundle_dir / "model_bundle.yaml"
    bundle_hash = "a" * 64
    bundle.write_text(
        yaml.safe_dump({"run_id": "run-a", "bundle_hash": bundle_hash}),
        encoding="utf-8",
    )
    pointer = task / "active_models.yaml"
    payload = {
        "schema_version": 1,
        "dataset": "custom",
        "models": {"product_a": "run-a/model_bundle.yaml"},
    }
    pointer.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(feedback_ui, "CUSTOM_TASK_ROOT", managed)
    app = _removal_app(pointer, payload)

    class StoreStub:
        def __init__(self) -> None:
            self.purge_calls = 0

        def plan_product_evidence_purge(self, *_args: object, **_kwargs: object):
            return {
                "plan_sha256": "b" * 64,
                "counts": {
                    "records": 0,
                    "images": 0,
                    "candidates": 0,
                    "revisions": 0,
                    "feature_banks": 0,
                    "queries": 0,
                },
            }

        def purge_product_evidence(self, *_args: object, **_kwargs: object):
            self.purge_calls += 1
            raise AssertionError("cancelled removal must not purge evidence")

    store = StoreStub()
    app.store = store
    app._askyesno = lambda *_args, **_kwargs: False
    errors: list[str] = []
    app._showerror = lambda _title, text, **_options: errors.append(text)
    app._showwarning = lambda *_args, **_kwargs: None

    app._delete_category_files_and_audit()

    assert task.is_dir()
    assert pointer.is_file()
    assert store.purge_calls == 0
    assert errors == []


def test_permanent_removal_deletes_managed_task_and_exact_audit_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = tmp_path / "custom_tasks"
    task = managed / "product_a"
    task.mkdir(parents=True)
    bundle_dir = task / "run-a"
    bundle_dir.mkdir()
    bundle_hash = "a" * 64
    bundle = bundle_dir / "model_bundle.yaml"
    bundle.write_text(
        yaml.safe_dump({"run_id": "run-a", "bundle_hash": bundle_hash}),
        encoding="utf-8",
    )
    pointer = task / "active_models.yaml"
    payload = {
        "schema_version": 1,
        "dataset": "custom",
        "models": {"product_a": "run-a/model_bundle.yaml"},
    }
    pointer.write_text(yaml.safe_dump(payload), encoding="utf-8")
    state = managed / "active_task.yaml"
    state.write_text(
        yaml.safe_dump({"active_models_pointer": str(pointer.resolve())}),
        encoding="utf-8",
    )
    monkeypatch.setattr(feedback_ui, "CUSTOM_TASK_ROOT", managed)
    monkeypatch.setattr(feedback_ui, "ACTIVE_TASK_STATE", state)

    store = FeedbackStore(tmp_path / "feedback")
    store.save_feedback(
        image_bytes=b"product-audit-image",
        original_name="sample.png",
        category="product_a",
        model_version="run-a",
        bundle_hash=bundle_hash,
        result={
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
        },
        operator_label=0,
        defect_type="",
        note="",
    )
    store.save_feedback(
        image_bytes=b"other-product-image",
        original_name="other.png",
        category="product_b",
        model_version="run-b",
        bundle_hash="b" * 64,
        result={
            "decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {"general": 0.0, "local": 0.0, "known": 0.0},
        },
        operator_label=0,
        defect_type="",
        note="",
    )

    app = _removal_app(pointer, payload)
    app.store = store
    app._askyesno = lambda *_args, **_kwargs: True
    errors: list[str] = []
    warnings: list[str] = []
    information: list[str] = []
    app._showerror = lambda _title, text, **_options: errors.append(text)
    app._showwarning = lambda _title, text, **_options: warnings.append(text)
    app._showinfo = lambda _title, text, **_options: information.append(text)

    app._delete_category_files_and_audit()

    assert not task.exists()
    assert not state.exists()
    assert store.records(category="product_a") == []
    assert len(store.records(category="product_b")) == 1
    assert app.category_var.get() == feedback_ui.EMPTY_PRODUCT_LABEL
    assert errors == []
    assert warnings == []
    assert information and "永久移除产品任务 product_a" in information[0]
