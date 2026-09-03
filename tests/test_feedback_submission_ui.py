from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

import feedback_ui
from aoi.feedback import FeedbackStore
from feedback_ui import FeedbackApp


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value
        self.history = [value]

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value
        self.history.append(value)


class FakeText:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self, *_args: object) -> str:
        return self.value

    def delete(self, *_args: object) -> None:
        self.value = ""


class FakeButton:
    def __init__(self) -> None:
        self.options = {"state": "normal", "text": "提交反馈"}
        self.history: list[dict[str, object]] = []

    def configure(self, **kwargs: object) -> None:
        self.options.update(kwargs)
        self.history.append(dict(kwargs))


class FakeRoot:
    def __init__(self) -> None:
        self.update_count = 0

    def update_idletasks(self) -> None:
        self.update_count += 1


def _make_app(tmp_path: Path) -> FeedbackApp:
    app = FeedbackApp.__new__(FeedbackApp)
    app.root = FakeRoot()
    app.store = FeedbackStore(tmp_path / "feedback")
    app._feedback_submit_inflight = False
    app.feedback_submit_button = FakeButton()
    app.feedback_submit_state_var = FakeVar("IDLE")
    app.feedback_submit_status_var = FakeVar("提交状态：等待提交")
    app.status_var = FakeVar("就绪")
    app.category_var = FakeVar("Class1")
    app.operator_label_var = FakeVar("正常")
    app.dataset_role_var = FakeVar("独立验证")
    app.defect_type_var = FakeVar("")
    app.note_text = FakeText("")
    app.image_bytes = b"same-image"
    app.image_path = Path("sample.png")
    app.image_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    app.defect_region = None
    app.source_type = "image"
    app.source_name = "sample.png"
    app.video_frame_index = 0
    app.detection = {
        "category": "Class1",
        "model_version": "model-a",
        "feedback_runtime_state": {},
        "result": {
            "decision": False,
            "model_decision": False,
            "overall_decision": False,
            "fusion_score": 0.1,
            "fusion_threshold": 0.5,
            "branch_margins": {
                "general": 0.0,
                "local": 0.0,
                "known": 0.0,
            },
        },
    }
    app._invalidate_feedback_candidate = lambda: None
    app._refresh_feedback_records = lambda: None
    return app


def _capture_dialogs(monkeypatch, *, decision: bool | None = None):
    events: list[tuple[str, str, str, dict[str, object]]] = []

    def record(kind: str):
        def inner(title: str, text: str, **kwargs: object):
            events.append((kind, title, text, kwargs))
            return decision if kind == "ask" else None

        return inner

    monkeypatch.setattr(feedback_ui.messagebox, "showinfo", record("info"))
    monkeypatch.setattr(feedback_ui.messagebox, "showwarning", record("warning"))
    monkeypatch.setattr(feedback_ui.messagebox, "showerror", record("error"))
    monkeypatch.setattr(feedback_ui.messagebox, "askyesnocancel", record("ask"))
    return events


def _assert_submission_button_restored(app: FeedbackApp) -> None:
    assert app._feedback_submit_inflight is False
    assert app.feedback_submit_button.options == {
        "state": "normal",
        "text": "提交反馈",
    }
    assert app.root.update_count >= 2


def test_scenarios_a_b_new_then_identical_feedback_exists(tmp_path, monkeypatch) -> None:
    app = _make_app(tmp_path)
    events = _capture_dialogs(monkeypatch)

    app._submit_feedback()

    assert len(app.store.records()) == 1
    feedback_id = app.store.records()[0]["feedback_id"]
    assert app.feedback_submit_state_var.get() == "SUCCESS"
    assert events[-1][1] == "反馈提交成功"
    assert feedback_id in events[-1][2]
    assert events[-1][3]["parent"] is app.root
    _assert_submission_button_restored(app)

    app._submit_feedback()

    assert len(app.store.records()) == 1
    assert app.feedback_submit_state_var.get() == "EXISTS"
    assert events[-1][1] == "反馈已存在"
    assert feedback_id in events[-1][2]
    _assert_submission_button_restored(app)


def test_scenario_c_changed_feedback_replace_keeps_one_id(tmp_path, monkeypatch) -> None:
    app = _make_app(tmp_path)
    _capture_dialogs(monkeypatch)
    app._submit_feedback()
    original = app.store.records()[0]
    app.operator_label_var.set("异常")
    app.defect_type_var.set("外观缺陷")
    events = _capture_dialogs(monkeypatch, decision=True)

    app._submit_feedback()

    rows = app.store.records()
    assert len(rows) == 1
    assert rows[0]["feedback_id"] == original["feedback_id"]
    assert rows[0]["operator_label"] == 1
    assert rows[0]["replacement_count"] == 1
    assert [event[0] for event in events] == ["ask", "info"]
    assert events[-1][1] == "反馈替换成功"
    assert app.feedback_submit_state_var.get() == "REPLACED"
    assert all(event[3]["parent"] is app.root for event in events)
    _assert_submission_button_restored(app)


def test_scenario_d_changed_feedback_keep_original(tmp_path, monkeypatch) -> None:
    app = _make_app(tmp_path)
    _capture_dialogs(monkeypatch)
    app._submit_feedback()
    original = app.store.records_path.read_bytes()
    app.note_text.value = "changed"
    events = _capture_dialogs(monkeypatch, decision=False)

    app._submit_feedback()

    assert app.store.records_path.read_bytes() == original
    assert [event[0] for event in events] == ["ask", "info"]
    assert events[-1][1] == "已保留原反馈"
    assert app.feedback_submit_state_var.get() == "CANCELLED"
    _assert_submission_button_restored(app)


def test_scenario_e_changed_feedback_cancel(tmp_path, monkeypatch) -> None:
    app = _make_app(tmp_path)
    _capture_dialogs(monkeypatch)
    app._submit_feedback()
    original = app.store.records_path.read_bytes()
    app.note_text.value = "changed"
    events = _capture_dialogs(monkeypatch, decision=None)

    app._submit_feedback()

    assert app.store.records_path.read_bytes() == original
    assert [event[0] for event in events] == ["ask", "info"]
    assert events[-1][1] == "已取消反馈提交"
    assert app.feedback_submit_state_var.get() == "CANCELLED"
    _assert_submission_button_restored(app)


def test_scenario_f_protected_training_feedback_cannot_become_validation(
    tmp_path, monkeypatch
) -> None:
    app = _make_app(tmp_path)
    app.dataset_role_var.set("训练反馈")
    _capture_dialogs(monkeypatch)
    app._submit_feedback()
    existing = app.store.records()[0]
    (app.store.candidates / "candidate-test.yaml").write_text(
        yaml.safe_dump(
            {
                "feedback_snapshot": {
                    "training": [{"feedback_id": existing["feedback_id"]}],
                    "validation": [],
                }
            }
        ),
        encoding="utf-8",
    )
    app.dataset_role_var.set("独立验证")
    events = _capture_dialogs(monkeypatch, decision=True)

    app._submit_feedback()

    assert app.store.records()[0]["dataset_role"] == "training"
    assert [event[0] for event in events] == ["ask", "error"]
    assert events[-1][1] == "无法替换反馈"
    assert "训练集与独立验证集隔离" in events[-1][2]
    assert app.feedback_submit_state_var.get() == "ERROR"
    _assert_submission_button_restored(app)


def test_scenario_g_unprotected_note_change_can_replace(tmp_path, monkeypatch) -> None:
    app = _make_app(tmp_path)
    _capture_dialogs(monkeypatch)
    app._submit_feedback()
    app.note_text.value = "operator reviewed twice"
    events = _capture_dialogs(monkeypatch, decision=True)

    app._submit_feedback()

    assert app.store.records()[0]["note"] == "operator reviewed twice"
    assert len(app.store.records()) == 1
    assert events[-1][1] == "反馈替换成功"
    _assert_submission_button_restored(app)


def test_scenario_h_nested_rapid_clicks_execute_only_once(tmp_path, monkeypatch) -> None:
    app = _make_app(tmp_path)
    events: list[str] = []

    def showinfo(title: str, _text: str, **kwargs: object) -> None:
        events.append(title)
        assert kwargs["parent"] is app.root
        for _index in range(4):
            app._submit_feedback()

    monkeypatch.setattr(feedback_ui.messagebox, "showinfo", showinfo)
    monkeypatch.setattr(feedback_ui.messagebox, "showwarning", lambda *_a, **_k: None)
    monkeypatch.setattr(feedback_ui.messagebox, "showerror", lambda *_a, **_k: None)

    app._submit_feedback()

    assert events == ["反馈提交成功"]
    assert len(app.store.records()) == 1
    assert app.feedback_submit_button.history[0] == {
        "state": "disabled",
        "text": "提交中...",
    }
    _assert_submission_button_restored(app)


def test_scenario_i_service_error_is_visible_and_button_recovers(
    tmp_path, monkeypatch
) -> None:
    app = _make_app(tmp_path)

    class FailingStore:
        def save_feedback(self, **_kwargs: object) -> None:
            raise OSError("simulated write failure")

    app.store = FailingStore()
    events = _capture_dialogs(monkeypatch)

    app._submit_feedback()

    assert [event[0] for event in events] == ["error"]
    assert "文件写入失败" in events[0][2]
    assert app.feedback_submit_state_var.get() == "ERROR"
    _assert_submission_button_restored(app)


def test_scenario_j_saved_but_refresh_failed_is_not_resaved(
    tmp_path, monkeypatch
) -> None:
    app = _make_app(tmp_path)

    def fail_refresh() -> None:
        raise RuntimeError("tree refresh failed")

    app._refresh_feedback_records = fail_refresh
    events = _capture_dialogs(monkeypatch)

    app._submit_feedback()

    assert len(app.store.records()) == 1
    assert [event[0] for event in events] == ["info", "warning"]
    assert events[-1][1] == "反馈已保存，但页面刷新失败"
    assert "不会再次保存" in events[-1][2]
    assert app.feedback_submit_state_var.get() == "ERROR"
    _assert_submission_button_restored(app)


def test_validation_warning_has_parent_and_button_recovers(tmp_path, monkeypatch) -> None:
    app = _make_app(tmp_path)
    app.detection = None
    events = _capture_dialogs(monkeypatch)

    app._submit_feedback()

    assert [event[0] for event in events] == ["warning"]
    assert events[0][3]["parent"] is app.root
    assert app.feedback_submit_state_var.get() == "ERROR"
    _assert_submission_button_restored(app)


def test_relative_improvement_candidate_displays_validation_delta() -> None:
    app = FeedbackApp.__new__(FeedbackApp)
    app.busy = True
    app.candidate = None
    app.detection = None
    app.base_metrics_var = FakeVar()
    app.candidate_metrics_var = FakeVar()
    app.validation_metrics_var = FakeVar()
    app.improvement_var = FakeVar()
    app.candidate_state_var = FakeVar("RUNNING")
    app.revision_detail_var = FakeVar()
    app.sample_comparison_var = FakeVar()
    app.status_var = FakeVar()
    candidate = {
        "revision_id": "revision-relative",
        "candidate_threshold": 0.42,
        "parent_threshold": 0.50,
        # Deliberately use very different training deltas.  The operator-facing
        # improvement must come from the independent validation metrics below.
        "base_metrics": {
            "precision": 0.20,
            "recall": 0.20,
            "f1": 0.20,
            "fpr": 0.80,
        },
        "candidate_metrics": {
            "precision": 0.90,
            "recall": 1.00,
            "f1": 0.95,
            "fpr": 0.10,
        },
        "parent_validation_metrics": {
            "precision": 1 / 3,
            "recall": 0.50,
            "f1": 0.40,
            "fpr": 1.00,
        },
        "validation_metrics": {
            "precision": 0.50,
            "recall": 0.50,
            "f1": 0.50,
            "fpr": 0.50,
        },
        "validation": {
            "passed": True,
            "acceptance_mode": "relative_improvement",
            "minimum_recall": 0.90,
        },
        "parameter_fingerprint": {"candidate": {"overall_sha256": "b" * 64}},
        "feature_count": 3,
    }

    app._revision_candidate_complete(candidate)

    assert app.busy is False
    assert app.candidate is candidate
    assert app.candidate_state_var.get() == "READY FOR REVIEW"
    assert "改善成功（未达到 90.0% 目标）" in app.validation_metrics_var.get()
    assert (
        app.improvement_var.get()
        == "独立验证：召回率 50.0% → 50.0%（+0.0 个百分点） · "
        "F1 40.0% → 50.0%（+10.0 个百分点）"
    )
    assert "相对改善但未达目标门槛" in app.status_var.get()


def test_validation_presentation_recomputes_missing_relative_mode() -> None:
    presentation = feedback_ui._feedback_validation_presentation(
        {
            "validation": {"minimum_recall": 0.90},
            "parent_validation_metrics": {"recall": 0.50, "f1": 0.40},
            "validation_metrics": {"recall": 0.50, "f1": 0.50},
        }
    )

    assert presentation["mode"] == "relative_improvement"
    assert presentation["status"] == "改善成功（未达到 90.0% 目标）"
