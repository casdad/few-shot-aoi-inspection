from __future__ import annotations

# Imports after the relaunch guard intentionally use the project interpreter.
# ruff: noqa: E402

import os
import queue
import logging
import math
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Callable
from uuid import uuid4


def _relaunch_with_project_python() -> None:
    """Make Windows double-click use the interpreter configured for this project."""
    if os.environ.get("AOI_FEEDBACK_UI_RELAUNCHED") == "1":
        return
    preferred = Path.home() / "anaconda3" / "envs" / "ZYL" / "pythonw.exe"
    if not preferred.is_file() or Path(sys.executable).resolve() == preferred.resolve():
        return
    environment = os.environ.copy()
    environment["AOI_FEEDBACK_UI_RELAUNCHED"] = "1"
    subprocess.Popen(
        [str(preferred), str(Path(__file__).resolve())],
        cwd=str(Path(__file__).resolve().parent),
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    raise SystemExit(0)


if __name__ == "__main__":
    _relaunch_with_project_python()

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # Graceful fallback keeps the standard Tk launcher usable.
    DND_FILES = None
    TkinterDnD = None

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageTk

from aoi.cli.benchmarking import cpu_device_name, run as run_benchmark
from aoi.cli.custom_evaluation import run as run_custom_evaluation
from aoi.custom_manifest import load_custom_test_manifest
from aoi.feedback import (
    DuplicateFeedbackError,
    FeedbackStore,
    apply_active_feedback,
    classification_metrics,
    propose_adaptive_revision,
    set_runtime_feature_bank,
)
from aoi.feedback.service import _revision_validation_acceptance
from aoi.feedback.active_learning import query_unlabeled_pool
from aoi.paths import PROJECT_ROOT, resolve_artifact
from aoi.runtime.config import resolve_device
from aoi.runtime.runtime import AdaptedRuntime
from aoi.training.config import load_config
from aoi.training.custom import build_custom_episode
from aoi.training.manifests import sha256_file
from aoi.training.pipeline import train_custom_episode
from aoi.ui.text import display_boolean, display_gate, display_status, display_text
from aoi.ui.text import status_tone as ui_status_tone
from aoi.ui.theme import (
    APP_BG,
    BORDER,
    DISABLED_BG,
    DISABLED_TEXT,
    ERROR_BG,
    ERROR_TEXT,
    HOVER_BG,
    INPUT_BORDER,
    PRIMARY,
    PRIMARY_HOVER,
    PRIMARY_LIGHT,
    SUCCESS_BG,
    SUCCESS_TEXT,
    SURFACE,
    SURFACE_SUBTLE,
    TABLE_HEADER_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    VIEWER_BG,
    VIEWER_TEXT,
    WAITING_BG,
    WAITING_TEXT,
    WARNING_BG,
    WARNING_TEXT,
)
from aoi.ui.widgets import (
    AdaptivePage,
    CompactProcessStrip,
    CompetitionFlowBar,
    CompetitionHint,
    ElidedTextLabel,
    MetricTargetCard,
    bind_path_entry,
    VerticalScrollPanel,
)


LOGGER = logging.getLogger(__name__)


CUSTOM_TASK_ROOT = PROJECT_ROOT / "outputs" / "custom_tasks"
ACTIVE_TASK_STATE = CUSTOM_TASK_ROOT / "active_task.yaml"
EMPTY_PRODUCT_LABEL = "（未加载产品）"
FEEDBACK_ROOT = PROJECT_ROOT / "outputs" / "feedback"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
CPU_DIAGNOSTIC_STANDARD = "cpu_diagnostic"
FINAL_INDEPENDENT_TEST_MINIMUM = 1000
DEFAULT_ACCEPTANCE_RECALL_PERCENT = 95.0
DEFAULT_ACCEPTANCE_STAGE1_FPR_PERCENT = 10.0
DEFAULT_ACCEPTANCE_STAGE2_FPR_PERCENT = 5.0


def _feedback_validation_presentation(
    candidate: dict[str, object],
) -> dict[str, object]:
    """Return an honest operator-facing summary of candidate validation."""

    raw_validation = candidate.get("validation")
    validation = dict(raw_validation) if isinstance(raw_validation, dict) else {}
    raw_parent = candidate.get(
        "parent_validation_metrics",
        validation.get("parent_metrics", validation.get("base_metrics")),
    )
    raw_current = candidate.get(
        "validation_metrics",
        validation.get("candidate_metrics"),
    )
    if not isinstance(raw_parent, dict) or not isinstance(raw_current, dict):
        raise ValueError("候选缺少独立验证前后指标。")
    parent = dict(raw_parent)
    current = dict(raw_current)
    parent_recall = float(parent["recall"])
    current_recall = float(current["recall"])
    parent_f1 = float(parent["f1"])
    current_f1 = float(current["f1"])
    recall_delta = current_recall - parent_recall
    f1_delta = current_f1 - parent_f1
    minimum_recall = float(validation.get("minimum_recall", 0.90))
    declared_mode = validation.get("acceptance_mode")
    if declared_mode is None:
        recalculated = _revision_validation_acceptance(
            parent,
            current,
            minimum_recall,
        )
        mode = str(recalculated["acceptance_mode"])
    else:
        mode = str(declared_mode)
    if mode == "relative_improvement":
        status = f"改善成功（未达到 {minimum_recall:.1%} 目标）"
    elif mode == "absolute_gate":
        status = "目标门槛通过"
    else:
        status = "独立验证未通过"
    improvement = (
        f"独立验证：召回率 {parent_recall:.1%} → {current_recall:.1%}"
        f"（{recall_delta * 100.0:+.1f} 个百分点） · "
        f"F1 {parent_f1:.1%} → {current_f1:.1%}"
        f"（{f1_delta * 100.0:+.1f} 个百分点）"
    )
    return {
        "mode": mode,
        "status": status,
        "improvement": improvement,
        "minimum_recall": minimum_recall,
        "recall_delta": recall_delta,
        "f1_delta": f1_delta,
    }


def _is_reparse_point(path: Path) -> bool:
    """Return whether one Windows path is a symlink, junction, or reparse point."""

    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return path.is_symlink() or bool(
        attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def _validate_managed_product_task_root(
    pointer_path: Path | None,
    pointer: dict[str, object],
    product: str,
    *,
    managed_root: Path | None = None,
) -> Path:
    """Resolve the one task directory that the UI is allowed to delete.

    Physical removal is deliberately limited to a single-product task created
    below ``outputs/custom_tasks``.  Paths referenced by the bundle (datasets,
    shared assets, manifests, or exported submissions) are never followed.
    """

    if pointer_path is None:
        raise ValueError("当前产品没有可删除的模型指针。")
    unresolved_pointer = Path(pointer_path)
    if _is_reparse_point(unresolved_pointer):
        raise PermissionError("模型指针不能是符号链接或目录联接。")
    resolved_pointer = unresolved_pointer.resolve(strict=True)
    if resolved_pointer.name != "active_models.yaml" or not resolved_pointer.is_file():
        raise PermissionError("只允许删除受管产品目录中的 active_models.yaml 任务。")

    root_unresolved = unresolved_pointer.parent
    if _is_reparse_point(root_unresolved):
        raise PermissionError("产品任务目录不能是符号链接或目录联接。")
    task_root = root_unresolved.resolve(strict=True)
    root = Path(managed_root or CUSTOM_TASK_ROOT).resolve(strict=True)
    if task_root == root or task_root.parent != root:
        raise PermissionError(
            "仅允许删除 outputs/custom_tasks 下的直接产品任务目录；外部加载任务只能“仅移除使用”。"
        )
    if task_root.name != product:
        raise PermissionError("产品名称与受管任务目录名称不一致，已拒绝删除。")

    models = pointer.get("models")
    if not isinstance(models, dict) or set(models) != {product}:
        raise PermissionError("物理删除只支持仅包含当前产品的单产品任务指针。")
    bundle_ref = models[product]
    if str(bundle_ref).strip() == "pending":
        raise ValueError("当前产品尚未生成模型文件。")
    bundle_path = resolve_artifact(bundle_ref, base=resolved_pointer.parent)
    try:
        bundle_path.relative_to(task_root)
    except ValueError as error:
        raise PermissionError("模型包不在当前受管任务目录内，已拒绝删除。") from error
    if not bundle_path.is_file() or _is_reparse_point(bundle_path):
        raise PermissionError("模型包缺失或属于链接文件，已拒绝删除。")

    pending = [task_root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                attributes = int(
                    getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
                )
                if entry.is_symlink() or attributes & int(
                    getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                ):
                    raise PermissionError(
                        f"任务目录含有链接或目录联接，已拒绝删除：{entry_path}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry_path)
    return task_root


def _task_disk_summary(task_root: Path) -> tuple[int, int]:
    """Return regular file count and byte size for a validated task tree."""

    count = 0
    total_bytes = 0
    for path in task_root.rglob("*"):
        if path.is_file():
            count += 1
            total_bytes += path.stat().st_size
    return count, total_bytes


def _acceptance_thresholds_from_percent(
    recall_floor: object,
    stage1_fpr_upper: object,
    stage2_fpr_upper: object,
) -> dict[str, float]:
    """Parse operator-entered percentages into strict evaluation fractions."""

    def parse(value: object, label: str) -> float:
        raw = str(value).strip()
        if raw.endswith("%"):
            raw = raw[:-1].strip()
        try:
            number = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}必须是数字百分比。") from error
        if not math.isfinite(number):
            raise ValueError(f"{label}必须是有限数字百分比。")
        return number

    recall = parse(recall_floor, "召回率下限")
    stage1 = parse(stage1_fpr_upper, "基础FPR上限")
    stage2 = parse(stage2_fpr_upper, "严格FPR上限")
    if not 0.0 <= recall <= 100.0:
        raise ValueError("召回率下限必须介于0%和100%之间。")
    if not 0.0 < stage1 <= 100.0:
        raise ValueError("基础FPR上限必须大于0%且不超过100%。")
    if not 0.0 < stage2 <= 100.0:
        raise ValueError("严格FPR上限必须大于0%且不超过100%。")
    if stage2 > stage1:
        raise ValueError("严格FPR上限不能高于基础FPR上限。")
    return {
        "recall_floor": recall / 100.0,
        "stage1_fpr_strict_upper_bound": stage1 / 100.0,
        "stage2_fpr_strict_upper_bound": stage2 / 100.0,
    }


def _empty_product_pointer() -> dict[str, object]:
    """Return an in-memory production workspace with no selected product."""

    return {
        "schema_version": 1,
        "dataset": "custom",
        "protocol": "private-100+30-product-task",
        "models": {},
    }


def _read_model_pointer(
    pointer_path: Path,
    *,
    expected_dataset: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Load and validate one model pointer without mutating UI state."""

    resolved = pointer_path.resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("active_models.yaml 必须是 YAML 映射")
    dataset = str(payload.get("dataset", "")).strip()
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(
            f"模型指针数据类型为 {dataset or '未知'}，当前入口只接受 {expected_dataset}"
        )
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("active_models.yaml 中没有可用模型")
    for category, bundle in models.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("模型产品/类别名称无效")
        if str(bundle).strip() == "pending":
            continue
        resolved_bundle = resolve_artifact(bundle, base=resolved.parent)
        if not resolved_bundle.is_file():
            raise FileNotFoundError(f"模型包不存在: {resolved_bundle}")
    return resolved, payload


def _load_saved_product_pointer(
    state_path: Path = ACTIVE_TASK_STATE,
) -> tuple[Path | None, dict[str, object]]:
    """Restore the last saved custom product pointer."""

    if not state_path.is_file():
        return None, _empty_product_pointer()
    state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
    if not isinstance(state, dict):
        raise TypeError("active_task.yaml 必须是 YAML 映射")
    raw_pointer = state.get("active_models_pointer")
    if not isinstance(raw_pointer, str) or not raw_pointer.strip():
        return None, _empty_product_pointer()
    pointer = resolve_artifact(raw_pointer, base=state_path.parent)
    return _read_model_pointer(pointer, expected_dataset="custom")


# ---- Shared UI design tokens (aliases keep the view code concise) ----
UI_BG = APP_BG
UI_PANEL = SURFACE
UI_PANEL_ALT = SURFACE_SUBTLE
UI_BORDER = BORDER
UI_TEXT = TEXT_PRIMARY
UI_MUTED = TEXT_SECONDARY
UI_SUBTLE = TEXT_TERTIARY
UI_PRIMARY = PRIMARY
UI_PRIMARY_DARK = PRIMARY_HOVER
UI_PRIMARY_SOFT = PRIMARY_LIGHT
UI_ACCENT = PRIMARY
UI_ACCENT_DARK = PRIMARY_HOVER
UI_ACCENT_SOFT = PRIMARY_LIGHT
UI_SUCCESS = SUCCESS_TEXT
UI_SUCCESS_SOFT = SUCCESS_BG
UI_WARNING = WARNING_TEXT
UI_WARNING_SOFT = WARNING_BG
UI_DANGER = ERROR_TEXT
UI_DANGER_SOFT = ERROR_BG
UI_CANVAS = VIEWER_BG
UI_CANVAS_TEXT = VIEWER_TEXT
UI_HEADING_BG = TABLE_HEADER_BG
UI_DISABLED = DISABLED_TEXT


class FeedbackApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AOI 质量检测工作台")
        self.root.geometry("1360x860")
        # Keep the responsive grid above the minimum at which controls would
        # need a second navigation mode.  1120x820 still fits typical 1080p
        # desktops while preserving the viewer and every primary action.
        self.root.minsize(1080, 700)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.store = FeedbackStore(FEEDBACK_ROOT)
        self.startup_pointer_warning: str | None = None
        try:
            self.pointer_path, self.pointer = _load_saved_product_pointer()
        except Exception as error:
            self.pointer_path = None
            self.pointer = _empty_product_pointer()
            self.startup_pointer_warning = str(error)
        self.categories = sorted(self.pointer.get("models", {})) or [EMPTY_PRODUCT_LABEL]
        self.runtimes: dict[tuple[str, str], tuple[AdaptedRuntime, str]] = {}
        self.image_path: Path | None = None
        self.image_bytes: bytes | None = None
        self.image_rgb: np.ndarray | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.heatmap: np.ndarray | None = None
        self.display_box = (0, 0, 1, 1)
        self.roi_start: tuple[int, int] | None = None
        self.defect_region: list[int] | None = None
        self.roi_rectangle: int | None = None
        self.folder_images: list[Path] = []
        self.image_index = -1
        self.source_type = "image"
        self.source_name = ""
        self.video_capture: cv2.VideoCapture | None = None
        self.video_cancel_event: threading.Event | None = None
        self.video_path: Path | None = None
        self.video_frame_index = 0
        self.video_frame_count = 0
        self.detection: dict[str, object] | None = None
        self.candidate: dict[str, object] | None = None
        self.busy = False
        self._feedback_submit_inflight = False
        self.closing = False
        self._ui_events: queue.Queue[
            tuple[Callable[..., None], tuple[object, ...]]
        ] = queue.Queue()
        self.last_benchmark_report: Path | None = None
        self.last_active_learning_manifest: Path | None = None
        self._records_hover_item = ""
        self._competition_flows: list[CompetitionFlowBar] = []
        self._competition_intros: list[
            tuple[ttk.Frame, ttk.Frame, ttk.Label, CompetitionHint]
        ] = []
        self._preview_resize_after_id: str | None = None
        self._last_preview_size: tuple[int, int] | None = None
        self._compact_viewport: bool | None = None

        self.current_category = self.categories[0]
        self.category_var = tk.StringVar(value=self.current_category)
        self.category_label_var = tk.StringVar(value="产品/任务")
        self.device_var = tk.StringVar(value="auto")
        self.device_display_var = tk.StringVar(value="自动")
        self.status_var = tk.StringVar(value="就绪")
        self.header_system_var = tk.StringVar(value="READY")
        self.result_var = tk.StringVar(value="WAITING")
        self.score_var = tk.StringVar(value="-")
        self.threshold_var = tk.StringVar(value="-")
        self.branch_var = tk.StringVar(value="-")
        self.margin_var = tk.StringVar(value="-")
        self.defect_types_result_var = tk.StringVar(value="-")
        self.feedback_status_var = tk.StringVar(value="当前使用模型原始阈值")
        self.feedback_submit_state_var = tk.StringVar(value="IDLE")
        self.feedback_submit_status_var = tk.StringVar(value="提交状态：等待提交")
        self.active_learning_status_var = tk.StringVar(value="尚未生成主动学习查询清单")
        self.image_position_var = tk.StringVar(value="未选择文件夹")
        self.viewer_file_var = tk.StringVar(value="未选择图片或视频")
        self.viewer_meta_var = tk.StringVar(value="-- × --  ·  等待输入")
        self.viewer_zoom_text_var = tk.StringVar(value="适应")
        self.viewer_zoom = 1.0
        self.viewer_fit_mode = True
        self.operator_label_var = tk.StringVar(value="正常")
        self.dataset_role_var = tk.StringVar(value="训练反馈")
        self.defect_type_var = tk.StringVar()
        self.show_heatmap_var = tk.BooleanVar(value=True)
        self.minimum_recall_var = tk.DoubleVar(value=0.90)
        self.base_metrics_var = tk.StringVar(value="尚未生成完整反馈候选")
        self.candidate_metrics_var = tk.StringVar(value="暂无数据")
        self.validation_metrics_var = tk.StringVar(value="尚未执行独立验证")
        self.sample_comparison_var = tk.StringVar(value="暂无数据")
        self.feature_bank_var = tk.StringVar(value="完整版本：未启用")
        self.revision_detail_var = tk.StringVar(
            value="输出层参数：原始模型包；反馈特征：0；反馈阈值：原始模型包"
        )
        self.task_name_var = tk.StringVar(value="private_task_01")
        self.annotation_mode_var = tk.StringVar(value="image_label")
        self.adapt_instruction_var = tk.StringVar(
            value="仅需选择100张正常和30张缺陷；目录名称就是图像级标签。测试目录必须与适配样本内容哈希不重叠。"
        )
        self.normal_dir_var = tk.StringVar(value="未选择")
        self.defect_dir_var = tk.StringVar(value="未选择")
        self.mask_dir_var = tk.StringVar(value="无需选择（仅分类标签模式）")
        self.test_manifest_var = tk.StringVar(value="可选：未选择")
        self.adapt_status_var = tk.StringVar(value="尚未开始适配")
        self.test_status_var = tk.StringVar(value="尚未执行独立验证")
        self.custom_accuracy_var = tk.StringVar(value="--")
        self.custom_recall_var = tk.StringVar(value="--")
        self.custom_f1_var = tk.StringVar(value="--")
        self.custom_fpr_var = tk.StringVar(value="--")
        self.custom_gate_var = tk.StringVar(value="WAITING")
        self.acceptance_recall_floor_var = tk.StringVar(
            value=f"{DEFAULT_ACCEPTANCE_RECALL_PERCENT:.1f}"
        )
        self.acceptance_stage1_fpr_var = tk.StringVar(
            value=f"{DEFAULT_ACCEPTANCE_STAGE1_FPR_PERCENT:.1f}"
        )
        self.acceptance_stage2_fpr_var = tk.StringVar(
            value=f"{DEFAULT_ACCEPTANCE_STAGE2_FPR_PERCENT:.1f}"
        )
        self.acceptance_threshold_summary_var = tk.StringVar(value="")
        self.adapt_stage_var = tk.StringVar(value="WAITING · 等待数据")
        self.adapt_count_var = tk.StringVar(value="0 / 130")
        self.adapt_elapsed_var = tk.StringVar(value="00:00")
        self.normal_sample_count_var = tk.StringVar(value="0 / 100")
        self.defect_sample_count_var = tk.StringVar(value="0 / 30")
        self.mask_sample_count_var = tk.StringVar(value="0 / 30")
        self.independent_test_count_var = tk.StringVar(value="0 / 1000+")
        self._small_manifest_approval: tuple[str, str, int] | None = None
        self._adapt_started_at: float | None = None
        self.benchmark_standard_var = tk.StringVar(value="auto")
        self.benchmark_allow_nonexclusive_var = tk.BooleanVar(value=True)
        self.fresh_attest_var = tk.BooleanVar(value=False)
        self.benchmark_detected_var = tk.StringVar(value="检测设备中...")
        self.benchmark_actual_device_var = tk.StringVar(value="检测中")
        self.benchmark_image_dir_var = tk.StringVar(value="未选择")
        self.benchmark_status_var = tk.StringVar(value="尚未执行性能测试")
        self.benchmark_details_expanded = True
        self.benchmark_hero_var = tk.StringVar(value="WAITING")
        self.benchmark_target_device_var = tk.StringVar(value="自动检测")
        self.benchmark_target_p50_var = tk.StringVar(value="自动选择")
        self.benchmark_p50_var = tk.StringVar(value="等待测试")
        self.benchmark_hardware_var = tk.StringVar(value="等待测试")
        self.benchmark_resolution_var = tk.StringVar(value="等待测试")
        self.benchmark_samples_var = tk.StringVar(value="等待测试")
        self.benchmark_validation_vars = {
            "hardware": tk.StringVar(value="WAITING"),
            "resolution": tk.StringVar(value="WAITING"),
            "latency": tk.StringVar(value="WAITING"),
            "protocol": tk.StringVar(value="WAITING"),
        }
        self.header_device_var = tk.StringVar(value="DEVICE DETECTING")
        self.header_model_var = tk.StringVar(
            value="MODEL LOADED"
            if self.current_category != EMPTY_PRODUCT_LABEL
            else "NO PRODUCT"
        )
        self.improvement_var = tk.StringVar(value="暂无数据")
        self.candidate_state_var = tk.StringVar(value="WAITING")
        self.active_revision_status_var = tk.StringVar(value="WAITING")
        self.record_search_var = tk.StringVar()
        self.record_role_filter_var = tk.StringVar(value="全部角色")
        self.record_source_filter_var = tk.StringVar(value="全部来源")
        self.record_type_filter_var = tk.StringVar(value="全部结果")
        self.record_model_filter_var = tk.StringVar(value="全部版本")
        self.records_count_var = tk.StringVar(value="0 条记录")

        self._configure_style()
        self._build_layout()
        self._detect_benchmark_device()
        self._refresh_all()
        if self.startup_pointer_warning:
            self.status_var.set("上次产品任务无法恢复，请重新加载产品模型")
            self.root.after(
                100,
                lambda: self._showwarning(
                    "产品任务恢复失败",
                    f"已安全进入空白生产模式。\n{self.startup_pointer_warning}",
                ),
            )
        self.root.after(40, self._poll_ui_events)
        self.root.bind("<Left>", lambda _event: self._keyboard_navigate(-1))
        self.root.bind("<Right>", lambda _event: self._keyboard_navigate(1))
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _showinfo(self, title: str, text: str, **options: object) -> str:
        options.setdefault("parent", self.root)
        return messagebox.showinfo(title, text, **options)

    def _showwarning(self, title: str, text: str, **options: object) -> str:
        options.setdefault("parent", self.root)
        return messagebox.showwarning(title, text, **options)

    def _showerror(self, title: str, text: str, **options: object) -> str:
        options.setdefault("parent", self.root)
        return messagebox.showerror(title, text, **options)

    def _askyesno(self, title: str, text: str, **options: object) -> bool:
        options.setdefault("parent", self.root)
        return bool(messagebox.askyesno(title, text, **options))

    def _askyesnocancel(
        self, title: str, text: str, **options: object
    ) -> bool | None:
        options.setdefault("parent", self.root)
        return messagebox.askyesnocancel(title, text, **options)

    def _post_ui(self, callback: Callable[..., None], *args: object) -> None:
        """Queue worker results so Tk widgets are only touched on the UI thread."""

        if not self.closing:
            self._ui_events.put((callback, args))

    def _poll_ui_events(self) -> None:
        if self.closing:
            return
        while True:
            try:
                callback, args = self._ui_events.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except Exception as error:
                self.busy = False
                self.status_var.set("操作失败")
                messagebox.showerror("操作失败", str(error), parent=self.root)
        if not self.closing:
            self.root.after(40, self._poll_ui_events)

    @staticmethod
    def _open_local_path(path: Path | None) -> None:
        if path is None or not path.exists():
            messagebox.showwarning("结果不存在", "尚未生成可打开的结果文件。")
            return
        os.startfile(str(path))

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        # clam 主题允许完整自定义背景/边框/配色，vista 只能改少量字段。
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.root.configure(bg=UI_BG)

        # 基础控件
        style.configure(
            ".",
            font=("Microsoft YaHei UI", 10),
            background=UI_BG,
            foreground=UI_TEXT,
        )
        style.configure("TFrame", background=UI_BG)
        style.configure("TLabel", background=UI_BG, foreground=UI_TEXT)
        style.configure(
            "TButton",
            background=UI_PANEL,
            foreground=UI_TEXT,
            bordercolor=UI_BORDER,
            lightcolor=UI_BORDER,
            darkcolor=UI_BORDER,
            focuscolor=UI_PRIMARY,
            padding=(13, 7),
            relief="flat",
            borderwidth=1,
        )
        style.map(
            "TButton",
            background=[
                ("disabled", DISABLED_BG),
                ("active", HOVER_BG),
                ("pressed", UI_PRIMARY_SOFT),
            ],
            foreground=[("disabled", UI_DISABLED)],
            bordercolor=[("focus", UI_PRIMARY), ("disabled", UI_BORDER)],
        )
        style.configure(
            "Accent.TButton",
            background=UI_PRIMARY,
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
            bordercolor=UI_PRIMARY,
            lightcolor=UI_PRIMARY,
            darkcolor=UI_PRIMARY,
            focuscolor=UI_PRIMARY,
            padding=(16, 8),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[
                ("disabled", DISABLED_BG),
                ("active", UI_PRIMARY_DARK),
                ("pressed", UI_PRIMARY_DARK),
            ],
            foreground=[("disabled", UI_DISABLED)],
            bordercolor=[("focus", UI_PRIMARY_DARK), ("disabled", UI_BORDER)],
        )
        style.configure(
            "Secondary.TButton",
            background=UI_PANEL,
            foreground=UI_TEXT,
            bordercolor=UI_BORDER,
            lightcolor=UI_BORDER,
            darkcolor=UI_BORDER,
            focuscolor=UI_PRIMARY,
            padding=(13, 7),
            relief="flat",
            borderwidth=1,
        )
        style.map(
            "Secondary.TButton",
            background=[
                ("disabled", DISABLED_BG),
                ("active", HOVER_BG),
                ("pressed", UI_PRIMARY_SOFT),
            ],
            foreground=[("disabled", UI_DISABLED)],
            bordercolor=[("focus", UI_PRIMARY), ("active", UI_BORDER)],
        )
        style.configure(
            "Ghost.TButton",
            background=UI_PANEL,
            foreground=UI_MUTED,
            bordercolor=UI_PANEL,
            lightcolor=UI_PANEL,
            darkcolor=UI_PANEL,
            padding=(10, 7),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Ghost.TButton",
            background=[("active", UI_PANEL_ALT)],
            foreground=[("active", UI_PRIMARY)],
        )
        style.configure(
            "Danger.TButton",
            background=UI_DANGER_SOFT,
            foreground=UI_DANGER,
            bordercolor=UI_DANGER,
            lightcolor=UI_DANGER,
            darkcolor=UI_DANGER,
            padding=(13, 7),
            relief="flat",
            borderwidth=1,
        )
        style.map(
            "Danger.TButton",
            background=[
                ("disabled", DISABLED_BG),
                ("active", "#FBE2E2"),
                ("pressed", "#F8D5D5"),
            ],
            foreground=[("disabled", UI_DISABLED)],
            bordercolor=[("focus", UI_DANGER), ("disabled", UI_BORDER)],
        )
        style.configure(
            "HeaderNav.TButton",
            background=UI_PANEL,
            foreground=UI_TEXT,
            bordercolor=UI_BORDER,
            lightcolor=UI_BORDER,
            darkcolor=UI_BORDER,
            focuscolor=UI_PRIMARY,
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(12, 6),
            relief="flat",
            borderwidth=1,
        )
        style.map(
            "HeaderNav.TButton",
            background=[("active", HOVER_BG), ("pressed", UI_PRIMARY_SOFT)],
            foreground=[("active", UI_PRIMARY_DARK)],
            bordercolor=[("focus", UI_PRIMARY), ("active", UI_BORDER)],
        )
        style.configure(
            "HeaderNavActive.TButton",
            background=UI_PRIMARY_SOFT,
            foreground=UI_PRIMARY_DARK,
            bordercolor=UI_PRIMARY,
            lightcolor=UI_PRIMARY,
            darkcolor=UI_PRIMARY,
            focuscolor=UI_PRIMARY,
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(12, 6),
            relief="flat",
            borderwidth=1,
        )
        style.map(
            "HeaderNavActive.TButton",
            background=[("active", HOVER_BG), ("pressed", UI_PRIMARY_SOFT)],
            foreground=[("active", UI_PRIMARY_DARK)],
            bordercolor=[("focus", UI_PRIMARY_DARK), ("active", UI_PRIMARY)],
        )

        # 输入类控件
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground=UI_TEXT,
            bordercolor=INPUT_BORDER,
            lightcolor=INPUT_BORDER,
            darkcolor=INPUT_BORDER,
            padding=(8, 7),
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", DISABLED_BG), ("readonly", UI_PANEL_ALT)],
            foreground=[("disabled", UI_DISABLED)],
            bordercolor=[("focus", UI_PRIMARY), ("disabled", UI_BORDER)],
        )
        style.configure(
            "TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground=UI_TEXT,
            bordercolor=INPUT_BORDER,
            arrowcolor=UI_TEXT,
            padding=(8, 7),
        )
        style.map(
            "TCombobox",
            bordercolor=[("focus", UI_PRIMARY), ("disabled", UI_BORDER)],
            fieldbackground=[("readonly", "#ffffff")],
            foreground=[("disabled", UI_DISABLED)],
        )
        style.configure(
            "TSpinbox",
            fieldbackground="#ffffff",
            foreground=UI_TEXT,
            bordercolor=INPUT_BORDER,
            arrowcolor=UI_TEXT,
            padding=4,
        )
        style.map(
            "TSpinbox",
            bordercolor=[("focus", UI_PRIMARY), ("disabled", UI_BORDER)],
            fieldbackground=[("disabled", DISABLED_BG)],
            foreground=[("disabled", UI_DISABLED)],
        )

        # 复选 / 单选
        style.configure(
            "TCheckbutton",
            background=UI_BG,
            foreground=UI_TEXT,
            indicatorcolor=UI_PANEL,
            bordercolor=UI_BORDER,
        )
        style.map(
            "TCheckbutton",
            indicatorcolor=[("selected", UI_PRIMARY)],
            background=[("active", UI_BG)],
        )
        style.configure(
            "TRadiobutton",
            background=UI_BG,
            foreground=UI_TEXT,
            indicatorcolor=UI_PANEL,
            bordercolor=UI_BORDER,
        )
        style.map(
            "TRadiobutton",
            indicatorcolor=[("selected", UI_PRIMARY)],
            background=[("active", UI_BG)],
        )

        # 标签样式
        style.configure(
            "Title.TLabel",
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "PageTitle.TLabel",
            background=UI_BG,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "PageSubtitle.TLabel",
            background=UI_BG,
            foreground=UI_MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Section.TLabel",
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 13, "bold"),
        )
        style.configure(
            "Result.TLabel",
            background=UI_PANEL,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure("Card.TLabel", background=UI_PANEL, foreground=UI_TEXT)
        style.configure(
            "CardSection.TLabel",
            background=UI_PANEL,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure("CardMuted.TLabel", background=UI_PANEL, foreground=UI_MUTED)
        style.configure(
            "Metric.TLabel",
            background=UI_PANEL,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "MetricSmall.TLabel",
            background=UI_PANEL,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        style.configure(
            "Success.TLabel", background=UI_PANEL, foreground=UI_SUCCESS,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Danger.TLabel", background=UI_PANEL, foreground=UI_DANGER,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure("Muted.TLabel", foreground=UI_MUTED)
        style.configure("Status.TLabel", background=UI_BG, foreground=UI_MUTED)
        style.configure(
            "Toolbar.TLabel",
            background=UI_PANEL,
            foreground=UI_MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure("ToolbarStatus.TLabel", background=UI_PANEL, foreground=UI_SUCCESS)

        # 卡片容器
        style.configure(
            "Card.TFrame",
            background=UI_PANEL,
            relief="solid",
            bordercolor=UI_BORDER,
            lightcolor=UI_BORDER,
            darkcolor=UI_BORDER,
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe",
            background=UI_PANEL,
            bordercolor=UI_BORDER,
            lightcolor=UI_BORDER,
            darkcolor=UI_BORDER,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=UI_PANEL,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
            padding=(6, 0),
        )
        style.configure(
            "Toolbar.TFrame",
            background=UI_PANEL,
            relief="solid",
            bordercolor=UI_BORDER,
            lightcolor=UI_BORDER,
            darkcolor=UI_BORDER,
            borderwidth=1,
        )
        style.configure("ValueRow.TFrame", background=UI_PANEL, relief="flat", borderwidth=0)

        # Notebook
        style.configure(
            "TNotebook",
            background=UI_BG,
            bordercolor=UI_BG,
            lightcolor=UI_BG,
            darkcolor=UI_BG,
            borderwidth=0,
            tabmargins=(0, 5, 0, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=UI_BG,
            foreground=UI_MUTED,
            bordercolor=UI_BG,
            lightcolor=UI_BG,
            darkcolor=UI_BG,
            padding=(18, 10),
            font=("Microsoft YaHei UI", 10),
            borderwidth=1,
            relief="flat",
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", UI_PRIMARY_SOFT), ("active", HOVER_BG)],
            foreground=[("selected", UI_PRIMARY), ("active", UI_TEXT)],
            bordercolor=[("selected", UI_PRIMARY), ("active", UI_BORDER)],
            lightcolor=[("selected", UI_PRIMARY)],
            darkcolor=[("selected", UI_PRIMARY)],
            font=[("selected", ("Microsoft YaHei UI", 10, "bold"))],
        )
        style.configure(
            "Content.TNotebook",
            background=UI_BG,
            bordercolor=UI_BG,
            lightcolor=UI_BG,
            darkcolor=UI_BG,
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.layout("Content.TNotebook.Tab", [])

        # 表格
        style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground=UI_TEXT,
            bordercolor=UI_BORDER,
            lightcolor=UI_BORDER,
            darkcolor=UI_BORDER,
            rowheight=31,
        )
        style.map(
            "Treeview",
            background=[("selected", UI_PRIMARY_SOFT)],
            foreground=[("selected", UI_TEXT)],
        )
        style.configure(
            "Treeview.Heading",
            background=UI_HEADING_BG,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
            relief="flat",
            borderwidth=0,
            padding=(6, 7),
        )
        style.map("Treeview.Heading", background=[("active", "#eef3f8")])

        # 滚动条 / 分隔线
        style.configure(
            "Vertical.TScrollbar",
            background="#b7c4ce",
            troughcolor=UI_BG,
            bordercolor=UI_BG,
            arrowcolor=UI_MUTED,
            relief="flat",
        )
        style.map("Vertical.TScrollbar", background=[("active", UI_PRIMARY)])
        style.configure(
            "Horizontal.TScrollbar",
            background="#b7c4ce",
            troughcolor=UI_BG,
            bordercolor=UI_BG,
            arrowcolor=UI_MUTED,
            relief="flat",
        )
        style.map("Horizontal.TScrollbar", background=[("active", UI_PRIMARY)])
        style.configure("TSeparator", background=UI_BORDER)

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, padding=(20, 12, 20, 12))
        self.shell = shell
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg=UI_PANEL, highlightthickness=1, highlightbackground=UI_BORDER)
        self.header = header
        header.pack(fill="x", pady=(0, 10))
        header.configure(height=68)
        header.pack_propagate(False)
        brand = tk.Frame(header, bg=UI_PANEL)
        self.brand = brand
        brand.pack(side="left", padx=20, pady=10)
        tk.Label(
            brand, text="AOI 质量检测工作台", bg=UI_PANEL, fg=UI_TEXT,
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand, text="检测、适配、反馈与竞赛验收一体化工作台", bg=UI_PANEL,
            fg=UI_MUTED, font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        badges = tk.Frame(header, bg=UI_PANEL)
        self.badges = badges
        badges.pack(side="right", padx=20, pady=15)

        self._status_badge(badges, self.header_system_var, prefix="系统").pack(
            side="left", padx=(8, 0)
        )
        self._status_badge(badges, self.header_model_var, prefix="模型").pack(
            side="left", padx=(8, 0)
        )
        self._status_badge(badges, self.header_device_var, prefix="设备").pack(
            side="left", padx=(8, 0)
        )
        self.records_navigation_button = ttk.Button(
            badges,
            text="反馈记录",
            style="HeaderNav.TButton",
            command=lambda: self._switch_tab(self.records_tab),
        )
        self.records_navigation_button.pack(side="left", padx=(12, 0))

        controls = ttk.Frame(shell, style="Toolbar.TFrame", padding=(16, 8))
        self.controls = controls
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(
            controls, textvariable=self.category_label_var, style="Toolbar.TLabel"
        ).pack(side="left")
        category = ttk.Combobox(
            controls, textvariable=self.category_var, values=self.categories,
            state="readonly", width=15,
        )
        self.category_combo = category
        category.pack(side="left", padx=(8, 20))
        category.bind("<<ComboboxSelected>>", lambda _event: self._category_changed())
        ttk.Label(controls, text="模型", style="Toolbar.TLabel").pack(side="left")
        model_pill = tk.Frame(controls, bg=UI_PRIMARY_SOFT, padx=9, pady=6)
        model_pill.pack(side="left", padx=(8, 20))
        self.feedback_status_badge = tk.Label(
            model_pill, textvariable=self.feedback_status_var, bg=UI_PRIMARY_SOFT,
            fg=UI_PRIMARY_DARK, font=("Microsoft YaHei UI", 9, "bold"),
        )
        self.feedback_status_badge.pack()
        ttk.Label(controls, text="设备", style="Toolbar.TLabel").pack(side="left")
        device_combo = ttk.Combobox(
            controls, textvariable=self.device_display_var, values=("自动", "cuda:0", "cpu"),
            state="readonly", width=10,
        )
        device_combo.pack(side="left", padx=(8, 20))
        device_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.device_var.set(
                "auto" if self.device_display_var.get() == "自动" else self.device_display_var.get()
            ),
        )

        ttk.Button(
            controls, text="加载产品任务", style="Secondary.TButton",
            command=self._choose_model_pointer,
        ).pack(side="right", padx=(8, 0))
        ttk.Button(
            controls, text="+ 新建产品", style="Accent.TButton",
            command=self._create_category,
        ).pack(side="right", padx=(8, 0))
        more = tk.Menubutton(
            controls, text="•••", bg=UI_PANEL, fg=UI_MUTED, relief="flat",
            activebackground=UI_PANEL_ALT, activeforeground=UI_TEXT, cursor="hand2",
            font=("Segoe UI", 12, "bold"), padx=10, pady=5,
        )
        more.pack(side="right")
        more_menu = tk.Menu(more, tearoff=False, bg=UI_PANEL, fg=UI_TEXT,
                            activebackground=UI_DANGER_SOFT, activeforeground=UI_DANGER)
        more_menu.add_command(
            label="仅移除使用（保留文件）",
            command=self._delete_category,
        )
        more_menu.add_separator()
        more_menu.add_command(
            label="移除磁盘文件和审计记录…",
            command=self._delete_category_files_and_audit,
        )
        self.product_actions_menu = more_menu
        more.configure(menu=more_menu)

        # The AOI flow bar is the primary navigation.  Keep the Notebook as a
        # content controller while suppressing its duplicate tab strip.
        self.notebook = ttk.Notebook(shell, style="Content.TNotebook")
        self.notebook.pack(fill="both", expand=True)
        page_padding = (0, 0, 0, 0)
        self.detect_tab = ttk.Frame(self.notebook, padding=page_padding)
        self.optimize_tab = ttk.Frame(self.notebook, padding=page_padding)
        self.acceptance_tab = ttk.Frame(self.notebook, padding=page_padding)
        self.records_tab = ttk.Frame(self.notebook, padding=page_padding)
        self.adapt_tab = ttk.Frame(self.notebook, padding=page_padding)
        self.notebook.add(self.detect_tab, text="检测工作台")
        self.notebook.add(self.adapt_tab, text="快速适配")
        self.notebook.add(self.optimize_tab, text="反馈优化")
        self.notebook.add(self.acceptance_tab, text="性能测试")
        self.notebook.add(self.records_tab, text="反馈记录")
        self.notebook.bind("<<NotebookTabChanged>>", self._notebook_tab_changed)
        self.detect_page = AdaptivePage(self.detect_tab)
        self.adapt_page = AdaptivePage(self.adapt_tab)
        self.optimize_page = AdaptivePage(self.optimize_tab)
        self.acceptance_page = AdaptivePage(self.acceptance_tab)
        self.records_page = AdaptivePage(self.records_tab)
        for page in (
            self.detect_page,
            self.adapt_page,
            self.optimize_page,
            self.acceptance_page,
            self.records_page,
        ):
            page.pack(fill="both", expand=True)
        self._build_detect_tab()
        self._build_adapt_tab()
        self._build_optimize_tab()
        self._build_acceptance_tab()
        self._build_records_tab()
        # Start at the first step of the AOI workflow instead of the online
        # detection page, which requires an adapted model and input image.
        self.notebook.select(self.adapt_tab)
        self.status_var.trace_add("write", self._sync_system_badge)
        self.root.bind("<Configure>", self._viewport_configured, add="+")

    def _card(self, parent: ttk.Frame, title: str | None = None) -> ttk.Frame:
        """返回白色卡片容器（内部统一用 pack 布局）。

        调用方必须只用 pack 向父级放置卡片；卡片内部也只允许 pack 子控件，
        避免 tkinter 的 pack/grid 混用冲突。
        """
        card = ttk.Frame(parent, style="Card.TFrame", padding=(18, 16))
        if title is not None:
            ttk.Label(card, text=title, style="CardSection.TLabel").pack(
                anchor="w", pady=(0, 10)
            )
        return card

    def _status_badge(
        self,
        parent: tk.Misc,
        source: tk.Variable,
        *,
        prefix: str | None = None,
        context: str = "general",
        waiting_text: str | None = None,
        failure_text: str | None = None,
    ) -> tk.Label:
        """Build a reusable, live status badge without mutating its source value."""

        label = tk.Label(
            parent,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=10,
            pady=5,
            relief="flat",
            borderwidth=0,
        )

        tone_colors = {
            "waiting": (WAITING_BG, WAITING_TEXT),
            "running": (UI_PRIMARY_SOFT, UI_PRIMARY_DARK),
            "success": (UI_SUCCESS_SOFT, UI_SUCCESS),
            "warning": (UI_WARNING_SOFT, UI_WARNING),
            "error": (UI_DANGER_SOFT, UI_DANGER),
            "neutral": (UI_PANEL_ALT, UI_MUTED),
        }

        def sync(*_args: object) -> None:
            if not label.winfo_exists():
                return
            raw = source.get()
            shown = display_status(raw, context=context)
            head = str(raw).strip().upper().split(" · ", maxsplit=1)[0]
            if waiting_text is not None and head == "WAITING":
                shown = waiting_text
            if failure_text is not None and head in {"FAIL", "FAILED", "ERROR"}:
                shown = failure_text
            tone = ui_status_tone(raw, context=context)
            if prefix == "设备" and tone == "neutral" and str(raw).strip():
                tone = "success"
            background, foreground = tone_colors[tone]
            text = f"{prefix} · {shown}" if prefix else shown
            label.configure(text=text, bg=background, fg=foreground)

        source.trace_add("write", sync)
        sync()
        return label

    def _sync_system_badge(self, *_args: object) -> None:
        status = self.status_var.get()
        if any(token in status for token in ("失败", "未通过", "禁止")):
            self.header_system_var.set("ATTENTION")
        elif any(token in status for token in ("正在", "执行", "生成")):
            self.header_system_var.set("RUNNING")
        else:
            self.header_system_var.set("READY")

    def _page_header(
        self, parent: ttk.Frame, title: str, subtitle: str
    ) -> tuple[ttk.Frame, ttk.Label, CompetitionHint]:
        header = ttk.Frame(parent)
        title_label = ttk.Label(header, text=title, style="PageTitle.TLabel")
        title_label.pack(anchor="w")
        hint_label = CompetitionHint(header, subtitle)
        hint_label.pack(fill="x", pady=(3, 0))
        return header, title_label, hint_label

    def _competition_intro(
        self,
        parent: ttk.Frame,
        *,
        active_stage: str | None,
        title: str,
        hint: str,
    ) -> ttk.Frame:
        """Build the shared competition flow and one-line page positioning."""

        intro = ttk.Frame(parent)
        flow = CompetitionFlowBar(
            intro,
            active_stage=active_stage,
            on_navigate=self._navigate_competition_stage,
        )
        flow.pack(fill="x")
        self._competition_flows.append(flow)
        page_header, title_label, hint_label = self._page_header(intro, title, hint)
        page_header.pack(fill="x", pady=(8, 0))
        self._competition_intros.append(
            (intro, page_header, title_label, hint_label)
        )
        return intro

    def _navigate_competition_stage(self, stage: str) -> None:
        """Use the existing tab switch helper as the flow bar navigation target."""

        targets = {
            "adapt": self.adapt_tab,
            "detect": self.detect_tab,
            "feedback": self.detect_tab,
            "optimize": self.optimize_tab,
            "acceptance": self.acceptance_tab,
        }
        target = targets.get(stage)
        if target is not None:
            self._switch_tab(target)

    def _viewport_configured(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        compact = event.height < 820
        if compact == self._compact_viewport:
            return
        self._compact_viewport = compact
        self._apply_viewport_density(compact)

    def _apply_viewport_density(self, compact: bool) -> None:
        """Give short windows more workspace by changing gaps, never scaling UI."""

        self.shell.configure(
            padding=(14, 7, 14, 7) if compact else (20, 12, 20, 12)
        )
        self.header.configure(height=58 if compact else 68)
        self.header.pack_configure(pady=(0, 6 if compact else 10))
        self.brand.pack_configure(
            padx=16 if compact else 20,
            pady=6 if compact else 10,
        )
        self.badges.pack_configure(
            padx=16 if compact else 20,
            pady=10 if compact else 15,
        )
        self.controls.configure(padding=(14, 5) if compact else (16, 8))
        self.controls.pack_configure(pady=(0, 6 if compact else 10))
        tab_padding = (0, 0, 0, 0)
        for tab in (
            self.detect_tab,
            self.adapt_tab,
            self.optimize_tab,
            self.acceptance_tab,
            self.records_tab,
        ):
            tab.configure(padding=tab_padding)
        for flow in self._competition_flows:
            flow.set_compact_height(compact)
        for intro, page_header, title_label, hint_label in self._competition_intros:
            intro.grid_configure(pady=(0, 8 if compact else 16))
            page_header.pack_configure(pady=(5 if compact else 8, 0))
            title_label.pack_forget()
            hint_label.pack_forget()
            if compact:
                title_label.pack(side="left", anchor="w")
                hint_label.pack(
                    side="left", fill="x", expand=True, padx=(14, 0)
                )
            else:
                title_label.pack(anchor="w")
                hint_label.pack(fill="x", pady=(3, 0))

    def _metric_card(
        self,
        parent: tk.Misc,
        label: str,
        variable: tk.StringVar,
        *,
        status: tk.StringVar | None = None,
        accent: str = UI_PRIMARY,
        compact: bool = False,
    ) -> tk.Frame:
        pad_x = 10 if compact else 14
        pad_y = 8 if compact else 12
        card = tk.Frame(
            parent, bg=UI_PANEL, highlightthickness=1, highlightbackground=UI_BORDER,
            padx=pad_x, pady=pad_y,
        )
        top = tk.Frame(card, bg=UI_PANEL)
        top.pack(fill="x")
        tk.Label(
            top, text=label, bg=UI_PANEL, fg=UI_MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(side="left")
        if status is not None:
            self._status_badge(top, status, context="quality").pack(side="right")
        tk.Label(
            card, textvariable=variable, bg=UI_PANEL, fg=UI_TEXT,
            font=("Microsoft YaHei UI", 15 if compact else 20, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(5 if compact else 8, 0))
        return card

    def _value_row(
        self,
        parent: tk.Misc,
        label: str,
        variable: tk.StringVar,
    ) -> ttk.Frame:
        """Render one compact label/value row shared by detection panels."""

        row = ttk.Frame(parent, style="ValueRow.TFrame")
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text=label, style="CardMuted.TLabel").pack(side="left")
        ttk.Label(row, textvariable=variable, style="CardValue.TLabel").pack(
            side="right"
        )
        return row

    def _segment_option(
        self,
        parent: tk.Misc,
        text: str,
        variable: tk.StringVar,
        value: str,
        command: Callable[[], None] | None = None,
    ) -> tk.Radiobutton:
        return tk.Radiobutton(
            parent, text=text, variable=variable, value=value, command=command,
            indicatoron=False, relief="flat", bd=0, padx=14, pady=8,
            bg=UI_PANEL_ALT, fg=UI_MUTED, selectcolor=UI_PRIMARY_SOFT,
            activebackground=UI_PRIMARY_SOFT, activeforeground=UI_PRIMARY_DARK,
            font=("Microsoft YaHei UI", 9, "bold"), cursor="hand2",
        )

    def _build_detect_tab(self) -> None:
        page = self.detect_page.body
        page.columnconfigure(0, weight=64)
        page.columnconfigure(1, weight=36)
        page.rowconfigure(1, weight=1)
        self._competition_intro(
            page,
            active_stage="detect",
            title="检测工作台",
            hint="赛题对应：图片/视频在线异常检测与操作员复核",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        preview_card = ttk.Frame(page, style="Card.TFrame", padding=(16, 14))
        preview_card.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        viewer_header = ttk.Frame(preview_card, style="ValueRow.TFrame")
        viewer_header.pack(fill="x", pady=(0, 10))
        ttk.Label(viewer_header, text="图像 / 视频预览", style="CardSection.TLabel").pack(side="left")
        ttk.Button(
            viewer_header, text="选择图片", style="Secondary.TButton",
            command=self._choose_image,
        ).pack(side="left", padx=(12, 0))
        ttk.Button(
            viewer_header, text="选择视频", style="Secondary.TButton",
            command=self._choose_video,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            viewer_header, text="全屏", style="Ghost.TButton", command=self._toggle_fullscreen,
        ).pack(side="right")
        ttk.Checkbutton(
            viewer_header, text="热力图", variable=self.show_heatmap_var,
            command=self._render_preview,
        ).pack(side="right", padx=(0, 6))
        ttk.Button(
            viewer_header, text="适应", style="Ghost.TButton", command=self._fit_viewer,
        ).pack(side="right")
        ttk.Button(
            viewer_header, text="+", style="Ghost.TButton", command=lambda: self._zoom_viewer(0.15),
        ).pack(side="right")
        ttk.Label(
            viewer_header, textvariable=self.viewer_zoom_text_var, style="CardMuted.TLabel",
        ).pack(side="right", padx=6)
        ttk.Button(
            viewer_header, text="−", style="Ghost.TButton", command=lambda: self._zoom_viewer(-0.15),
        ).pack(side="right")
        self.image_canvas = tk.Canvas(
            preview_card, background=UI_CANVAS, highlightthickness=0, cursor="crosshair"
        )
        self.image_canvas.pack(fill="both", expand=True)
        self._draw_viewer_empty_state()
        self.image_canvas.bind("<Configure>", self._preview_canvas_changed)
        self.image_canvas.bind("<ButtonPress-1>", self._roi_press)
        self.image_canvas.bind("<B1-Motion>", self._roi_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self._roi_release)
        if (
            DND_FILES is not None
            and hasattr(self.root, "TkdndVersion")
            and hasattr(self.image_canvas, "drop_target_register")
        ):
            self.image_canvas.drop_target_register(DND_FILES)
            self.image_canvas.dnd_bind("<<Drop>>", self._drop_media)
        viewer_footer = tk.Frame(preview_card, bg=UI_CANVAS, padx=12, pady=7)
        viewer_footer.pack(fill="x")
        tk.Label(
            viewer_footer, textvariable=self.viewer_file_var, bg=UI_CANVAS,
            fg="#d6dee8", font=("Microsoft YaHei UI", 9),
        ).pack(side="left")
        tk.Label(
            viewer_footer, textvariable=self.viewer_meta_var, bg=UI_CANVAS,
            fg=UI_CANVAS_TEXT, font=("Segoe UI", 9),
        ).pack(side="right")
        buttons = ttk.Frame(preview_card, style="ValueRow.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        self.previous_button = ttk.Button(buttons, text="上一张", style="Ghost.TButton", command=lambda: self._navigate_image(-1))
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(buttons, text="下一张", style="Ghost.TButton", command=lambda: self._navigate_image(1))
        self.next_button.pack(side="left")
        self._update_navigation_buttons()

        right_viewport = self.detect_right_viewport = VerticalScrollPanel(page)
        right_viewport.grid(row=1, column=1, sticky="nsew")
        right = right_viewport.body
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        result_card = ttk.Frame(right, style="Card.TFrame", padding=(14, 10))
        result_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        hero = tk.Frame(result_card, bg=UI_PANEL_ALT, padx=12, pady=8)
        hero.pack(fill="x")
        self.detect_button = ttk.Button(
            hero, text="开始检测", style="Accent.TButton",
            command=self._start_detection,
        )
        self.detect_button.pack(side="right", padx=(12, 0))
        hero_status = tk.Frame(hero, bg=UI_PANEL_ALT)
        hero_status.pack(side="left", fill="x", expand=True)
        tk.Label(
            hero_status, text=display_text("DETECTION STATUS"), bg=UI_PANEL_ALT, fg=UI_MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w")
        self.result_label = self._status_badge(
            hero_status,
            self.result_var,
            context="quality",
            waiting_text="等待检测",
            failure_text="检测失败",
        )
        self.result_label.configure(font=("Microsoft YaHei UI", 20, "bold"), padx=12, pady=6)
        self.result_label.pack(anchor="w", pady=(3, 0))
        metrics = ttk.Frame(result_card, style="ValueRow.TFrame")
        metrics.pack(fill="x", pady=(8, 0))
        metrics.columnconfigure((0, 1, 2), weight=1, uniform="detect_metric")
        self._metric_card(metrics, "异常分数", self.score_var, compact=True).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._metric_card(metrics, "决策阈值", self.threshold_var, compact=True).grid(row=0, column=1, sticky="ew", padx=6)
        self._metric_card(metrics, "判定边际", self.margin_var, compact=True).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        self._value_row(result_card, "缺陷类型", self.defect_types_result_var)

        panel_card = ttk.Frame(right, style="Card.TFrame", padding=(14, 10))
        panel_card.grid(row=1, column=0, sticky="nsew")
        feedback_header = ttk.Frame(panel_card, style="ValueRow.TFrame")
        feedback_header.pack(fill="x")
        ttk.Label(feedback_header, text="操作员反馈", style="CardSection.TLabel").pack(side="left")
        self.feedback_submit_button = ttk.Button(
            feedback_header, text="提交反馈", style="Accent.TButton",
            command=self._submit_feedback,
        )
        self.feedback_submit_button.pack(side="right")
        ttk.Label(
            panel_card, text="复核结果仅进入当前反馈链路，不改变冻结评估集。",
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(2, 10))
        ttk.Label(
            panel_card,
            textvariable=self.feedback_submit_status_var,
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        fields = ttk.Frame(panel_card, style="ValueRow.TFrame")
        fields.pack(fill="x")
        fields.columnconfigure((0, 1), weight=1, uniform="feedback_fields")
        ttk.Label(fields, text="实际状态", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(fields, text="反馈用途", style="CardMuted.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Combobox(fields, textvariable=self.operator_label_var, values=("正常", "异常"), state="readonly").grid(row=1, column=0, sticky="ew", pady=(3, 8))
        ttk.Combobox(
            fields, textvariable=self.dataset_role_var,
            values=("训练反馈", "独立验证"), state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(3, 8))
        ttk.Label(fields, text="缺陷类型", style="CardMuted.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        ttk.Entry(fields, textvariable=self.defect_type_var).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(3, 8)
        )
        ttk.Label(panel_card, text="备注", style="CardMuted.TLabel").pack(anchor="w")
        self.note_text = tk.Text(
            panel_card, height=2, wrap="word", relief="solid", borderwidth=0,
            bg=UI_PANEL_ALT, fg=UI_TEXT, highlightthickness=1,
            highlightbackground=UI_BORDER, highlightcolor=UI_PRIMARY,
        )
        self.note_text.pack(fill="x", pady=(3, 0))
        right_viewport.bind_mousewheel_tree()

    def _build_optimize_tab(self) -> None:
        page = self.optimize_page.body
        page.columnconfigure(0, weight=1)
        page.rowconfigure(4, weight=1)
        self._competition_intro(
            page,
            active_stage="optimize",
            title="反馈优化",
            hint="赛题对应：操作员反馈驱动的候选模型生成、验证与回滚",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 16))
        self.total_var = tk.StringVar(value="0")
        self.normal_var = tk.StringVar(value="0")
        self.defect_var = tk.StringVar(value="0")
        self.validation_count_var = tk.StringVar(value="0")
        self.feedback_loop = CompactProcessStrip(
            page,
            ("操作员反馈", "候选模型", "独立验证", "审核启用", "可回滚"),
        )
        self.feedback_loop.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        self.candidate_state_var.trace_add("write", self._sync_feedback_loop)
        self.total_var.trace_add("write", self._sync_feedback_loop)
        self._sync_feedback_loop()

        stats = ttk.Frame(page)
        stats.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        stat_defs = (
            ("反馈总数", self.total_var, UI_PRIMARY),
            ("优化训练正常", self.normal_var, UI_SUCCESS),
            ("优化训练异常", self.defect_var, UI_DANGER),
            ("独立验证", self.validation_count_var, UI_ACCENT_DARK),
        )
        for column, (title, variable, color) in enumerate(stat_defs):
            stats.columnconfigure(column, weight=1, uniform="feedback_stats")
            self._metric_card(stats, title, variable, accent=color).grid(
                row=0, column=column, sticky="ew",
                padx=(0 if column == 0 else 6, 0 if column == 3 else 6),
            )

        control_row = ttk.Frame(page)
        control_row.grid(row=3, column=0, sticky="ew", pady=(0, 16))
        control_row.columnconfigure((0, 1), weight=1, uniform="optimize_control")
        constraint = ttk.Frame(control_row, style="Card.TFrame", padding=(18, 16))
        constraint.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(constraint, text="候选版本要求", style="CardSection.TLabel").pack(anchor="w")
        ttk.Label(
            constraint,
            text=(
                "训练至少 6 条，独立验证至少 4 条，且都包含正常与异常。"
                "达到目标门槛可通过；未达目标时，独立验证召回率/F1均不下降且至少一项提升也算改善成功。"
            ),
            style="CardMuted.TLabel", wraplength=500, justify="left",
        ).pack(anchor="w", pady=(3, 12))
        recall_row = ttk.Frame(constraint, style="ValueRow.TFrame")
        recall_row.pack(fill="x")
        ttk.Label(recall_row, text="目标召回率", style="Card.TLabel").pack(side="left")
        ttk.Spinbox(
            recall_row, from_=0.50, to=1.0, increment=0.01,
            textvariable=self.minimum_recall_var, width=8,
        ).pack(side="left", padx=10)
        ttk.Button(
            recall_row,
            text="生成完整反馈候选",
            style="Accent.TButton",
            command=self._propose_revision,
        ).pack(side="right")

        version = ttk.Frame(control_row, style="Card.TFrame", padding=(18, 16))
        version.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        state_row = ttk.Frame(version, style="ValueRow.TFrame")
        state_row.pack(fill="x")
        ttk.Label(state_row, text="版本状态", style="CardSection.TLabel").pack(side="left")
        self._status_badge(
            state_row,
            self.active_revision_status_var,
            waiting_text="未启用",
            failure_text="校验失败",
        ).pack(side="right")
        ttk.Label(version, textvariable=self.feature_bank_var, style="Card.TLabel", wraplength=500, justify="left").pack(anchor="w", pady=(12, 4))
        ttk.Label(
            version, textvariable=self.revision_detail_var, style="CardMuted.TLabel",
            wraplength=500, justify="left",
        ).pack(anchor="w")

        self.optimize_comparison_viewport = VerticalScrollPanel(page)
        self.optimize_comparison_viewport.grid(row=4, column=0, sticky="nsew")
        comparison_host = self.optimize_comparison_viewport.body
        comparison_host.columnconfigure(0, weight=1)
        comparison_host.rowconfigure(0, weight=1)
        comparison = ttk.Frame(comparison_host)
        comparison.grid(row=0, column=0, sticky="nsew")
        comparison.columnconfigure((0, 1), weight=1, uniform="model_comparison")
        current = ttk.Frame(comparison, style="Card.TFrame", padding=(18, 16))
        current.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(
            current,
            text=display_text("CURRENT"),
            bg=UI_PRIMARY_SOFT,
            fg=UI_PRIMARY_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=9,
            pady=4,
        ).pack(anchor="w")
        ttk.Label(
            current,
            text="训练反馈拟合指标（仅诊断）",
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(10, 0))
        ttk.Label(current, textvariable=self.base_metrics_var, style="Card.TLabel", wraplength=500, justify="left").pack(anchor="w", pady=(4, 8))
        ttk.Label(current, text="当前样本", style="CardMuted.TLabel").pack(anchor="w")
        ttk.Label(current, textvariable=self.sample_comparison_var, style="Card.TLabel", wraplength=500, justify="left").pack(anchor="w", pady=(4, 0))

        candidate = tk.Frame(
            comparison,
            bg=UI_PRIMARY_SOFT,
            highlightthickness=1,
            highlightbackground=UI_PRIMARY,
            padx=18,
            pady=16,
        )
        candidate.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        candidate_top = tk.Frame(candidate, bg=UI_PRIMARY_SOFT)
        candidate_top.pack(fill="x")
        tk.Label(
            candidate_top,
            text=display_text("CANDIDATE"),
            bg=UI_PRIMARY,
            fg="#FFFFFF",
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=9,
            pady=4,
        ).pack(side="left")
        self._status_badge(
            candidate_top,
            self.candidate_state_var,
            waiting_text="等待候选版本",
            failure_text="候选/验证未通过",
        ).pack(side="right")
        tk.Label(
            candidate, textvariable=self.improvement_var,
            bg=UI_PRIMARY_SOFT, fg=UI_PRIMARY_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", pady=(10, 0))
        tk.Label(
            candidate, text="训练反馈拟合指标（仅诊断）",
            bg=UI_PRIMARY_SOFT, fg=UI_MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(10, 0))
        tk.Label(
            candidate, textvariable=self.candidate_metrics_var,
            bg=UI_PRIMARY_SOFT, fg=UI_TEXT,
            font=("Microsoft YaHei UI", 10), wraplength=500, justify="left",
        ).pack(anchor="w", pady=(4, 8))
        tk.Label(
            candidate, text="独立验证", bg=UI_PRIMARY_SOFT, fg=UI_MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w")
        tk.Label(
            candidate, textvariable=self.validation_metrics_var,
            bg=UI_PRIMARY_SOFT, fg=UI_TEXT,
            font=("Microsoft YaHei UI", 10), wraplength=500, justify="left",
        ).pack(anchor="w", pady=(4, 0))
        actions = tk.Frame(candidate, bg=UI_PRIMARY_SOFT)
        actions.pack(fill="x", side="bottom", pady=(16, 0))
        ttk.Button(
            actions,
            text="审核并启用完整候选",
            style="Accent.TButton",
            command=self._activate_candidate,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="回滚上一完整版本",
            style="Danger.TButton",
            command=self._rollback,
        ).pack(side="left", padx=10)
        ttk.Label(
            current,
            text="候选包含同一不可变收据中的 DINO 输出层叠加层、缺陷特征库和融合阈值；启用或回滚始终整体执行。",
            style="CardMuted.TLabel", wraplength=500, justify="left",
        ).pack(anchor="w", side="bottom", pady=(16, 0))
        self.optimize_comparison_viewport.bind_mousewheel_tree()

    def _sync_feedback_loop(self, *_args: object) -> None:
        """Render the local loop only from existing, observed UI state."""

        if not hasattr(self, "feedback_loop"):
            return
        raw_state = self.candidate_state_var.get().strip().upper()
        try:
            feedback_count = int(self.total_var.get())
        except (TypeError, ValueError):
            feedback_count = 0
        completed: set[int] = set()
        active: int | None = 0 if feedback_count > 0 else None
        error: int | None = None
        if raw_state == "RUNNING":
            completed = {0} if feedback_count > 0 else set()
            active = 1
        elif raw_state == "READY FOR REVIEW":
            completed = {0, 1, 2}
            active = 3
        elif raw_state == "ACTIVE":
            completed = {0, 1, 2, 3}
            active = 4
        elif raw_state in {"FAILED", "FAIL", "ERROR"}:
            active = None
            error = 1
        self.feedback_loop.set_state(
            active_index=active,
            completed_indexes=completed,
            error_index=error,
        )

    def _build_acceptance_tab(self) -> None:
        page = self.acceptance_page.body
        page.columnconfigure(0, weight=1)
        page.rowconfigure(4, weight=1)
        self._competition_intro(
            page,
            active_stage="acceptance",
            title="性能测试与验收",
            hint="GPU 用于竞赛验收；CPU 按 2500×2500 单图 P50 < 2 秒进行诊断",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 16))

        config = ttk.Frame(page, style="Card.TFrame", padding=(16, 14))
        config.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        config.columnconfigure(0, weight=1)
        standards = tk.Frame(config, bg=UI_PANEL_ALT)
        standards.grid(row=0, column=0, sticky="w")
        self._segment_option(
            standards, display_text("AUTO"), self.benchmark_standard_var, "auto",
            self._benchmark_standard_changed,
        ).pack(side="left")
        self._segment_option(
            standards, "RTX 4070 · P50 < 100 ms", self.benchmark_standard_var,
            "internal_rtx4070", self._benchmark_standard_changed,
        ).pack(side="left")
        self._segment_option(
            standards, "RTX 2060 · P50 < 200 ms", self.benchmark_standard_var,
            "official_rtx2060", self._benchmark_standard_changed,
        ).pack(side="left")
        self._segment_option(
            standards, "CPU 诊断 · P50 < 2 秒", self.benchmark_standard_var,
            CPU_DIAGNOSTIC_STANDARD, self._benchmark_standard_changed,
        ).pack(side="left")
        actions = ttk.Frame(config, style="Card.TFrame")
        actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.benchmark_run_button = ttk.Button(
            actions, text="开始验收", style="Accent.TButton", command=self._start_benchmark,
        )
        self.benchmark_run_button.pack(side="right")
        self.open_benchmark_report_button = ttk.Button(
            actions, text="打开报告", style="Secondary.TButton",
            command=self._open_benchmark_report, state="disabled",
        )
        self.open_benchmark_report_button.pack(side="right", padx=(0, 8))
        ElidedTextLabel(
            config,
            self.benchmark_detected_var,
            max_chars=100,
            style="CardMuted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(9, 0))

        hero = tk.Frame(
            page, bg=UI_PANEL, highlightthickness=1,
            highlightbackground=UI_BORDER, padx=18, pady=14,
        )
        hero.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        hero_left = tk.Frame(hero, bg=UI_PANEL)
        hero_left.pack(side="left")
        tk.Label(
            hero_left, text="测试状态", bg=UI_PANEL, fg=UI_MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w")
        self.benchmark_hero_label = self._status_badge(
            hero_left,
            self.benchmark_hero_var,
            waiting_text="等待性能测试",
        )
        self.benchmark_hero_label.configure(
            font=("Microsoft YaHei UI", 20, "bold"), padx=14, pady=7
        )
        self.benchmark_hero_label.pack(anchor="w", pady=(2, 0))

        MetricTargetCard(
            hero,
            title="当前测试模式",
            rows=(
                ("输入", "2500 × 2500"),
                ("设备", self.benchmark_target_device_var),
                ("P50", self.benchmark_target_p50_var),
            ),
            accent=True,
        ).pack(side="left", fill="y", padx=(18, 0))
        MetricTargetCard(
            hero,
            title="当前实测",
            rows=(
                ("设备", self.benchmark_actual_device_var),
                ("分辨率", self.benchmark_resolution_var),
                ("P50", self.benchmark_p50_var),
                ("硬件预检", self.benchmark_hardware_var),
                ("样本数", self.benchmark_samples_var),
            ),
        ).pack(side="left", fill="y", padx=(10, 0))
        sample = tk.Frame(hero, bg=UI_PANEL)
        sample.pack(side="right", fill="x", expand=True, padx=(32, 0))
        tk.Label(
            sample, text="性能测试样本目录", bg=UI_PANEL, fg=UI_MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w")
        sample_row = tk.Frame(sample, bg=UI_PANEL)
        sample_row.pack(fill="x", pady=(4, 0))
        benchmark_path_entry = ttk.Entry(
            sample_row,
            textvariable=self.benchmark_image_dir_var,
            state="readonly",
        )
        benchmark_path_entry.pack(side="left", fill="x", expand=True)
        bind_path_entry(benchmark_path_entry, self.benchmark_image_dir_var)
        ttk.Button(
            sample_row, text="选择目录", style="Secondary.TButton",
            command=lambda: self.benchmark_image_dir_var.set(
                filedialog.askdirectory(title="选择性能测试样本目录")
                or self.benchmark_image_dir_var.get()
            ),
        ).pack(side="left", padx=(8, 0))

        metrics = ttk.Frame(page)
        metrics.grid(row=3, column=0, sticky="ew", pady=(0, 16))
        metric_defs = (
            (display_text("P50 LATENCY"), self.benchmark_p50_var),
            ("硬件预检", self.benchmark_hardware_var),
            (display_text("RESOLUTION"), self.benchmark_resolution_var),
            ("测试样本数", self.benchmark_samples_var),
        )
        last_metric_column = len(metric_defs) - 1
        for column, (label, variable) in enumerate(metric_defs):
            metrics.columnconfigure(column, weight=1, uniform="benchmark_metric")
            self._metric_card(metrics, label, variable).grid(
                row=0, column=column, sticky="ew",
                padx=(0 if column == 0 else 5, 0 if column == last_metric_column else 5),
            )

        self.acceptance_lower_viewport = VerticalScrollPanel(page)
        self.acceptance_lower_viewport.grid(row=4, column=0, sticky="nsew")
        lower_host = self.acceptance_lower_viewport.body
        lower_host.columnconfigure(0, weight=1)
        lower_host.rowconfigure(0, weight=1)
        lower = ttk.Frame(lower_host)
        lower.grid(row=0, column=0, sticky="nsew")
        lower.columnconfigure((0, 1), weight=1, uniform="benchmark_lower")
        checklist = ttk.Frame(lower, style="Card.TFrame", padding=(18, 16))
        checklist.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(checklist, text="验证清单", style="CardSection.TLabel").pack(anchor="w")
        checks = (
            (display_text("Hardware Match"), "hardware"),
            (display_text("Resolution Match"), "resolution"),
            (display_text("Latency Target"), "latency"),
            ("样本协议", "protocol"),
        )
        for label, key in checks:
            row = ttk.Frame(checklist, style="ValueRow.TFrame")
            row.pack(fill="x", pady=(10, 0))
            ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
            self._status_badge(
                row,
                self.benchmark_validation_vars[key],
                waiting_text="等待测试",
            ).pack(side="right")

        details = ttk.Frame(lower, style="Card.TFrame", padding=(18, 16))
        details.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        detail_header = ttk.Frame(details, style="ValueRow.TFrame")
        detail_header.pack(fill="x")
        ttk.Label(detail_header, text="详细证据", style="CardSection.TLabel").pack(side="left")
        self.benchmark_details_button = ttk.Button(
            detail_header, text="收起", style="Ghost.TButton",
            command=self._toggle_benchmark_details,
        )
        self.benchmark_details_button.pack(side="right")
        self.benchmark_nonexclusive_check = ttk.Checkbutton(
            detail_header, text="GPU 占用时仍计时（仅诊断）",
            variable=self.benchmark_allow_nonexclusive_var,
        )
        self.benchmark_nonexclusive_check.pack(side="right")
        self.benchmark_details_body = ttk.Label(
            details, textvariable=self.benchmark_status_var, style="CardMuted.TLabel",
            justify="left", wraplength=520,
        )
        self.benchmark_details_body.pack(anchor="nw", fill="both", expand=True, pady=(12, 0))
        self.acceptance_lower_viewport.bind_mousewheel_tree()

    def _toggle_benchmark_details(self) -> None:
        if self.benchmark_details_expanded:
            self.benchmark_details_body.pack_forget()
            self.benchmark_details_button.configure(text="展开")
        else:
            self.benchmark_details_body.pack(
                anchor="nw", fill="both", expand=True, pady=(12, 0)
            )
            self.benchmark_details_button.configure(text="收起")
        self.benchmark_details_expanded = not self.benchmark_details_expanded

    def _apply_benchmark_mode(self) -> None:
        """Refresh target labels and controls for the selected benchmark mode."""

        standard = self.benchmark_standard_var.get()
        if standard == CPU_DIAGNOSTIC_STANDARD:
            self.benchmark_target_device_var.set("当前 CPU")
            self.benchmark_target_p50_var.set("< 2000 ms")
            self.benchmark_run_button.configure(text="开始 CPU 测试")
            self.benchmark_nonexclusive_check.configure(state="disabled")
        elif standard == "official_rtx2060":
            self.benchmark_target_device_var.set("RTX 2060 官方线")
            self.benchmark_target_p50_var.set("< 200 ms")
            self.benchmark_run_button.configure(text="开始验收")
            self.benchmark_nonexclusive_check.configure(state="normal")
        elif standard == "internal_rtx4070":
            self.benchmark_target_device_var.set("RTX 4070 内部线")
            self.benchmark_target_p50_var.set("< 100 ms")
            self.benchmark_run_button.configure(text="开始验收")
            self.benchmark_nonexclusive_check.configure(state="normal")
        else:
            self.benchmark_target_device_var.set("自动检测")
            self.benchmark_target_p50_var.set("自动选择")
            self.benchmark_run_button.configure(text="开始测试")
            self.benchmark_nonexclusive_check.configure(state="normal")

    def _benchmark_standard_changed(self) -> None:
        self._detect_benchmark_device()

    def _detect_benchmark_device(self) -> None:
        """Detect test hardware and pre-select a safe matching mode."""
        try:
            if self.benchmark_standard_var.get() == CPU_DIAGNOSTIC_STANDARD:
                name = cpu_device_name()
                self.benchmark_detected_var.set(
                    f"CPU 诊断设备：{name} · P50 目标 < 2000 ms · 不作为竞赛 GPU 验收声明"
                )
                self.benchmark_actual_device_var.set(name)
                self._apply_benchmark_mode()
                return
            if not torch.cuda.is_available():
                if self.benchmark_standard_var.get() == "auto":
                    self.benchmark_standard_var.set(CPU_DIAGNOSTIC_STANDARD)
                    self._detect_benchmark_device()
                    return
                self.benchmark_detected_var.set(
                    "未检测到 CUDA GPU；可选择“CPU 诊断 · P50 < 2 秒”"
                )
                self.header_device_var.set("CPU")
                self.benchmark_actual_device_var.set("未检测到 CUDA GPU")
                self._apply_benchmark_mode()
                return
            raw_device_name = torch.cuda.get_device_name(0)
            device_name = " ".join(raw_device_name.strip().casefold().split())
            self.header_device_var.set(raw_device_name.replace("NVIDIA GeForce ", ""))
            self.benchmark_actual_device_var.set(
                raw_device_name.replace("NVIDIA GeForce ", "")
            )
            if device_name == "nvidia geforce rtx 2060":
                self.benchmark_detected_var.set("当前设备：RTX 2060（官方线可测）")
                if self.benchmark_standard_var.get() == "auto":
                    self.benchmark_standard_var.set("official_rtx2060")
            elif device_name == "nvidia geforce rtx 4070":
                self.benchmark_detected_var.set(
                    "当前设备：RTX 4070（内部强化线可测；2060 官方线需换机实测）"
                )
                if self.benchmark_standard_var.get() == "auto":
                    self.benchmark_standard_var.set("internal_rtx4070")
            else:
                self.benchmark_detected_var.set(
                    f"当前设备：{raw_device_name}（不在标准允许清单内）"
                )
            self._apply_benchmark_mode()
        except Exception as error:  # pragma: no cover - environment dependent
            self.benchmark_detected_var.set(f"设备检测失败：{error}")
            self.header_device_var.set("Unavailable")
            self.benchmark_actual_device_var.set("不可用")

    def _start_benchmark(self) -> None:
        if self.busy:
            return
        if not self._require_ready_model("执行性能测试"):
            return
        image_dir = Path(self.benchmark_image_dir_var.get().strip())
        if not image_dir.is_dir():
            self._showwarning("缺少样本目录", "请先选择有效的性能测试样本目录。")
            return
        category = self.category_var.get()
        self.busy = True
        self.benchmark_run_button.configure(state="disabled")
        standard = self.benchmark_standard_var.get()
        if standard == "auto":
            self._detect_benchmark_device()
            standard = self.benchmark_standard_var.get()
            if standard == "auto":
                standard = (
                    "internal_rtx4070"
                    if torch.cuda.is_available()
                    else CPU_DIAGNOSTIC_STANDARD
                )
                self.benchmark_standard_var.set(standard)
        cpu_diagnostic = standard == CPU_DIAGNOSTIC_STANDARD
        self._apply_benchmark_mode()
        self.benchmark_status_var.set("正在执行20次预热和100次同步单图计时...")
        self.benchmark_hero_var.set("RUNNING")
        self.benchmark_p50_var.set("测试中")
        self.benchmark_hardware_var.set(
            "不适用（CPU）" if cpu_diagnostic else "测试中"
        )
        self.benchmark_resolution_var.set("测试中")
        self.benchmark_samples_var.set("测试中")
        for variable in self.benchmark_validation_vars.values():
            variable.set("RUNNING")
        self.status_var.set(
            "正在执行 CPU 性能诊断" if cpu_diagnostic else "正在执行性能验收"
        )
        stamp = f"{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
        output_root = Path(self.pointer_path).parent / "benchmark_runs"
        config_root = output_root / "configs"
        report_path = output_root / f"{standard}_{stamp}.yaml"
        config_path = config_root / f"{standard}_{stamp}.yaml"
        config = {
            "schema_version": 2,
            "benchmark_standard": standard,
            "active_models_pointer": str(Path(self.pointer_path).resolve()),
            "device": "cpu" if cpu_diagnostic else "cuda:0",
            "category": category,
            "image_dir": str(image_dir.resolve()),
            "warmup": 20,
            "runs": 100,
            "allow_nonexclusive_measurement": bool(
                not cpu_diagnostic
                and self.benchmark_allow_nonexclusive_var.get()
            ),
            "output": str(report_path.resolve()),
            "measurement_note": (
                "UI CPU diagnostic: P50 < 2000 ms; not competition claim eligible"
                if cpu_diagnostic
                else "UI-generated immutable benchmark protocol"
            ),
            "timing_scope": (
                "model/runtime inference only; disk decode, 2500x2500 preparation, "
                "model loading, and output encoding excluded"
            ),
        }
        config_root.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        def work() -> None:
            try:
                exit_code = run_benchmark(config_path)
                report = yaml.safe_load(
                    report_path.read_text(encoding="utf-8")
                ) or {}
                self._post_ui(
                    self._benchmark_complete, report, report_path, exit_code
                )
            except Exception as error:
                self._post_ui(self._benchmark_failed, error)

        threading.Thread(target=work, daemon=True).start()

    def _benchmark_complete(
        self, report: dict[str, object], path: Path, exit_code: int
    ) -> None:
        self.busy = False
        self.benchmark_run_button.configure(state="normal")
        self.last_benchmark_report = path
        self.open_benchmark_report_button.configure(state="normal")
        performance = report.get("performance", {})
        if not isinstance(performance, dict):
            performance = {}
        validation = performance.get("validation", {})
        if not isinstance(validation, dict):
            validation = {}
        measurement = report.get("measurement", {})
        if not isinstance(measurement, dict):
            measurement = {}
        cpu_diagnostic = (
            report.get("selected_standard") == CPU_DIAGNOSTIC_STANDARD
            or report.get("measurement_mode") == "cpu_diagnostic"
        )
        p50 = report.get("p50_ms")
        standard = report.get("standard", {})
        if not isinstance(standard, dict):
            standard = {}
        legacy_p95_report = (
            str(standard.get("metric", "")).strip().casefold() == "p95_ms"
            or (
                p50 is None
                and report.get("p95_ms") is not None
            )
            or (
                "p95_under_target" in report
                and "p50_under_target" not in report
                and "latency_p50_strictly_under_target" not in validation
            )
        )
        p50_text = (
            "历史性能报告"
            if legacy_p95_report
            else ("等待测试" if p50 is None else f"{float(p50):.2f} ms")
        )
        self.benchmark_p50_var.set(p50_text)
        self.benchmark_actual_device_var.set(
            str(report.get("device_name") or "等待测试")
        )
        resolution = report.get("resolution", [])
        self.benchmark_resolution_var.set(
            " × ".join(str(value) for value in resolution) if resolution else "等待测试"
        )
        exclusivity = report.get("gpu_exclusivity", {})
        memory_used = exclusivity.get("memory_used_mib_at_preflight") if isinstance(exclusivity, dict) else None
        self.benchmark_hardware_var.set(
            "不适用（CPU）"
            if cpu_diagnostic
            else ("等待测试" if memory_used is None else f"{float(memory_used):.0f} MB")
        )
        measured_runs = report.get("runs")
        self.benchmark_samples_var.set(
            str(int(measured_runs)) if isinstance(measured_runs, (int, float)) else "等待测试"
        )
        if legacy_p95_report:
            self.benchmark_status_var.set(
                "历史协议报告不兼容：该报告使用旧统计协议生成，仅用于历史诊断，"
                f"不用于当前 P50 验收。请重新运行性能测试。报告：{path}"
            )
            self.benchmark_hero_var.set("DIAGNOSTIC")
            self.benchmark_validation_vars["hardware"].set(
                "PASS" if bool(validation.get("hardware_match")) else "FAIL"
            )
            self.benchmark_validation_vars["resolution"].set(
                "PASS" if bool(validation.get("resolution_match")) else "FAIL"
            )
            self.benchmark_validation_vars["latency"].set("WARNING · 需重测")
            self.benchmark_validation_vars["protocol"].set(
                "PASS"
                if bool(validation.get("warmup_sufficient"))
                and bool(validation.get("runs_sufficient"))
                else "FAIL"
            )
            self.status_var.set("历史性能报告仅作诊断，请重新运行 P50 验收")
            return
        performance_status = performance.get("status", report.get("status", "UNKNOWN"))
        if cpu_diagnostic:
            self.benchmark_status_var.set(
                f"CPU 诊断{display_status(performance_status)} · "
                f"设备：{report.get('device_name', '等待测试')} · "
                f"P50：{p50_text} · 目标：P50 < 2000 ms\n"
                f"分辨率匹配：{display_boolean(validation.get('resolution_match'))} · "
                f"样本协议满足：{display_boolean(bool(validation.get('warmup_sufficient')) and bool(validation.get('runs_sufficient')))} · "
                f"P50 小于 2 秒：{display_boolean(report.get('p50_under_target'))} · "
                f"本结果仅用于 CPU 诊断，不作为竞赛 GPU 验收声明 · 报告：{path}"
            )
        else:
            self.benchmark_status_var.set(
                f"{display_status(performance_status)} · "
                f"当前测试设备：{report.get('device_name', '等待测试')} · P50：{p50_text}\n"
                f"硬件匹配：{display_boolean(validation.get('hardware_match'))} · "
                f"分辨率匹配：{display_boolean(validation.get('resolution_match'))} · "
                f"延迟达标：{display_boolean(report.get('p50_under_target'))} · "
                f"结果可用于验收声明：{display_boolean(performance.get('claim_eligible'))} · "
                f"报告：{path}"
            )
        nonexclusive_diagnostic = bool(
            measurement.get("completed")
            and report.get("measurement_mode") == "contended_diagnostic"
        )
        hero = (
            "DIAGNOSTIC"
            if nonexclusive_diagnostic
            else ("PASS" if exit_code == 0 else "FAIL")
        )
        self.benchmark_hero_var.set(hero)
        self.benchmark_validation_vars["hardware"].set(
            "PASS" if bool(validation.get("hardware_match")) else "FAIL"
        )
        self.benchmark_validation_vars["resolution"].set(
            "PASS" if bool(validation.get("resolution_match")) else "FAIL"
        )
        self.benchmark_validation_vars["latency"].set(
            "PASS" if bool(validation.get("latency_p50_strictly_under_target")) else "FAIL"
        )
        self.benchmark_validation_vars["protocol"].set(
            "PASS"
            if bool(validation.get("warmup_sufficient"))
            and bool(validation.get("runs_sufficient"))
            else "FAIL"
        )
        if cpu_diagnostic:
            self.status_var.set(
                "CPU 性能诊断达标（P50 < 2 秒；不作为竞赛验收声明）"
                if exit_code == 0
                else "CPU 性能诊断未达标（P50 必须 < 2 秒）"
            )
        elif nonexclusive_diagnostic:
            self.status_var.set("性能诊断完成（GPU非独占）")
        else:
            self.status_var.set(
                "性能验收通过" if exit_code == 0 else "性能验收未通过"
            )

    def _benchmark_failed(self, error: Exception) -> None:
        self.busy = False
        self.benchmark_run_button.configure(state="normal")
        self.benchmark_status_var.set("性能测试执行失败")
        self.benchmark_hero_var.set("FAIL")
        self.benchmark_p50_var.set("等待测试")
        self.benchmark_hardware_var.set("等待测试")
        self.benchmark_resolution_var.set("等待测试")
        self.benchmark_samples_var.set("等待测试")
        for variable in self.benchmark_validation_vars.values():
            variable.set("FAIL")
        self.status_var.set("性能测试执行失败")
        self._showerror("性能测试执行失败", str(error))

    def _open_benchmark_report(self) -> None:
        self._open_local_path(self.last_benchmark_report)

    def _build_records_tab(self) -> None:
        page = self.records_page.body
        page.columnconfigure(0, weight=1)
        page.rowconfigure(3, weight=1)
        header = self._competition_intro(
            page,
            active_stage=None,
            title="反馈记录",
            hint="赛题对应：训练反馈、验证反馈与模型版本全过程追溯",
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(
            header,
            textvariable=self.records_count_var,
            style="PageSubtitle.TLabel",
        ).place(relx=1.0, rely=1.0, anchor="se")

        toolbar = ttk.Frame(page, style="Card.TFrame", padding=(14, 10))
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(toolbar, text="搜索", style="CardMuted.TLabel").pack(side="left")
        search = ttk.Entry(toolbar, textvariable=self.record_search_var, width=24)
        search.pack(side="left", padx=(8, 12))
        ttk.Combobox(
            toolbar, textvariable=self.record_role_filter_var,
            values=("全部角色", "训练", "验证"), state="readonly", width=10,
        ).pack(side="left", padx=(0, 8))
        ttk.Combobox(
            toolbar, textvariable=self.record_source_filter_var,
            values=("全部来源", "图片", "视频"), state="readonly", width=10,
        ).pack(side="left", padx=(0, 8))
        ttk.Combobox(
            toolbar, textvariable=self.record_type_filter_var,
            values=("全部结果", "正确", "误检", "漏检"), state="readonly", width=10,
        ).pack(side="left", padx=(0, 8))
        self.record_model_combo = ttk.Combobox(
            toolbar, textvariable=self.record_model_filter_var,
            values=("全部版本",), state="readonly", width=16,
        )
        self.record_model_combo.pack(side="left")

        actions = ttk.Frame(page)
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self.delete_records_button = ttk.Button(
            actions,
            text="删除选中记录",
            style="Danger.TButton",
            command=self._delete_selected_records,
            state="disabled",
        )
        self.delete_records_button.pack(side="left")
        self.clear_records_button = ttk.Button(
            actions,
            text="清空当前产品记录",
            style="Danger.TButton",
            command=self._clear_current_category_records,
            state="disabled",
        )
        self.clear_records_button.pack(side="left", padx=(8, 0))
        self.query_active_learning_button = ttk.Button(
            actions,
            text="主动学习查询",
            style="Accent.TButton",
            command=self._query_active_learning_pool,
        )
        self.query_active_learning_button.pack(side="left", padx=(8, 0))
        self.open_active_learning_button = ttk.Button(
            actions,
            text="打开查询清单",
            style="Secondary.TButton",
            command=self._open_active_learning_manifest,
            state="disabled",
        )
        self.open_active_learning_button.pack(side="left", padx=(8, 0))
        ttk.Label(
            actions,
            textvariable=self.active_learning_status_var,
            style="Muted.TLabel",
        ).pack(side="right")

        table = ttk.Frame(page, style="Card.TFrame", padding=(1, 1))
        table.grid(row=3, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ("time", "role", "source", "type", "label", "score", "threshold", "region", "defect", "model")
        self.records_tree = ttk.Treeview(
            table, columns=columns, show="headings", selectmode="extended"
        )
        headings = {
            "time": "时间", "role": "用途", "source": "来源", "type": "反馈类型",
            "label": "人工标签", "score": "分数", "threshold": "阈值", "region": "缺陷区域",
            "defect": "缺陷类型", "model": "模型版本",
        }
        widths = {
            "time": 145, "role": 60, "source": 85, "type": 80, "label": 70,
            "score": 72, "threshold": 72, "region": 110, "defect": 90, "model": 150,
        }
        self.records_tree.tag_configure("even", background="#ffffff")
        self.records_tree.tag_configure("odd", background=UI_PANEL_ALT)
        self.records_tree.tag_configure("hover", background="#eef5ff")
        for key in columns:
            self.records_tree.heading(key, text=headings[key])
            anchor = "e" if key in {"score", "threshold"} else "w"
            self.records_tree.column(key, width=widths[key], minwidth=52, anchor=anchor, stretch=False)
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.records_tree.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.records_tree.xview)
        self.records_tree.configure(
            yscrollcommand=vertical.set, xscrollcommand=horizontal.set
        )
        self.records_tree.grid(row=0, column=0, sticky="nsew")
        self.records_empty_label = tk.Label(
            table,
            text="暂无反馈记录",
            bg=UI_PANEL,
            fg=UI_MUTED,
            font=("Microsoft YaHei UI", 12, "bold"),
            padx=18,
            pady=10,
        )
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.records_tree.bind(
            "<<TreeviewSelect>>", self._records_selection_changed
        )
        self.records_tree.bind(
            "<Delete>", lambda _event: self._delete_selected_records()
        )
        self.records_tree.bind("<Motion>", self._records_hover)
        self.records_tree.bind("<Leave>", self._records_hover_clear)
        for variable in (
            self.record_search_var, self.record_role_filter_var,
            self.record_source_filter_var, self.record_type_filter_var,
            self.record_model_filter_var,
        ):
            variable.trace_add("write", lambda *_args: self._refresh_all())

    def _query_active_learning_pool(self) -> None:
        """Rank an unlabeled image directory for operator annotation."""
        if self.busy:
            return
        if not self._require_ready_model("生成主动学习查询"):
            return
        selected = filedialog.askdirectory(title="选择主动学习候选图片目录")
        if not selected:
            return
        budget = simpledialog.askinteger(
            "主动学习查询预算",
            "本轮需要人工标注多少张图片？",
            initialvalue=20,
            minvalue=1,
            parent=self.root,
        )
        if budget is None:
            return
        category = self.category_var.get()
        requested_device = self.device_var.get()
        self.busy = True
        self.query_active_learning_button.configure(state="disabled")
        self.active_learning_status_var.set("正在计算不确定性、分支分歧和特征稀有度...")
        self.status_var.set("正在生成主动学习查询清单...")

        def work() -> None:
            try:
                runtime, version, _device = self._runtime(category, requested_device)
                result = query_unlabeled_pool(
                    runtime,
                    self.store,
                    category,
                    selected,
                    budget=budget,
                    model_version=version,
                )
                self._post_ui(self._active_learning_complete, result)
            except Exception as error:
                self._post_ui(self._active_learning_failed, error)

        threading.Thread(target=work, daemon=True).start()

    def _active_learning_complete(self, result: dict[str, object]) -> None:
        self.busy = False
        self.query_active_learning_button.configure(state="normal")
        raw_path = result.get("manifest_path")
        self.last_active_learning_manifest = (
            self.store.root / str(raw_path) if raw_path else None
        )
        self.open_active_learning_button.configure(
            state="normal" if self.last_active_learning_manifest else "disabled"
        )
        raw_candidates = result.get("candidates", [])
        query_images = [
            Path(str(row["path"])).resolve()
            for row in raw_candidates
            if isinstance(row, dict)
            and row.get("path")
            and Path(str(row["path"])).is_file()
        ]
        if query_images:
            # Reuse the normal detection/feedback navigator, but preserve the
            # active-learning ranking rather than the source directory order.
            self._release_video()
            self.source_type = "image"
            self.folder_images = query_images
            self.image_index = 0
            self.source_name = query_images[0].name
            self._load_image(query_images[0])
        self.active_learning_status_var.set(
            f"已生成 {int(result.get('selected_count', 0))} 张待标注样本；"
            f"候选池 {int(result.get('pool_count', 0))} 张 · "
            f"清单 {self.last_active_learning_manifest}"
        )
        self.status_var.set("主动学习查询完成，等待人工标注")
        self._showinfo(
            "主动学习查询完成",
            f"已选出 {result.get('selected_count', 0)} 张高价值样本。\n"
            f"请按清单完成标注后，再提交训练/验证反馈。\n\n"
            f"清单：{self.last_active_learning_manifest}",
        )

    def _active_learning_failed(self, error: Exception) -> None:
        self.busy = False
        self.query_active_learning_button.configure(state="normal")
        self.active_learning_status_var.set("主动学习查询失败")
        self.status_var.set("主动学习查询失败")
        self._showerror("主动学习查询失败", str(error))

    def _open_active_learning_manifest(self) -> None:
        self._open_local_path(self.last_active_learning_manifest)

    def _build_adapt_tab(self) -> None:
        page = self.adapt_page.body
        page.columnconfigure(0, weight=1)
        page.rowconfigure(3, weight=1)
        self._competition_intro(
            page,
            active_stage="adapt",
            title="快速适配",
            hint="赛题对应：100 张正常 + 30 张缺陷少样本快速部署",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 16))

        setup = ttk.Frame(page, style="Card.TFrame", padding=(18, 16))
        setup.grid(row=1, column=0, sticky="ew")
        setup.columnconfigure(1, weight=1)
        ttk.Label(setup, text="任务名称", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Entry(setup, textvariable=self.task_name_var).grid(row=0, column=1, sticky="ew")
        mode_frame = tk.Frame(setup, bg=UI_PANEL_ALT)
        mode_frame.grid(row=0, column=2, sticky="e", padx=(20, 0))
        self._segment_option(
            mode_frame, "图像级标签", self.annotation_mode_var, "image_label",
            self._annotation_mode_changed,
        ).pack(side="left")
        self._segment_option(
            mode_frame, "像素掩码", self.annotation_mode_var, "pixel_mask",
            self._annotation_mode_changed,
        ).pack(side="left")
        ttk.Label(
            setup, textvariable=self.adapt_instruction_var, style="CardMuted.TLabel",
            wraplength=720, justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        action_row = ttk.Frame(setup, style="ValueRow.TFrame")
        action_row.grid(row=1, column=2, sticky="e", padx=(16, 0), pady=(10, 0))
        self.adapt_button = ttk.Button(
            action_row, text="开始适配", style="Accent.TButton",
            command=self._start_custom_adaptation,
        )
        self.adapt_button.pack(side="left")
        self.evaluate_button = ttk.Button(
            action_row, text="运行真实标签评测", style="Secondary.TButton",
            command=self._evaluate_custom_test,
        )
        self.evaluate_button.pack(side="left", padx=(10, 0))

        data_row = ttk.Frame(page)
        data_row.grid(row=2, column=0, sticky="ew", pady=16)
        data_row.columnconfigure((0, 1, 2), weight=1, uniform="adapt_data")

        def data_card(
            column: int,
            count_variable: tk.StringVar,
            title: str,
            hint: str,
            variable: tk.StringVar,
            target: int,
        ) -> tuple[ttk.Entry, ttk.Button]:
            card = tk.Frame(
                data_row, bg=UI_PANEL, highlightthickness=1,
                highlightbackground=UI_BORDER, padx=12, pady=10,
            )
            card.grid(
                row=0, column=column, sticky="nsew",
                padx=(0 if column == 0 else 8, 0 if column == 2 else 8),
            )
            card_header = tk.Frame(card, bg=UI_PANEL)
            card_header.pack(fill="x")
            tk.Label(
                card_header, text=title, bg=UI_PANEL, fg=UI_TEXT,
                font=("Microsoft YaHei UI", 10, "bold"),
            ).pack(anchor="w")
            tk.Label(
                card, textvariable=count_variable, bg=UI_PANEL, fg=UI_TEXT,
                font=("Microsoft YaHei UI", 20, "bold"),
            ).pack(anchor="w", pady=(4, 1))
            tk.Label(
                card, text=hint, bg=UI_PANEL, fg=UI_MUTED,
                font=("Microsoft YaHei UI", 9),
            ).pack(anchor="w", pady=(8, 8))
            input_row = tk.Frame(card, bg=UI_PANEL)
            input_row.pack(fill="x")
            entry = ttk.Entry(input_row, textvariable=variable, state="readonly")
            entry.pack(side="left", fill="x", expand=True)
            bind_path_entry(entry, variable)

            def choose() -> None:
                selected = filedialog.askdirectory(title=f"选择{title}")
                if selected:
                    selected_path = Path(selected).resolve()
                    variable.set(str(selected_path))
                    actual = sum(
                        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                        for path in selected_path.rglob("*")
                    )
                    count_variable.set(f"{actual} / {target}")

            button = ttk.Button(
                input_row, text="选择", style="Secondary.TButton", command=choose,
            )
            button.pack(side="left", padx=(8, 0))
            return entry, button

        data_card(
            0, self.normal_sample_count_var, "正常样本", "等待选择 100 张正常样本目录",
            self.normal_dir_var, 100,
        )
        data_card(
            1, self.defect_sample_count_var, "缺陷样本", "等待选择 30 张缺陷样本目录",
            self.defect_dir_var, 30,
        )
        self.mask_entry, self.mask_button = data_card(
            2, self.mask_sample_count_var, "像素掩码", "仅像素掩码模式必需",
            self.mask_dir_var, 30,
        )

        self.adapt_lower_viewport = VerticalScrollPanel(page)
        self.adapt_lower_viewport.grid(row=3, column=0, sticky="nsew")
        lower_host = self.adapt_lower_viewport.body
        lower_host.columnconfigure(0, weight=1)
        lower_host.rowconfigure(0, weight=1)
        lower = ttk.Frame(lower_host)
        lower.grid(row=0, column=0, sticky="nsew")
        lower.columnconfigure((0, 1), weight=1, uniform="adapt_lower")
        advanced = ttk.Frame(lower, style="Card.TFrame", padding=(18, 16))
        advanced.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(advanced, text="独立评估", style="CardSection.TLabel").pack(anchor="w")
        ttk.Label(
            advanced,
            text=(
                "可选择真实标签 YAML；少于1000张时可确认继续作为开发诊断，"
                "1000+且满足冻结后首次评测时才可作为最终独立验收。"
            ),
            style="CardMuted.TLabel", wraplength=500, justify="left",
        ).pack(anchor="w", pady=(3, 12))
        tk.Label(
            advanced,
            textvariable=self.independent_test_count_var,
            bg=UI_PANEL,
            fg=UI_TEXT,
            font=("Microsoft YaHei UI", 20, "bold"),
        ).pack(anchor="w", pady=(0, 1))
        ttk.Label(
            advanced, text="独立测试", style="CardMuted.TLabel"
        ).pack(anchor="w", pady=(0, 10))
        manifest_row = ttk.Frame(advanced, style="ValueRow.TFrame")
        manifest_row.pack(fill="x")
        manifest_entry = ttk.Entry(
            manifest_row, textvariable=self.test_manifest_var, state="readonly",
        )
        manifest_entry.pack(side="left", fill="x", expand=True)
        bind_path_entry(manifest_entry, self.test_manifest_var)

        def choose_manifest() -> None:
            selected = filedialog.askopenfilename(
                title="选择独立评估 YAML",
                filetypes=(("YAML 清单", "*.yaml *.yml"), ("所有文件", "*.*")),
            )
            if selected:
                self._small_manifest_approval = None
                self.test_manifest_var.set(str(Path(selected).resolve()))

        ttk.Button(
            manifest_row, text="选择清单", style="Secondary.TButton",
            command=choose_manifest,
        ).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            advanced, text="本测试集为模型冻结后首次评测", variable=self.fresh_attest_var,
        ).pack(anchor="w", pady=(8, 6))
        evaluation_metrics = tk.Frame(advanced, bg=UI_PANEL_ALT, padx=10, pady=7)
        evaluation_metrics.pack(fill="x")
        metric_items = (
            (display_text("Accuracy"), self.custom_accuracy_var),
            (display_text("Recall"), self.custom_recall_var),
            (display_text("F1"), self.custom_f1_var),
            (display_text("FPR"), self.custom_fpr_var),
            (display_text("Gate"), self.custom_gate_var),
        )
        for column, (label, variable) in enumerate(metric_items):
            evaluation_metrics.columnconfigure(column, weight=2 if label == "验收门槛" else 1)
            item = tk.Frame(evaluation_metrics, bg=UI_PANEL_ALT)
            item.grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 8, 0))
            tk.Label(
                item, text=label, bg=UI_PANEL_ALT, fg=UI_MUTED,
                font=("Segoe UI", 7),
            ).pack(anchor="w")
            if variable is self.custom_gate_var:
                self._status_badge(
                    item, variable, waiting_text="等待验证"
                ).pack(anchor="w")
            else:
                tk.Label(
                    item, textvariable=variable, bg=UI_PANEL_ALT, fg=UI_TEXT,
                    font=("Segoe UI", 9, "bold"),
                ).pack(anchor="w")

        progress = ttk.Frame(lower, style="Card.TFrame", padding=(18, 16))
        progress.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        top = ttk.Frame(progress, style="ValueRow.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="适配进度", style="CardSection.TLabel").pack(side="left")
        elapsed = ttk.Label(top, textvariable=self.adapt_elapsed_var, style="CardMuted.TLabel")
        elapsed.pack(side="right")
        ttk.Label(top, text="训练耗时", style="CardMuted.TLabel").pack(side="right", padx=(0, 6))
        ttk.Label(top, textvariable=self.adapt_count_var, style="CardMuted.TLabel").pack(side="right", padx=(0, 14))
        stage_row = ttk.Frame(progress, style="ValueRow.TFrame")
        stage_row.pack(fill="x", pady=(12, 8))
        ttk.Label(stage_row, text="当前阶段", style="CardMuted.TLabel").pack(side="left")
        self._status_badge(
            stage_row,
            self.adapt_stage_var,
            waiting_text="尚未开始适配",
        ).pack(side="left", padx=(8, 0))
        self._status_badge(
            stage_row,
            self.custom_gate_var,
            waiting_text="等待验证",
        ).pack(side="right")
        ttk.Label(stage_row, text="评估状态", style="CardMuted.TLabel").pack(side="right", padx=(0, 8))
        self.adapt_progress = ttk.Progressbar(progress, mode="indeterminate")
        self.adapt_progress.pack(fill="x")
        ttk.Label(
            progress, textvariable=self.adapt_status_var, style="CardMuted.TLabel",
            wraplength=500, justify="left",
        ).pack(anchor="w", pady=(10, 3))
        ttk.Label(
            progress, textvariable=self.test_status_var, style="CardMuted.TLabel",
            wraplength=500, justify="left",
        ).pack(anchor="w")

        threshold_panel = tk.Frame(
            progress,
            bg=UI_PANEL_ALT,
            highlightthickness=1,
            highlightbackground=UI_BORDER,
            padx=12,
            pady=10,
        )
        threshold_panel.pack(fill="x", pady=(14, 0))
        threshold_header = tk.Frame(threshold_panel, bg=UI_PANEL_ALT)
        threshold_header.pack(fill="x")
        tk.Label(
            threshold_header,
            text="验收门槛",
            bg=UI_PANEL_ALT,
            fg=UI_TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="left")
        ttk.Button(
            threshold_header,
            text="恢复默认",
            style="Secondary.TButton",
            command=self._reset_acceptance_thresholds,
        ).pack(side="right")
        ttk.Button(
            threshold_header,
            text="应用门槛",
            style="Secondary.TButton",
            command=self._apply_acceptance_thresholds,
        ).pack(side="right", padx=(0, 8))

        threshold_inputs = tk.Frame(threshold_panel, bg=UI_PANEL_ALT)
        threshold_inputs.pack(fill="x", pady=(10, 6))
        for column in range(3):
            threshold_inputs.columnconfigure(column, weight=1, uniform="gate_threshold")
        threshold_specs = (
            ("召回率下限", self.acceptance_recall_floor_var),
            ("基础 FPR 上限", self.acceptance_stage1_fpr_var),
            ("严格 FPR 上限", self.acceptance_stage2_fpr_var),
        )
        for column, (label, variable) in enumerate(threshold_specs):
            item = tk.Frame(threshold_inputs, bg=UI_PANEL_ALT)
            item.grid(
                row=0,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 7, 0),
            )
            tk.Label(
                item,
                text=label,
                bg=UI_PANEL_ALT,
                fg=UI_MUTED,
                font=("Microsoft YaHei UI", 8),
            ).pack(anchor="w")
            input_row = tk.Frame(item, bg=UI_PANEL_ALT)
            input_row.pack(anchor="w", pady=(3, 0))
            ttk.Spinbox(
                input_row,
                from_=0.1 if "FPR" in label else 0.0,
                to=100.0,
                increment=0.5,
                textvariable=variable,
                width=7,
            ).pack(side="left")
            tk.Label(
                input_row,
                text="%",
                bg=UI_PANEL_ALT,
                fg=UI_TEXT,
                font=("Segoe UI", 9, "bold"),
            ).pack(side="left", padx=(4, 0))
        tk.Label(
            threshold_panel,
            textvariable=self.acceptance_threshold_summary_var,
            bg=UI_PANEL_ALT,
            fg=UI_TEXT,
            font=("Microsoft YaHei UI", 8),
            justify="left",
            wraplength=500,
        ).pack(anchor="w", pady=(3, 0))
        tk.Label(
            threshold_panel,
            text="自定义门槛只改变评估判定，不改变100+30适配训练过程。",
            bg=UI_PANEL_ALT,
            fg=UI_MUTED,
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w", pady=(5, 0))
        self._refresh_acceptance_threshold_summary()
        self.adapt_lower_viewport.bind_mousewheel_tree()
        self._annotation_mode_changed()

    @staticmethod
    def _directory_row(
        parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, browse: bool
    ) -> tuple[ttk.Entry, ttk.Button | None]:
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky="w", pady=7)
        entry = ttk.Entry(
            parent, textvariable=variable,
            state="normal" if not browse else "readonly", width=48,
        )
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=7)
        button = None
        if browse:
            bind_path_entry(entry, variable)
            button = ttk.Button(
                parent, text="选择目录", style="Secondary.TButton",
                command=lambda v=variable: v.set(
                    filedialog.askdirectory(title=f"选择{label}") or v.get()
                ),
            )
            button.grid(row=row, column=2, sticky="e", pady=7)
        return entry, button

    @staticmethod
    def _file_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        filetypes: tuple[tuple[str, str], ...],
    ) -> tuple[ttk.Entry, ttk.Button]:
        ttk.Label(parent, text=label, width=14).grid(
            row=row, column=0, sticky="w", pady=7
        )
        entry = ttk.Entry(parent, textvariable=variable, state="readonly", width=48)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=7)
        bind_path_entry(entry, variable)

        def choose() -> None:
            selected = filedialog.askopenfilename(
                title=f"选择{label}", filetypes=filetypes
            )
            if selected:
                variable.set(str(Path(selected).resolve()))

        button = ttk.Button(parent, text="选择文件", style="Secondary.TButton", command=choose)
        button.grid(row=row, column=2, sticky="e", pady=7)
        return entry, button

    def _annotation_mode_changed(self) -> None:
        if self.annotation_mode_var.get() == "pixel_mask":
            self.adapt_count_var.set("适配样本 0 / 130 · 掩码 0 / 30")
            if self.mask_dir_var.get() in {"未选择", "无需选择（仅分类标签模式）"}:
                self.mask_sample_count_var.set("0 / 30")
            if self.mask_dir_var.get().startswith("无需选择"):
                self.mask_dir_var.set("未选择")
            self.mask_entry.configure(state="readonly")
            self.mask_button.configure(state="normal")
            self.adapt_instruction_var.set(
                "选择100张正常、30张缺陷和30张同名对应掩码，可进行精确区域监督。测试目录必须与适配样本内容哈希不重叠。"
            )
            self.adapt_status_var.set("尚未开始适配")
        else:
            self.adapt_count_var.set("适配样本 0 / 130")
            self.mask_sample_count_var.set("未启用")
            self.mask_dir_var.set("无需选择（仅分类标签模式）")
            self.mask_entry.configure(state="disabled")
            self.mask_button.configure(state="disabled")
            self.adapt_instruction_var.set(
                "仅需选择100张正常和30张缺陷；目录名称就是图像级标签。测试目录必须与适配样本内容哈希不重叠。"
            )
            self.adapt_status_var.set("尚未开始适配")

    def _create_category(self) -> None:
        """Start one isolated production product task and jump to adaptation."""
        if self.busy:
            return
        name = simpledialog.askstring(
            "新建产品",
            "输入新产品任务名称（字母/数字/下划线，单个文件夹名）：\n"
            "例如 screen、battery、middle_frame。",
            parent=self.root,
        )
        if not name:
            return
        name = name.strip()
        if not name or Path(name).name != name or name in {".", ".."}:
            self._showerror("产品名称无效", "产品任务名称必须是单个文件夹名（字母/数字/下划线）。")
            return
        if name in self.pointer.get("models", {}):
            self._showwarning("产品已存在", f"产品任务 {name} 已存在，请直接加载其模型指针。")
            self.category_var.set(name)
            return
        # A pending task remains in memory only.  Creating its output directory
        # here would make the strict custom trainer reject the task as an
        # overwrite.  The completed task receives its own active_models.yaml.
        self.category_label_var.set("产品/任务")
        self.pointer_path = None
        self.pointer = _empty_product_pointer()
        self.pointer["models"] = {name: "pending"}
        self.categories = [name]
        self.category_combo.configure(values=self.categories)
        self.task_name_var.set(name)
        self.current_category = name
        self.category_var.set(name)
        self.runtimes.clear()
        self.detection = None
        self.candidate = None
        self._reset_result()
        self._refresh_all()
        self._switch_tab(self.adapt_tab)
        self.adapt_status_var.set(
            f"新产品 {name}：请选择 100 张正常、30 张缺陷目录后点击「开始适配」。"
        )
        self.status_var.set(f"新建产品 {name}，请完成少样本适配")

    def _delete_category(self) -> None:
        """Detach the current product while preserving all persisted evidence."""
        if self.busy:
            return
        selected = self.category_var.get()
        if selected == EMPTY_PRODUCT_LABEL or selected not in self.categories:
            self._showwarning("无法移除", "当前没有可移除的产品任务。")
            return
        selected_ref = self.pointer.get("models", {}).get(selected)
        trained = selected_ref is not None and str(selected_ref).strip() != "pending"
        if not self._askyesno(
            "仅移除使用",
            f"确定仅从当前工作台移除产品任务 {selected} 吗？\n\n"
            + (
                "模型、评测文件和审计记录都会保留；之后可通过“加载产品任务”恢复使用。"
                if trained
                else "该产品尚未训练：只取消本次待适配任务，不删除任何磁盘文件。"
            ),
            parent=self.root,
            default="no",
        ):
            return
        state_warning = self._detach_current_product(
            selected,
            status=(
                f"已仅移除产品任务 {selected} 的使用（磁盘文件和审计记录保留）"
                if trained
                else f"已取消待适配产品任务 {selected}"
            ),
        )
        if state_warning:
            self._showwarning("仅移除使用", state_warning)

    def _clear_saved_task_reference(self, pointer_path: Path | None) -> str | None:
        """Delete active_task.yaml only when it still belongs to this window."""

        if not ACTIVE_TASK_STATE.is_file():
            return None
        try:
            if pointer_path is not None:
                state = yaml.safe_load(
                    ACTIVE_TASK_STATE.read_text(encoding="utf-8")
                ) or {}
                raw = state.get("active_models_pointer") if isinstance(state, dict) else None
                if not isinstance(raw, str) or not raw.strip():
                    return "活动任务状态格式异常，未删除该状态文件。"
                saved_pointer = Path(raw)
                if not saved_pointer.is_absolute():
                    saved_pointer = ACTIVE_TASK_STATE.parent / saved_pointer
                if saved_pointer.resolve(strict=False) != Path(pointer_path).resolve(
                    strict=False
                ):
                    # Another window activated a different task after this one
                    # loaded.  Never clear that newer selection.
                    return None
            ACTIVE_TASK_STATE.unlink(missing_ok=True)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            return f"活动任务状态清理失败：{error}"
        return None

    def _detach_current_product(self, selected: str, *, status: str) -> str | None:
        """Reset product-scoped UI state and forget only its saved reference."""

        previous_pointer = self.pointer_path
        self.pointer = _empty_product_pointer()
        self.pointer_path = None
        self.categories = [EMPTY_PRODUCT_LABEL]
        self.current_category = self.categories[0]
        self.category_var.set(self.current_category)
        self.category_combo.configure(values=self.categories)
        self.runtimes.clear()
        self.detection = None
        self.candidate = None
        self.last_benchmark_report = None
        self.last_active_learning_manifest = None
        if hasattr(self, "open_benchmark_report_button"):
            self.open_benchmark_report_button.configure(state="disabled")
        if hasattr(self, "open_active_learning_button"):
            self.open_active_learning_button.configure(state="disabled")
        if hasattr(self, "active_learning_status_var"):
            self.active_learning_status_var.set("尚未生成主动学习查询清单")
        self._reset_result()
        self._refresh_all()
        self.status_var.set(status)
        return self._clear_saved_task_reference(previous_pointer)

    def _product_audit_scope(self, category: str) -> tuple[str, str | None]:
        """Return the persisted run and bundle scope for destructive cleanup."""

        if self.pointer_path is None:
            raise ValueError("当前产品没有已加载的模型指针。")
        models = self.pointer.get("models")
        if not isinstance(models, dict) or category not in models:
            raise ValueError("当前模型指针中没有该产品。")
        bundle_path = resolve_artifact(
            models[category], base=Path(self.pointer_path).parent
        )
        payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError("模型包必须是 YAML 映射。")
        run_id = str(payload.get("run_id", bundle_path.parent.name)).strip()
        if not run_id:
            raise ValueError("模型包缺少可用的 run_id。")
        raw_hash = payload.get("bundle_hash")
        bundle_hash = None if raw_hash is None else str(raw_hash).strip().lower()
        if bundle_hash is not None and (
            len(bundle_hash) != 64
            or any(character not in "0123456789abcdef" for character in bundle_hash)
        ):
            raise ValueError("模型包 bundle_hash 无效，已拒绝删除。")
        return run_id, bundle_hash

    def _delete_category_files_and_audit(self) -> None:
        """Permanently remove one managed task and its exact audit scope."""

        if self.busy:
            self._showwarning("操作进行中", "请等待当前操作完成后再移除产品任务。")
            return
        selected = self.category_var.get()
        if selected == EMPTY_PRODUCT_LABEL or selected not in self.categories:
            self._showwarning("无法移除", "当前没有可移除的产品任务。")
            return
        selected_ref = self.pointer.get("models", {}).get(selected)
        trained = selected_ref is not None and str(selected_ref).strip() != "pending"
        if not trained:
            if self._askyesno(
                "取消待适配任务",
                f"产品任务 {selected} 尚未训练，没有模型文件或审计记录。\n\n"
                "是否取消本次待适配任务？",
                parent=self.root,
                default="no",
            ):
                warning = self._detach_current_product(
                    selected, status=f"已取消待适配产品任务 {selected}"
                )
                if warning:
                    self._showwarning("取消待适配任务", warning)
            return

        try:
            task_root = _validate_managed_product_task_root(
                self.pointer_path,
                self.pointer,
                selected,
            )
            run_id, bundle_hash = self._product_audit_scope(selected)
            file_count, total_bytes = _task_disk_summary(task_root)
            plan = self.store.plan_product_evidence_purge(
                selected,
                run_id,
                bundle_hash=bundle_hash,
            )
        except Exception as error:
            self._showerror("无法安全移除", str(error))
            return

        counts = dict(plan.get("counts", {}))
        size_mb = total_bytes / (1024 * 1024)
        if not self._askyesno(
            "移除磁盘文件和审计记录",
            f"即将永久移除产品任务 {selected}：\n\n"
            f"磁盘目录：{task_root}\n"
            f"任务文件：{file_count} 个，共 {size_mb:.1f} MB\n"
            f"反馈记录：{int(counts.get('records', 0))} 条；"
            f"审计图片：{int(counts.get('images', 0))} 个\n"
            f"候选/修订：{int(counts.get('candidates', 0))} / "
            f"{int(counts.get('revisions', 0))}；"
            f"特征库：{int(counts.get('feature_banks', 0))}；"
            f"查询清单：{int(counts.get('queries', 0))}\n\n"
            "不会删除 data 数据集、assets 源模型、外部评估清单或导出的提交包。\n"
            "此操作不可恢复，是否继续？",
            parent=self.root,
            default="no",
        ):
            return

        staged_root = task_root.with_name(
            f".pending-delete-{task_root.name}-{uuid4().hex}"
        )
        try:
            self.runtimes.clear()
            task_root.rename(staged_root)
        except OSError as error:
            self._showerror("磁盘移除失败", f"任务目录无法进入安全删除阶段：{error}")
            return

        try:
            audit_summary = self.store.purge_product_evidence(
                selected,
                run_id,
                bundle_hash=bundle_hash,
                expected_plan_sha256=str(plan["plan_sha256"]),
            )
        except Exception as error:
            rollback_error: Exception | None = None
            try:
                staged_root.rename(task_root)
            except OSError as restore_error:
                rollback_error = restore_error
            detail = f"审计清理失败，任务目录已恢复。\n\n{error}"
            if rollback_error is not None:
                detail = (
                    "审计清理失败，且任务目录自动恢复失败。请勿手动移动其他目录。\n"
                    f"隔离目录：{staged_root}\n恢复错误：{rollback_error}\n\n原始错误：{error}"
                )
            self._showerror("移除失败", detail)
            return

        task_cleanup_error: OSError | None = None
        try:
            shutil.rmtree(staged_root)
        except OSError as error:
            task_cleanup_error = error

        state_warning = self._detach_current_product(
            selected,
            status=f"已移除产品任务 {selected} 的磁盘文件和审计记录",
        )
        audit_pending = audit_summary.get("pending_cleanup_path")
        warnings: list[str] = []
        if task_cleanup_error is not None:
            warnings.append(
                f"任务已停用且审计已清理，但部分磁盘文件仍在隔离目录：\n"
                f"{staged_root}\n错误：{task_cleanup_error}"
            )
        if audit_pending:
            warnings.append(f"审计隔离文件待清理：{audit_pending}")
        if state_warning:
            warnings.append(state_warning)
        if warnings:
            self._showwarning("移除部分完成", "\n\n".join(warnings))
        else:
            removed = dict(audit_summary.get("counts", counts))
            self._showinfo(
                "移除完成",
                f"已永久移除产品任务 {selected}。\n\n"
                f"任务文件：{file_count} 个\n"
                f"反馈记录：{int(removed.get('records', 0))} 条\n"
                f"候选/修订/查询：{int(removed.get('candidates', 0))} / "
                f"{int(removed.get('revisions', 0))} / "
                f"{int(removed.get('queries', 0))}",
            )

    def _switch_tab(self, tab: ttk.Frame) -> None:
        try:
            self.notebook.select(tab)
        except Exception:
            pass

    def _notebook_tab_changed(self, _event: tk.Event | None = None) -> None:
        """Keep tab selection presentation-only; business mutations refresh data."""

        selected = self.notebook.select()
        self.records_navigation_button.configure(
            style=(
                "HeaderNavActive.TButton"
                if selected == str(self.records_tab)
                else "HeaderNav.TButton"
            )
        )
        if selected == str(self.detect_tab):
            self.root.after_idle(self._render_preview_if_size_changed)

    def _category_changed(self) -> None:
        selected = self.category_var.get()
        if self.busy:
            self.category_var.set(self.current_category)
            return
        if selected == self.current_category:
            return
        self.current_category = selected
        self.detection = None
        self.candidate = None
        self._reset_result()
        self._refresh_all()

    def _choose_model_pointer(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(
            title="选择产品任务 active_models.yaml",
            initialdir=str(PROJECT_ROOT / "outputs" / "custom_tasks"),
            filetypes=(("YAML 模型指针", "*.yaml *.yml"), ("所有文件", "*.*")),
        )
        if not selected:
            return
        try:
            self._activate_model_pointer(Path(selected))
        except Exception as error:
            self._showerror("产品任务加载失败", str(error))

    def _activate_model_pointer(
        self,
        pointer_path: Path,
        *,
        selected_category: str | None = None,
    ) -> None:
        resolved, payload = _read_model_pointer(
            pointer_path,
            expected_dataset="custom",
        )
        models = dict(payload["models"])
        self.pointer_path = resolved
        self.pointer = payload
        self.categories = sorted(models)
        self.category_combo.configure(values=self.categories)
        self.current_category = (
            selected_category if selected_category in models else self.categories[0]
        )
        self.category_var.set(self.current_category)
        self.category_label_var.set("产品/任务")
        ACTIVE_TASK_STATE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_TASK_STATE.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "active_models_pointer": str(resolved),
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        self.runtimes.clear()
        self.detection = None
        self.candidate = None
        self._reset_result()
        self._refresh_all()
        self.status_var.set(
            f"已加载产品任务：{resolved}"
        )

    def _start_adaptation_progress(self) -> None:
        self._adapt_started_at = time.monotonic()
        self.adapt_stage_var.set("RUNNING · 数据校验完成，正在适配")
        self.adapt_elapsed_var.set("00:00")
        self.adapt_progress.start(12)
        self._update_adaptation_elapsed()

    def _update_adaptation_elapsed(self) -> None:
        if self._adapt_started_at is None or self.closing:
            return
        elapsed = max(0, int(time.monotonic() - self._adapt_started_at))
        minutes, seconds = divmod(elapsed, 60)
        self.adapt_elapsed_var.set(f"{minutes:02d}:{seconds:02d}")
        self.root.after(1000, self._update_adaptation_elapsed)

    def _stop_adaptation_progress(self, stage: str) -> None:
        self.adapt_progress.stop()
        self._adapt_started_at = None
        self.adapt_stage_var.set(stage)

    def _confirm_small_test_manifest(
        self,
        manifest_path: Path,
        sample_count: int,
    ) -> bool:
        """Ask once per manifest revision before using a sub-1000 holdout."""

        if sample_count >= FINAL_INDEPENDENT_TEST_MINIMUM:
            return True
        approval = (
            str(manifest_path),
            sha256_file(manifest_path),
            sample_count,
        )
        if approval == self._small_manifest_approval:
            return True
        accepted = self._askyesno(
            "评估样本少于1000张，是否继续？",
            f"当前清单包含 {sample_count} 张有标签图片，少于最终独立验收要求的 "
            f"{FINAL_INDEPENDENT_TEST_MINIMUM} 张。\n\n"
            "选择“是”：继续适配/评估，结果标记为“开发诊断”，"
            "不作为最终竞赛验收声明。\n"
            "选择“否”：取消并返回更换清单。",
            icon="warning",
            default="no",
        )
        if accepted:
            self._small_manifest_approval = approval
        return accepted

    def _acceptance_thresholds(self) -> dict[str, float]:
        return _acceptance_thresholds_from_percent(
            self.acceptance_recall_floor_var.get(),
            self.acceptance_stage1_fpr_var.get(),
            self.acceptance_stage2_fpr_var.get(),
        )

    def _refresh_acceptance_threshold_summary(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        try:
            configured = thresholds or self._acceptance_thresholds()
        except ValueError as error:
            self.acceptance_threshold_summary_var.set(f"输入待检查：{error}")
            return
        recall = configured["recall_floor"]
        stage1 = configured["stage1_fpr_strict_upper_bound"]
        stage2 = configured["stage2_fpr_strict_upper_bound"]
        self.acceptance_threshold_summary_var.set(
            f"基础通过：Recall ≥ {recall:.1%} 且 FPR < {stage1:.1%}  ·  "
            f"严格通过：Recall ≥ {recall:.1%} 且 FPR < {stage2:.1%}"
        )

    def _apply_acceptance_thresholds(self) -> None:
        try:
            thresholds = self._acceptance_thresholds()
        except ValueError as error:
            self._showerror("验收门槛无效", str(error))
            return
        self._refresh_acceptance_threshold_summary(thresholds)
        self.custom_gate_var.set("WAITING")
        self.test_status_var.set("验收门槛已更新，请重新运行真实标签评测")

    def _reset_acceptance_thresholds(self) -> None:
        self.acceptance_recall_floor_var.set(
            f"{DEFAULT_ACCEPTANCE_RECALL_PERCENT:.1f}"
        )
        self.acceptance_stage1_fpr_var.set(
            f"{DEFAULT_ACCEPTANCE_STAGE1_FPR_PERCENT:.1f}"
        )
        self.acceptance_stage2_fpr_var.set(
            f"{DEFAULT_ACCEPTANCE_STAGE2_FPR_PERCENT:.1f}"
        )
        self._apply_acceptance_thresholds()

    def _start_custom_adaptation(self) -> None:
        if self.busy:
            return
        task_name = self.task_name_var.get().strip()
        if not task_name or Path(task_name).name != task_name or task_name in {".", ".."}:
            self._showerror("任务名称无效", "任务名称只能包含单个文件夹名。")
            return
        annotation_mode = self.annotation_mode_var.get()
        values = (self.normal_dir_var.get(), self.defect_dir_var.get())
        if any(value in {"未选择", ""} for value in values):
            self._showwarning("缺少适配目录", "请先选择100正常和30缺陷目录。")
            return
        mask_dir = None
        if annotation_mode == "pixel_mask":
            mask_dir = self.mask_dir_var.get()
            if mask_dir in {"未选择", ""}:
                self._showwarning("缺少掩码目录", "像素掩码模式需要选择30张对应掩码。")
                return
        try:
            episode = build_custom_episode(
                normal_dir=self.normal_dir_var.get(), defect_dir=self.defect_dir_var.get(),
                mask_dir=mask_dir, category=task_name,
            )
            manifest_value = self.test_manifest_var.get().strip()
            if manifest_value and not manifest_value.startswith("可选"):
                manifest_path = Path(manifest_value).resolve(strict=True)
                support_hashes = {
                    sample.sha256 for sample in episode.dataset.samples
                }
                test_samples = load_custom_test_manifest(
                    manifest_path,
                    minimum_samples=1,
                    excluded_hashes=support_hashes,
                )
                if not self._confirm_small_test_manifest(
                    manifest_path,
                    len(test_samples),
                ):
                    self.adapt_status_var.set(
                        "已取消：评估清单不足1000张，请更换清单后重试"
                    )
                    return
                episode = replace(
                    episode,
                    independent_test_count=len(test_samples),
                    test_normal_count=sum(
                        sample.label == 0 for sample in test_samples
                    ),
                    test_defect_count=sum(
                        sample.label == 1 for sample in test_samples
                    ),
                    test_manifest=manifest_path,
                )
            base_config = load_config(
                PROJECT_ROOT / "configs" / "custom_training_base.yaml"
            )
            output_root = PROJECT_ROOT / "outputs" / "custom_tasks" / task_name
            if output_root.exists():
                raise FileExistsError(f"任务输出目录已存在，请换一个任务名称: {output_root}")
            config = replace(
                base_config, dataset="custom", data_root=episode.dataset.normal[0].image_path.parent,
                run_name=task_name, candidate_id=f"AOI-CUSTOM-{task_name}",
                output_root=output_root, categories=(task_name,), minimum_pooled_test=0,
            )
        except Exception as error:
            self._showerror("适配数据检查失败", str(error))
            return
        self.busy = True
        normal_count = len(episode.dataset.normal)
        defect_count = len(episode.dataset.defect)
        mask_count = sum(sample.mask_path is not None for sample in episode.dataset.defect)
        self.normal_sample_count_var.set(f"{normal_count} / 100")
        self.defect_sample_count_var.set(f"{defect_count} / 30")
        self.mask_sample_count_var.set(
            f"{mask_count} / 30"
            if annotation_mode == "pixel_mask"
            else "未启用"
        )
        self.independent_test_count_var.set(
            f"{episode.independent_test_count} / 1000+"
        )
        self.adapt_button.configure(state="disabled")
        self.evaluate_button.configure(state="disabled")
        mode_text = "仅分类标签弱监督" if annotation_mode == "image_label" else "像素掩码监督"
        self.adapt_status_var.set(f"数据检查通过，正在执行100+30少样本适配（{mode_text}）...")
        self.adapt_count_var.set(
            "适配样本 130 / 130 · 掩码 30 / 30"
            if annotation_mode == "pixel_mask"
            else "适配样本 130 / 130"
        )
        self._start_adaptation_progress()

        def work() -> None:
            try:
                pointer_path = train_custom_episode(config, episode)
                self._post_ui(self._custom_adaptation_complete, pointer_path)
            except Exception as error:
                self._post_ui(self._custom_adaptation_failed, error)

        threading.Thread(target=work, daemon=True).start()

    def _custom_adaptation_complete(self, pointer_path: Path) -> None:
        self.busy = False
        self.adapt_button.configure(state="normal")
        self.evaluate_button.configure(state="normal")
        self._stop_adaptation_progress("COMPLETE · 模型已就绪")
        resolved, new_pointer = _read_model_pointer(
            pointer_path,
            expected_dataset="custom",
        )
        new_models = dict(new_pointer["models"])
        selected_product = next(iter(new_models))
        self._activate_model_pointer(
            resolved,
            selected_category=selected_product,
        )
        self.adapt_status_var.set(f"适配完成，独立产品模型指针：{resolved}")
        self.status_var.set(f"产品 {selected_product} 模型已就绪")
        mode_text = "仅分类标签弱监督" if new_pointer.get("annotation_mode") == "image_label" else "像素掩码监督"
        self._showinfo(
            "新产品适配完成",
            f"标注方式：{mode_text}\n"
            f"产品 {selected_product} 已自动设为当前任务。\n"
            f"独立模型指针：{resolved}",
        )

    def _custom_adaptation_failed(self, error: Exception) -> None:
        self.busy = False
        self.adapt_button.configure(state="normal")
        self.evaluate_button.configure(state="normal")
        self._stop_adaptation_progress("FAILED · 请检查数据与配置")
        self.adapt_status_var.set("适配失败")
        self._showerror("新任务适配失败", str(error))

    def _evaluate_custom_test(self) -> None:
        if self.busy:
            return
        if self.pointer.get("dataset") != "custom":
            self._showwarning("尚未完成新任务适配", "请先完成当前新任务的100+30少样本适配。")
            return
        manifest_value = self.test_manifest_var.get().strip()
        if manifest_value.startswith("可选") or not Path(manifest_value).is_file():
            self._showwarning(
                "缺少标签清单",
                "请先选择包含正常/异常真实标签的 YAML 清单。"
                "少于1000张可在确认后作为开发诊断。",
            )
            return
        if not self.pointer.get("models") or self.category_var.get() not in self.pointer["models"]:
            self._showwarning("模型未就绪", "请先完成新任务适配。")
            return
        try:
            manifest_path = Path(manifest_value).resolve(strict=True)
            test_samples = load_custom_test_manifest(
                manifest_path,
                minimum_samples=1,
            )
        except Exception as error:
            self._showerror("评估清单检查失败", str(error))
            return
        test_count = len(test_samples)
        development_diagnostic = test_count < FINAL_INDEPENDENT_TEST_MINIMUM
        if development_diagnostic and not self._confirm_small_test_manifest(
            manifest_path,
            test_count,
        ):
            self.test_status_var.set("已取消少样本开发诊断")
            return
        if not development_diagnostic and not self.fresh_attest_var.get():
            self._showwarning(
                "缺少冻结后声明",
                "请勾选“本测试集为模型冻结后首次评测（未在冻结前观察）”。\n"
                "只有冻结后首次运行的1000+独立测试才能作为最终独立成绩声明。",
            )
            return
        try:
            acceptance_thresholds = self._acceptance_thresholds()
        except ValueError as error:
            self._showerror("验收门槛无效", str(error))
            return
        self._refresh_acceptance_threshold_summary(acceptance_thresholds)
        self.busy = True
        self.evaluate_button.configure(state="disabled")
        self.test_status_var.set(
            f"正在评测 {test_count} 张真实标签图片"
            + (
                "（开发诊断，不作为最终验收）..."
                if development_diagnostic
                else "（1000+独立验收）..."
            )
        )
        for variable in (
            self.custom_accuracy_var, self.custom_recall_var,
            self.custom_f1_var, self.custom_fpr_var,
        ):
            variable.set("--")
        self.custom_gate_var.set("RUNNING")
        category = self.category_var.get()
        requested_device = self.device_var.get()
        stamp = f"{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
        task_root = Path(self.pointer_path).parent
        config_dir = task_root / "ui_test_configs"
        output_root = task_root / "test_runs" / stamp
        config_path = config_dir / f"test_{stamp}.yaml"
        config_payload = {
            "schema_version": 1,
            "active_models_pointer": str(Path(self.pointer_path).resolve()),
            "category": category,
            "test_manifest": str(manifest_path),
            "device": requested_device,
            "batch_size": 4,
            "minimum_total_images": (
                test_count
                if development_diagnostic
                else FINAL_INDEPENDENT_TEST_MINIMUM
            ),
            "evaluation_role": (
                "development_diagnostic"
                if development_diagnostic
                else "independent_test"
            ),
            "fresh_independent_test_attested": (
                False if development_diagnostic else True
            ),
            "test_observed_before_freeze": development_diagnostic,
            "acceptance_thresholds": acceptance_thresholds,
            "output_root": str(output_root.resolve()),
            "timestamp_output": False,
            "refuse_overwrite": True,
        }
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        def work() -> None:
            try:
                exit_code = run_custom_evaluation(config_path)
                if exit_code != 0:
                    raise RuntimeError(
                        f"真实标签评测返回非零状态: {exit_code}"
                    )
                output = output_root / "metrics.yaml"
                metrics = yaml.safe_load(output.read_text(encoding="utf-8")) or {}
                self._post_ui(self._custom_test_complete, metrics, output)
            except Exception as error:
                self._post_ui(self._custom_test_failed, error)

        threading.Thread(target=work, daemon=True).start()

    def _custom_test_complete(self, metrics: dict[str, object], output: Path) -> None:
        self.busy = False
        self.evaluate_button.configure(state="normal")
        self.custom_accuracy_var.set(f"{float(metrics['accuracy']):.2%}")
        self.custom_recall_var.set(f"{float(metrics['recall']):.2%}")
        self.custom_f1_var.set(f"{float(metrics['f1']):.2%}")
        self.custom_fpr_var.set(f"{float(metrics['fpr']):.2%}")
        self.custom_gate_var.set(str(metrics["acceptance"]["status"]))
        development_diagnostic = (
            metrics.get("evaluation_role") == "development_diagnostic"
        )
        self.independent_test_count_var.set(
            f"{int(metrics['images'])} / 1000"
            + ("（开发诊断）" if development_diagnostic else "+")
        )
        self.test_status_var.set(
            f"评测完成：{metrics['images']}张 · "
            f"准确率 {float(metrics['accuracy']):.2%} · "
            f"召回率 {float(metrics['recall']):.2%} · "
            f"F1 分数 {float(metrics['f1']):.2%} · "
            f"假阳性率 {float(metrics['fpr']):.2%} · "
            f"{display_gate(metrics['acceptance']['status'])} · "
            + (
                "开发诊断，不具备最终独立声明资格 · "
                if development_diagnostic
                else ""
            )
            + str(output)
        )
        self.status_var.set("真实标签评测完成")
        self._showinfo(
            "开发诊断评测完成"
            if development_diagnostic
            else "真实标签评测完成",
            f"准确率：{float(metrics['accuracy']):.2%}\n"
            f"召回率：{float(metrics['recall']):.2%}\n"
            f"F1 分数：{float(metrics['f1']):.2%}\n"
            f"假阳性率：{float(metrics['fpr']):.2%}\n"
            + (
                "评估样本少于1000张，本结果只作为开发诊断，"
                "不作为最终竞赛验收声明。\n"
                if development_diagnostic
                else ""
            )
            + f"证据：{output}",
        )

    def _custom_test_failed(self, error: Exception) -> None:
        self.busy = False
        self.evaluate_button.configure(state="normal")
        self.test_status_var.set("真实标签评测失败")
        self.custom_gate_var.set("FAILED")
        self.status_var.set("真实标签评测失败")
        self._showerror("真实标签评测失败", str(error))

    def _choose_image(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(title="选择待检测图片", filetypes=(("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("所有文件", "*.*")))
        if not selected:
            return
        self._load_media_path(Path(selected).resolve())

    def _drop_media(self, event: tk.Event) -> str:
        if self.busy:
            return "break"
        raw = str(getattr(event, "data", "")).strip()
        if not raw:
            return "break"
        try:
            values = self.root.tk.splitlist(raw)
            path = Path(str(values[0]).strip("{}")).resolve(strict=True)
            self._load_media_path(path)
        except (OSError, RuntimeError, ValueError) as error:
            self._showerror("无法加载拖放文件", str(error))
        return "break"

    def _load_media_path(self, selected_path: Path) -> None:
        suffix = selected_path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            self._load_video_source(selected_path)
            return
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图片/视频格式: {selected_path.suffix}")
        self._release_video()
        self.source_type = "image"
        self.source_name = selected_path.name
        self.folder_images = sorted(
            (
                path.resolve()
                for path in selected_path.parent.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
        self.image_index = self.folder_images.index(selected_path)
        self._load_image(selected_path)

    def _load_image(self, path: Path, *, preserve_view: bool = False) -> None:
        self.image_path = path
        self.image_bytes = self.image_path.read_bytes()
        image = Image.open(BytesIO(self.image_bytes)).convert("RGB")
        self.image_rgb = np.array(image, copy=True)
        self.heatmap = None
        self.defect_region = None
        self.detection = None
        if not preserve_view:
            self.viewer_fit_mode = True
            self.viewer_zoom = 1.0
            self.viewer_zoom_text_var.set("适应")
        self.viewer_file_var.set(self.image_path.name)
        self.viewer_meta_var.set(f"{image.width} × {image.height}  ·  图像")
        self._reset_result()
        self._render_preview()
        self._update_navigation_buttons()
        self.status_var.set(f"已选择 {self.image_path.name}")

    def _navigate_image(self, step: int) -> None:
        if self.busy:
            return
        if self.source_type == "video":
            self._load_video_frame(
                self.video_frame_index + step,
                preserve_view=True,
            )
            return
        if not self.folder_images:
            return
        target = self.image_index + step
        if target < 0 or target >= len(self.folder_images):
            return
        self.image_index = target
        self._load_image(
            self.folder_images[self.image_index],
            preserve_view=True,
        )

    def _keyboard_navigate(self, step: int) -> None:
        focused = self.root.focus_get()
        if focused is not None and focused.winfo_class() in {
            "Entry", "TEntry", "Text", "TCombobox", "TSpinbox"
        }:
            return
        if self.notebook.select() != str(self.detect_tab):
            return
        self._navigate_image(step)

    def _update_navigation_buttons(self) -> None:
        if self.source_type == "video":
            total = self.video_frame_count
            index = self.video_frame_index
            name = f"{self.video_path.name} · 帧 {index + 1}" if self.video_path else "视频"
        else:
            total = len(self.folder_images)
            index = self.image_index
            name = self.image_path.name if self.image_path else ""
        if total == 0 or index < 0:
            self.image_position_var.set("未选择文件夹")
            previous_state = next_state = "disabled"
        else:
            self.image_position_var.set(f"{index + 1} / {total}  {name}")
            previous_state = "normal" if index > 0 else "disabled"
            next_state = "normal" if index + 1 < total else "disabled"
        self.previous_button.configure(state=previous_state)
        self.next_button.configure(state=next_state)

    def _draw_viewer_empty_state(self) -> None:
        if not hasattr(self, "image_canvas"):
            return
        width = max(320, self.image_canvas.winfo_width())
        height = max(240, self.image_canvas.winfo_height())
        self.image_canvas.delete("all")
        center_x, center_y = width // 2, height // 2
        self.image_canvas.create_rectangle(
            center_x - 34, center_y - 48, center_x + 34, center_y + 6,
            outline="#405064", width=2,
        )
        self.image_canvas.create_line(
            center_x - 22, center_y - 6, center_x - 6, center_y - 21,
            center_x + 17, center_y - 1, fill="#405064", width=2,
        )
        self.image_canvas.create_text(
            center_x, center_y + 34, text="尚未选择检测图像",
            fill="#d6dee8", font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.image_canvas.create_text(
            center_x, center_y + 58, text="拖放图片/视频到此处，或点击下方选择文件",
            fill=UI_CANVAS_TEXT, font=("Microsoft YaHei UI", 9),
        )

    def _preview_canvas_changed(self, event: tk.Event) -> None:
        """Debounce expensive image/heatmap resampling while resizing."""

        size = (event.width, event.height)
        if size == self._last_preview_size:
            return
        if self._preview_resize_after_id is not None:
            try:
                self.root.after_cancel(self._preview_resize_after_id)
            except tk.TclError:
                pass
        self._preview_resize_after_id = self.root.after(
            48, self._render_preview_after_resize
        )

    def _render_preview_after_resize(self) -> None:
        self._preview_resize_after_id = None
        if self.closing or self.notebook.select() != str(self.detect_tab):
            return
        self._render_preview_if_size_changed()

    def _render_preview_if_size_changed(self) -> None:
        if not hasattr(self, "image_canvas"):
            return
        size = (
            self.image_canvas.winfo_width(),
            self.image_canvas.winfo_height(),
        )
        if size != self._last_preview_size:
            self._render_preview()

    def _zoom_viewer(self, delta: float) -> None:
        if self.image_rgb is None:
            return
        self.viewer_fit_mode = False
        self.viewer_zoom = min(3.0, max(0.35, self.viewer_zoom + delta))
        self.viewer_zoom_text_var.set(f"{self.viewer_zoom:.0%}")
        self._render_preview()

    def _fit_viewer(self) -> None:
        self.viewer_fit_mode = True
        self.viewer_zoom = 1.0
        self.viewer_zoom_text_var.set("适应")
        self._render_preview()

    def _toggle_fullscreen(self) -> None:
        current = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not current)

    def _render_preview(self) -> None:
        if not hasattr(self, "image_canvas"):
            return
        self._last_preview_size = (
            self.image_canvas.winfo_width(),
            self.image_canvas.winfo_height(),
        )
        if self.image_rgb is None:
            self._draw_viewer_empty_state()
            return
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        if canvas_width < 80 or canvas_height < 80:
            return
        image = Image.fromarray(self.image_rgb)
        if self.heatmap is not None and self.show_heatmap_var.get():
            heatmap = cv2.resize(
                self.heatmap.astype(np.float32), (image.width, image.height)
            )
            colored = cv2.applyColorMap(
                np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), cv2.COLORMAP_TURBO
            )
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            image = Image.blend(image, Image.fromarray(colored), 0.38)
        max_width = max(1, canvas_width - 12)
        max_height = max(1, canvas_height - 12)
        base_scale = min(max_width / image.width, max_height / image.height, 1.0)
        scale = base_scale if self.viewer_fit_mode else base_scale * self.viewer_zoom
        target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(target, Image.Resampling.LANCZOS)
        left = (canvas_width - image.width) // 2
        top = (canvas_height - image.height) // 2
        self.display_box = (left, top, image.width, image.height)
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(left, top, image=self.preview_image, anchor="nw")
        if self.defect_region is not None:
            x1, y1, x2, y2 = self._original_to_canvas(self.defect_region)
            self.roi_rectangle = self.image_canvas.create_rectangle(
                x1, y1, x2, y2, outline=UI_DANGER, width=3
            )

    def _original_to_canvas(self, region: list[int]) -> tuple[int, int, int, int]:
        left, top, width, height = self.display_box
        image_height, image_width = self.image_rgb.shape[:2]
        x1, y1, x2, y2 = region
        return (
            round(left + x1 / image_width * width),
            round(top + y1 / image_height * height),
            round(left + x2 / image_width * width),
            round(top + y2 / image_height * height),
        )

    def _inside_preview(self, x: int, y: int) -> bool:
        left, top, width, height = self.display_box
        return left <= x <= left + width and top <= y <= top + height

    def _roi_press(self, event: tk.Event) -> None:
        if self.image_rgb is None or not self._inside_preview(event.x, event.y):
            return
        self.roi_start = (event.x, event.y)
        if self.roi_rectangle is not None:
            self.image_canvas.delete(self.roi_rectangle)
        self.roi_rectangle = self.image_canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline=UI_DANGER, width=3
        )

    def _roi_drag(self, event: tk.Event) -> None:
        if self.roi_start is None or self.roi_rectangle is None:
            return
        left, top, width, height = self.display_box
        x = min(max(event.x, left), left + width)
        y = min(max(event.y, top), top + height)
        self.image_canvas.coords(
            self.roi_rectangle, self.roi_start[0], self.roi_start[1], x, y
        )

    def _roi_release(self, event: tk.Event) -> None:
        if self.roi_start is None or self.image_rgb is None:
            return
        left, top, width, height = self.display_box
        end_x = min(max(event.x, left), left + width)
        end_y = min(max(event.y, top), top + height)
        start_x, start_y = self.roi_start
        self.roi_start = None
        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))
        if x2 - x1 < 5 or y2 - y1 < 5:
            self.defect_region = None
            self._render_preview()
            return
        image_height, image_width = self.image_rgb.shape[:2]
        self.defect_region = [
            max(0, round((x1 - left) / width * image_width)),
            max(0, round((y1 - top) / height * image_height)),
            min(image_width, round((x2 - left) / width * image_width)),
            min(image_height, round((y2 - top) / height * image_height)),
        ]
        self.status_var.set(f"已标注缺陷区域 {self.defect_region}")

    def _choose_video(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(
            title="选择检测视频",
            filetypes=(("视频", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")),
        )
        if not selected:
            return
        self._load_media_path(Path(selected).resolve())

    def _load_video_source(self, path: Path) -> None:
        self._release_video()
        self.video_path = path
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            self._showerror("视频读取失败", f"无法打开视频: {self.video_path}")
            return
        self.video_capture = capture
        self.video_frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.video_frame_index = 0
        self.source_type = "video"
        self.source_name = self.video_path.name
        self.folder_images = []
        self.image_index = -1
        self._load_video_frame(0)

    def _load_video_frame(self, index: int, *, preserve_view: bool = False) -> None:
        if self.video_capture is None or index < 0 or index >= self.video_frame_count:
            return
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.video_capture.read()
        if not ok:
            self._showerror("视频读取失败", f"无法读取第 {index + 1} 帧")
            return
        self.video_frame_index = index
        self.image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoded, payload = cv2.imencode(".png", frame)
        if not encoded:
            raise RuntimeError("failed to encode video frame")
        self.image_bytes = payload.tobytes()
        self.image_path = Path(f"{self.video_path.stem}_frame_{index:06d}.png")
        self.heatmap = None
        self.defect_region = None
        self.detection = None
        if not preserve_view:
            self.viewer_fit_mode = True
            self.viewer_zoom = 1.0
            self.viewer_zoom_text_var.set("适应")
        height, width = self.image_rgb.shape[:2]
        self.viewer_file_var.set(f"{self.video_path.name} · 帧 {index + 1}")
        self.viewer_meta_var.set(f"{width} × {height}  ·  视频")
        self._reset_result()
        self._render_preview()
        self._update_navigation_buttons()
        self.status_var.set(f"已载入视频第 {index + 1} 帧")

    def _release_video(self) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = None
        self.video_path = None
        self.video_frame_count = 0
        self.video_frame_index = 0

    def _close(self) -> None:
        if self.closing:
            return
        self.closing = True
        try:
            for after_id in (self._preview_resize_after_id,):
                if after_id is None:
                    continue
                try:
                    self.root.after_cancel(after_id)
                except tk.TclError:
                    pass
            self._preview_resize_after_id = None
            if self.video_cancel_event is not None:
                self.video_cancel_event.set()
            self._release_video()
        finally:
            # ``quit`` stops mainloop even if a future cleanup operation
            # fails; ``destroy`` then closes the native window and widgets.
            try:
                self.root.quit()
            finally:
                try:
                    self.root.destroy()
                except tk.TclError:
                    pass

    def _has_ready_model(self, category: str | None = None) -> bool:
        selected = category or self.category_var.get()
        models = self.pointer.get("models", {})
        if not isinstance(models, dict) or selected not in models:
            return False
        return str(models[selected]).strip() not in {"", "pending"}

    def _require_ready_model(self, action: str) -> bool:
        if self._has_ready_model():
            return True
        self._showwarning(
            "尚未加载产品模型",
            f"{action}前，请先新建并完成产品适配，或加载已有产品任务。",
        )
        return False

    def _runtime(
        self,
        category: str | None = None,
        requested_device: str | None = None,
    ) -> tuple[AdaptedRuntime, str, str]:
        category = category or self.category_var.get()
        if not self._has_ready_model(category):
            raise RuntimeError(
                f"产品 {category} 尚无可用模型；请先完成 100+30 少样本适配或加载产品任务。"
            )
        device = resolve_device(requested_device or self.device_var.get())
        key = (category, device)
        if key not in self.runtimes:
            bundle_ref = self.pointer["models"][key[0]]
            if str(bundle_ref).strip() == "pending":
                raise RuntimeError(
                    f"产品 {key[0]} 尚未训练：请先在「快速适配」页完成 100+30 少样本适配。"
                )
            bundle_path = resolve_artifact(bundle_ref, base=self.pointer_path.parent)
            bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {}
            runtime = AdaptedRuntime(bundle_path, device=device)
            self.runtimes[key] = (runtime, str(bundle.get("run_id", bundle_path.parent.name)))
        runtime, version = self.runtimes[key]
        state = apply_active_feedback(
            runtime,
            self.store,
            category,
            model_version=version,
        )
        runtime._last_feedback_runtime_state = state
        return runtime, version, device

    def _model_version(self, category: str | None = None) -> str:
        category = category or self.category_var.get()
        models = self.pointer.get("models", {})
        if not isinstance(models, dict) or category not in models:
            return "unavailable"
        bundle_ref = models[category]
        if str(bundle_ref).strip() == "pending":
            return "pending"
        bundle_path = resolve_artifact(
            bundle_ref, base=self.pointer_path.parent
        )
        bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {}
        return str(bundle.get("run_id", bundle_path.parent.name))

    def _start_detection(self) -> None:
        if self.busy:
            return
        if not self._require_ready_model("开始检测"):
            return
        if self.image_rgb is None or self.image_path is None:
            self._showwarning("缺少图片", "请先选择一张待检测图片。")
            return
        self.busy = True
        self.result_var.set("RUNNING")
        self.detect_button.configure(state="disabled")
        self.status_var.set("正在加载模型并检测...")
        image = self.image_rgb.copy()
        category = self.category_var.get()
        requested_device = self.device_var.get()
        sample_name = self.image_path.name

        def work() -> None:
            try:
                runtime, version, device = self._runtime(category, requested_device)
                raw = runtime.predict_explain(image, sample_name)
                # ``AdaptedRuntime`` applies an active complete revision under
                # its inference lock.  Do not separately patch a legacy
                # threshold here, or a result can carry mismatched components.
                result = dict(raw)
                self._post_ui(
                    self._detection_complete, result, category, version, device
                )
            except Exception as error:
                self._post_ui(self._task_failed, error)

        threading.Thread(target=work, daemon=True).start()

    def _detection_complete(
        self, result: dict[str, object], category: str, version: str, device: str
    ) -> None:
        self.busy = False
        self.detect_button.configure(state="normal")
        runtime_state = None
        for (cached_category, _device), (runtime, cached_version) in self.runtimes.items():
            if cached_category == category and cached_version == version:
                state = getattr(runtime, "_last_feedback_runtime_state", None)
                runtime_state = None if state is None else state.receipt()
                break
        self.detection = {
            "result": result,
            "category": category,
            "model_version": version,
            "feedback_runtime_state": runtime_state,
        }
        anomaly = bool(result["decision"])
        self.operator_label_var.set("异常" if anomaly else "正常")
        self.heatmap = result.get("_anomaly_map")
        self._render_preview()
        self.result_var.set("NG" if anomaly else "PASS")
        score = float(result["fusion_score"])
        threshold = float(result["fusion_threshold"])
        self.score_var.set(f"{score:.4f}")
        self.threshold_var.set(f"{threshold:.4f}")
        self.margin_var.set(f"{score - threshold:+.4f}")
        margins = result["branch_margins"]
        self.branch_var.set(f"通用 {margins['general']:.2f}  局部 {margins['local']:.2f}  已知 {margins['known']:.2f}")
        raw_types = result.get("defect_types", [])
        defect_types = [str(value) for value in raw_types] if isinstance(raw_types, list) else []
        if anomaly and not defect_types:
            defect_types = ["appearance"]
        type_labels = [
            "外观异常" if value == "appearance" else value
            for value in defect_types
        ]
        self.defect_types_result_var.set("、".join(type_labels) or "无")
        if anomaly and type_labels:
            self.defect_type_var.set("、".join(type_labels))
        self.status_var.set(f"检测完成 · 外观模型 · {device}")

    def _task_failed(self, error: Exception) -> None:
        self.busy = False
        self.detect_button.configure(state="normal")
        if self.candidate_state_var.get() == "RUNNING":
            self.candidate_state_var.set("FAILED")
            self.improvement_var.set("候选版本生成失败")
        if self.result_var.get() == "RUNNING":
            self.result_var.set("ERROR")
        self.status_var.set("操作失败")
        self._showerror("操作失败", str(error))

    def _submit_feedback(self) -> None:
        if self._feedback_submit_inflight:
            return
        self._feedback_submit_inflight = True
        self.feedback_submit_button.configure(state="disabled", text="提交中...")
        self._set_feedback_submit_state("VALIDATING", "正在校验...")
        try:
            if (
                self.detection is None
                or self.image_bytes is None
                or self.image_path is None
            ):
                self._set_feedback_submit_state("ERROR", "提交失败 · 尚未完成检测")
                self._showwarning(
                    "尚未检测",
                    "请先完成图片检测，再提交反馈。",
                    parent=self.root,
                )
                return
            detection_category = str(
                self.detection.get("category", self.category_var.get())
            )
            if detection_category != self.category_var.get():
                self._set_feedback_submit_state("ERROR", "提交失败 · 产品已切换")
                self._showwarning(
                    "产品已切换",
                    "当前检测结果属于 "
                    f"{detection_category}。请切回该产品后再提交反馈，"
                    "以保持模型版本和模型包绑定一致。",
                    parent=self.root,
                )
                return
            label = 1 if self.operator_label_var.get() == "异常" else 0
            dataset_role = (
                "validation"
                if self.dataset_role_var.get() == "独立验证"
                else "training"
            )
            if (
                label == 1
                and dataset_role == "training"
                and self.defect_region is None
            ):
                self._set_feedback_submit_state("ERROR", "提交失败 · 缺少缺陷区域")
                self._showwarning(
                    "缺少缺陷区域",
                    "训练用异常反馈必须先在图片上拖拽框选缺陷区域。",
                    parent=self.root,
                )
                return
            state = self.detection.get("feedback_runtime_state")
            state = state if isinstance(state, dict) else {}
            image_size = None
            if self.image_rgb is not None:
                image_size = (
                    int(self.image_rgb.shape[1]),
                    int(self.image_rgb.shape[0]),
                )
            submission = dict(
                image_bytes=self.image_bytes,
                original_name=self.image_path.name,
                category=detection_category,
                model_version=str(self.detection["model_version"]),
                result=self.detection["result"],
                operator_label=label,
                defect_type=self.defect_type_var.get(),
                note=self.note_text.get("1.0", "end").strip(),
                dataset_role=dataset_role,
                defect_region=self.defect_region,
                source_type=self.source_type,
                source_name=self.source_name,
                frame_index=(
                    self.video_frame_index if self.source_type == "video" else None
                ),
                image_size=image_size,
                run_id=str(self.detection["model_version"]),
                bundle_hash=state.get("bundle_hash"),
                adaptive_revision_id=state.get("revision_id"),
                head_version=state.get("head_version"),
                feature_version=state.get("feature_version"),
                threshold_version=state.get("threshold_version"),
                appearance_decision=bool(
                    self.detection["result"].get(
                        "model_decision", self.detection["result"]["decision"]
                    )
                ),
                overall_decision=bool(
                    self.detection["result"].get(
                        "overall_decision", self.detection["result"]["decision"]
                    )
                ),
                optimization_target="appearance",
            )
            self._set_feedback_submit_state("SUBMITTING", "正在提交反馈...")
            record = self.store.save_feedback(**submission)
        except DuplicateFeedbackError as error:
            self._resolve_duplicate_feedback(error)
        except Exception as error:
            LOGGER.exception("feedback submission failed")
            self._set_feedback_submit_state("ERROR", "提交失败 · 查看提示")
            self._showerror(
                "反馈提交失败",
                self._feedback_submission_error_text(error),
                parent=self.root,
            )
        else:
            self._feedback_persisted(record, replaced=False, changes={})
        finally:
            self._feedback_submit_inflight = False
            self.feedback_submit_button.configure(state="normal", text="提交反馈")

    def _set_feedback_submit_state(self, state: str, text: str) -> None:
        allowed = {
            "IDLE",
            "VALIDATING",
            "SUBMITTING",
            "EXISTS",
            "CONFIRM_REPLACE",
            "REPLACING",
            "SUCCESS",
            "REPLACED",
            "CANCELLED",
            "ERROR",
        }
        if state not in allowed:
            raise ValueError(f"unknown feedback submission state: {state}")
        self.feedback_submit_state_var.set(state)
        self.feedback_submit_status_var.set(f"提交状态：{text}")
        self.status_var.set(text)
        self.root.update_idletasks()

    def _resolve_duplicate_feedback(self, error: DuplicateFeedbackError) -> None:
        existing = error.existing_record
        proposed = error.proposed_record
        changes = self.store.feedback_changes(existing, proposed)
        feedback_id = str(existing.get("feedback_id", "-"))
        if not changes:
            self._set_feedback_submit_state(
                "EXISTS", f"反馈已存在 · {feedback_id}"
            )
            self._showinfo(
                "反馈已存在",
                "该图片在当前模型版本下已经存在完全相同的反馈记录。\n\n"
                f"记录 ID：{feedback_id}\n"
                f"{self._feedback_record_summary(existing)}\n\n"
                "无需重复提交。",
                parent=self.root,
            )
            return

        self._set_feedback_submit_state("CONFIRM_REPLACE", "等待确认替换")
        decision = self._askyesnocancel(
            "是否替换原反馈",
            "该图片已经存在反馈记录。\n\n"
            f"原反馈（{feedback_id}）：\n"
            f"{self._feedback_record_summary(existing)}\n\n"
            "本次反馈：\n"
            f"{self._feedback_record_summary(proposed)}\n\n"
            "变更内容：\n"
            f"{self._format_feedback_changes(changes)}\n\n"
            "是：替换原反馈\n否：保留原反馈\n取消：取消本次提交",
            parent=self.root,
        )
        if decision is None:
            self._set_feedback_submit_state("CANCELLED", "已取消反馈提交")
            self._showinfo(
                "已取消反馈提交",
                "本次提交已取消，原反馈记录保持不变。\n\n"
                f"记录 ID：{feedback_id}",
                parent=self.root,
            )
            return
        if decision is False:
            self._set_feedback_submit_state("CANCELLED", "已保留原反馈")
            self._showinfo(
                "已保留原反馈",
                "本次修改没有保存，原记录保持不变。\n\n"
                f"记录 ID：{feedback_id}",
                parent=self.root,
            )
            return

        self.feedback_submit_button.configure(text="替换中...")
        self._set_feedback_submit_state("REPLACING", "正在替换...")
        try:
            record = self.store.replace_feedback(
                feedback_id,
                proposed,
                expected_record_sha256=self.store.feedback_record_sha256(existing),
            )
        except Exception as replace_error:
            LOGGER.exception("feedback replacement failed")
            self._set_feedback_submit_state("ERROR", "提交失败 · 无法替换反馈")
            self._showerror(
                "无法替换反馈",
                self._feedback_submission_error_text(replace_error),
                parent=self.root,
            )
            return
        self._feedback_persisted(record, replaced=True, changes=changes)

    def _feedback_persisted(
        self,
        record: dict[str, object],
        *,
        replaced: bool,
        changes: dict[str, dict[str, object]],
    ) -> None:
        feedback_id = str(record["feedback_id"])
        self.defect_type_var.set("")
        self.note_text.delete("1.0", "end")
        self._invalidate_feedback_candidate()
        if replaced:
            self._set_feedback_submit_state(
                "REPLACED", f"✓ 已替换 · {feedback_id}"
            )
            self._showinfo(
                "反馈替换成功",
                "原反馈记录已使用本次内容安全更新。\n\n"
                f"记录 ID：{feedback_id}\n\n"
                f"{self._format_feedback_changes(changes)}",
                parent=self.root,
            )
        else:
            self._set_feedback_submit_state(
                "SUCCESS", f"✓ 已保存 · {feedback_id}"
            )
            self._showinfo(
                "反馈提交成功",
                "反馈已成功保存。\n\n"
                f"记录 ID：{feedback_id}\n"
                f"图片：{record.get('original_name', '-')}\n"
                f"{self._feedback_record_summary(record)}",
                parent=self.root,
            )
        try:
            self._refresh_feedback_records()
        except Exception as refresh_error:
            LOGGER.exception("feedback persisted but records refresh failed")
            self._set_feedback_submit_state(
                "ERROR", f"反馈已保存，但页面刷新失败 · {feedback_id}"
            )
            self._showwarning(
                "反馈已保存，但页面刷新失败",
                "反馈记录已成功写入，不会再次保存。\n\n"
                f"记录 ID：{feedback_id}\n"
                f"刷新错误：{refresh_error}\n\n"
                "请重新进入“反馈记录”页面查看。",
                parent=self.root,
            )

    @staticmethod
    def _feedback_record_summary(record: dict[str, object]) -> str:
        label_text = "异常" if int(record.get("operator_label", 0)) else "正常"
        role_text = (
            "独立验证"
            if record.get("dataset_role") == "validation"
            else "训练反馈"
        )
        defect_type = str(record.get("defect_type", "")).strip() or "未填写"
        return (
            f"人工判定：{label_text}\n"
            f"反馈用途：{role_text}\n"
            f"缺陷类型：{defect_type}"
        )

    @staticmethod
    def _format_feedback_changes(
        changes: dict[str, dict[str, object]],
    ) -> str:
        labels = {
            "operator_label": "人工状态",
            "dataset_role": "反馈用途",
            "defect_type": "缺陷类型",
            "note": "备注",
            "defect_region": "缺陷区域",
        }

        def display(field: str, value: object) -> str:
            if field == "operator_label":
                return "异常" if int(value or 0) else "正常"
            if field == "dataset_role":
                return "独立验证" if value == "validation" else "训练反馈"
            if value is None or value == "":
                return "未填写"
            return str(value)

        lines = []
        for field in labels:
            if field not in changes:
                continue
            change = changes[field]
            lines.append(
                f"{labels[field]}：{display(field, change['old'])} → "
                f"{display(field, change['new'])}"
            )
        technical_count = len(changes) - len(lines)
        if technical_count:
            lines.append(f"检测证据/版本绑定：{technical_count} 项变化")
        return "\n".join(lines) or "无关键字段变化"

    @staticmethod
    def _feedback_submission_error_text(error: Exception) -> str:
        message = str(error).strip() or error.__class__.__name__
        if isinstance(error, PermissionError):
            return (
                "该反馈已被候选模型、当前活动版本、独立验证或回滚证据引用。\n\n"
                "为保证训练集与独立验证集隔离，当前记录不能替换。\n"
                "请使用新的独立验证图片。\n\n"
                f"详情：{message}"
            )
        if isinstance(error, FileNotFoundError):
            return f"反馈保存失败：源图片或反馈记录文件不存在。\n\n详情：{message}"
        if error.__class__.__name__ == "JSONDecodeError":
            return f"反馈保存失败：反馈记录文件读取异常。\n\n详情：{message}"
        if isinstance(error, OSError):
            return f"反馈保存失败：文件写入失败或没有写入权限。\n\n详情：{message}"
        if "duplicate feedback content" in message:
            assigned = (
                "独立验证"
                if "assigned to validation" in message
                else "训练反馈"
            )
            return (
                "该图片内容已在当前模型版本中提交过，本次未重复保存。\n\n"
                f"已归属：{assigned}\n"
                "训练反馈与独立验证必须使用内容不重复的图片。\n"
                "请更换图片，或先在“反馈记录”中处理原记录。"
            )
        return (
            "反馈提交失败，系统未保存本次反馈。\n\n"
            f"原因：{message}\n\n"
            "请查看程序日志，并核对当前图片、反馈用途、缺陷区域和模型版本后重试。"
        )

    @staticmethod
    def _format_metrics(threshold: float, metrics: dict[str, object]) -> str:
        return f"阈值 {threshold:.4f}    精确率 {metrics['precision']:.1%}    召回率 {metrics['recall']:.1%}    F1 分数 {metrics['f1']:.1%}    假阳性率 {metrics['fpr']:.1%}"

    def _propose_legacy_threshold(self) -> None:
        if self.busy:
            return
        try:
            runtime, version, _device = self._runtime()
            base = float(runtime.calibration["continuous_fusion"]["fusion"]["threshold"])
            self.candidate = self.store.propose_threshold(
                self.category_var.get(),
                version,
                base_threshold=base,
                minimum_recall=float(self.minimum_recall_var.get()),
            )
        except Exception as error:
            self._showerror("无法生成候选阈值", str(error))
            return
        self.base_metrics_var.set(self._format_metrics(self.candidate["base_threshold"], self.candidate["base_metrics"]))
        self.candidate_metrics_var.set(self._format_metrics(self.candidate["candidate_threshold"], self.candidate["candidate_metrics"]))
        validation = self.candidate["validation"]
        if validation["passed"]:
            self.validation_metrics_var.set(
                "通过 · "
                + self._format_metrics(
                    self.candidate["candidate_threshold"], validation["candidate_metrics"]
                )
            )
        else:
            self.validation_metrics_var.set(f"未通过 · {validation['reason']}")
        if self.detection is not None:
            score = float(self.detection["result"]["fusion_score"])
            old_label = "异常" if score >= self.candidate["base_threshold"] else "正常"
            new_label = "异常" if score >= self.candidate["candidate_threshold"] else "正常"
            self.sample_comparison_var.set(
                f"分数 {score:.4f} · 优化前 {old_label} → 优化后 {new_label}"
            )
        else:
            self.sample_comparison_var.set("完成一次图片检测后，可显示当前样本的新旧判断")

    def _activate_legacy_threshold_candidate(self) -> None:
        if self.busy:
            return
        if self.candidate is None:
            self._showwarning("没有候选版本", "请先生成候选阈值。")
            return
        if not self._askyesno("启用候选阈值", "候选指标来自当前反馈集。确认已检查反馈标签和新旧指标，并启用该阈值吗？"):
            return
        try:
            self.store.activate(self.candidate)
        except ValueError as error:
            self._showerror("禁止启用", str(error))
            return
        self.detection = None
        self._reset_result()
        self._refresh_all()
        self._showinfo("启用成功", "后续桌面页面检测将使用新的反馈阈值。")

    def _rollback_legacy_threshold(self) -> None:
        if self.busy:
            return
        category = self.category_var.get()
        version = self._model_version(category)
        if self.store.active_threshold(category, version) is None:
            self._showinfo("无需回滚", "当前产品没有启用反馈阈值。")
            return
        if not self._askyesno("回滚阈值", "确认回滚当前产品的上一阈值版本吗？"):
            return
        self.store.rollback(category, version)
        self.detection = None
        self._reset_result()
        self._refresh_all()
        self._showinfo("回滚成功", "阈值版本已回滚。")

    def _start_feature_bank_update(self) -> None:
        if self.busy:
            return
        category = self.category_var.get()
        version = self._model_version(category)
        rows = [
            row for row in self.store.records(category, "training", version)
            if int(row["operator_label"]) == 1 and row.get("defect_region") is not None
        ]
        if not rows:
            self._showwarning("没有区域样本", "至少需要一条已框选缺陷区域的训练异常反馈。")
            return
        requested_device = self.device_var.get()
        self.busy = True
        self.status_var.set("正在提取反馈缺陷特征...")

        def work() -> None:
            try:
                runtime, model_version, _device = self._runtime(
                    category, requested_device
                )
                crops = []
                sample_ids = []
                for row in rows:
                    image = np.array(
                        Image.open(self.store.root / row["image"]).convert("RGB"), copy=True
                    )
                    left, top, right, bottom = row["defect_region"]
                    crop = image[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    crops.append(crop)
                    sample_ids.append(row["feedback_id"])
                if not crops:
                    raise ValueError("all annotated regions are empty")
                with torch.inference_mode():
                    features = runtime.dino._features(crops)
                    embeddings = torch.nn.functional.normalize(
                        features.mean(dim=1), dim=1
                    ).cpu().numpy()
                validation = self._validate_feature_bank_candidate(
                    runtime,
                    category,
                    model_version,
                    embeddings,
                )
                entry = self.store.save_feature_bank(
                    category,
                    model_version,
                    embeddings,
                    sample_ids,
                    validation=validation,
                )
                apply_active_feedback(
                    runtime,
                    self.store,
                    category,
                    model_version=model_version,
                )
                self._post_ui(self._feature_bank_complete, entry)
            except Exception as error:
                self._post_ui(self._task_failed, error)

        threading.Thread(target=work, daemon=True).start()

    def _validate_feature_bank_candidate(
        self,
        runtime: AdaptedRuntime,
        category: str,
        model_version: str,
        embeddings: np.ndarray,
    ) -> dict[str, object]:
        validation_rows = self.store.records(
            category, "validation", model_version
        )
        labels = np.asarray(
            [int(row["operator_label"]) for row in validation_rows],
            dtype=np.uint8,
        )
        if len(validation_rows) < 4 or set(labels.tolist()) != {0, 1}:
            raise ValueError(
                "特征库启用至少需要 4 条当前模型版本的独立验证反馈，"
                "并同时包含正常与异常标签。"
            )
        images = [
            np.array(
                Image.open(self.store.root / row["image"]).convert("RGB"),
                copy=True,
            )
            for row in validation_rows
        ]
        ids = [str(row["feedback_id"]) for row in validation_rows]
        active_state = apply_active_feedback(
            runtime,
            self.store,
            category,
            model_version=model_version,
        )
        threshold = active_state.threshold
        base_rows = runtime.predict_batch(images, ids)
        base_scores = np.asarray(
            [float(row["fusion_score"]) for row in base_rows], dtype=np.float64
        )
        base_threshold = (
            float(threshold)
            if threshold is not None
            else float(base_rows[0]["fusion_threshold"])
        )
        base_metrics = classification_metrics(labels, base_scores, base_threshold)

        candidate_version = f"candidate:{time.time_ns()}"
        try:
            set_runtime_feature_bank(
                runtime,
                embeddings,
                feature_version=candidate_version,
            )
            candidate_rows = runtime.predict_batch(images, ids)
        finally:
            runtime._feedback_feature_version = candidate_version
            apply_active_feedback(
                runtime,
                self.store,
                category,
                model_version=model_version,
            )
        candidate_scores = np.asarray(
            [float(row["fusion_score"]) for row in candidate_rows],
            dtype=np.float64,
        )
        candidate_metrics = classification_metrics(
            labels, candidate_scores, base_threshold
        )
        minimum_recall = float(self.minimum_recall_var.get())
        passed = bool(
            candidate_metrics["recall"] >= minimum_recall
            and candidate_metrics["f1"] >= base_metrics["f1"]
        )
        if not passed:
            raise ValueError(
                "候选特征库未通过独立验证：必须满足最低召回率，且 F1 分数不低于当前版本。"
            )
        return {
            "passed": True,
            "method": "independent_operator_feedback_feature_bank_comparison",
            "category": category,
            "model_version": model_version,
            "sample_ids": ids,
            "sample_count": len(ids),
            "normal_count": int(np.sum(labels == 0)),
            "defect_count": int(np.sum(labels == 1)),
            "minimum_recall": minimum_recall,
            "decision_threshold": base_threshold,
            "base_metrics": base_metrics,
            "candidate_metrics": candidate_metrics,
        }

    def _feature_bank_complete(self, entry: dict[str, object]) -> None:
        self.busy = False
        self.status_var.set("缺陷特征库更新完成")
        self._refresh_all()
        self._showinfo(
            "特征库已更新",
            f"版本：{entry['version']}\n缺陷样本：{entry['sample_count']}\n"
            "后续检测会将反馈特征并入 DINOv2 已知缺陷库。",
        )

    def _propose_revision(self) -> None:
        """Generate one auditable head/bank/threshold candidate in the background."""
        if self.busy:
            return
        if not self._require_ready_model("生成反馈候选"):
            return
        category = self.category_var.get()
        requested_device = self.device_var.get()
        self.busy = True
        self.status_var.set("正在重跑反馈样本并生成完整候选...")
        self.candidate_state_var.set("RUNNING")
        self.improvement_var.set("正在计算独立验证指标")

        def work() -> None:
            try:
                runtime, version, _device = self._runtime(category, requested_device)
                candidate = propose_adaptive_revision(
                    runtime,
                    self.store,
                    category,
                    run_id=version,
                    minimum_recall=float(self.minimum_recall_var.get()),
                )
                self._post_ui(self._revision_candidate_complete, candidate)
            except Exception as error:
                self._post_ui(self._task_failed, error)

        threading.Thread(target=work, daemon=True).start()

    def _revision_candidate_complete(self, candidate: dict[str, object]) -> None:
        self.busy = False
        self.candidate = candidate
        parent_threshold = float(candidate.get("parent_threshold", 0.0))
        threshold = float(candidate["candidate_threshold"])
        base_metrics = dict(candidate["base_metrics"])
        candidate_metrics = dict(candidate["candidate_metrics"])
        validation_metrics = dict(candidate["validation_metrics"])
        presentation = _feedback_validation_presentation(candidate)
        self.base_metrics_var.set(self._format_metrics(parent_threshold, base_metrics))
        self.candidate_metrics_var.set(self._format_metrics(threshold, candidate_metrics))
        self.validation_metrics_var.set(
            f"{presentation['status']} · "
            + self._format_metrics(threshold, validation_metrics)
        )
        self.improvement_var.set(str(presentation["improvement"]))
        self.candidate_state_var.set("READY FOR REVIEW")
        fingerprint = dict(candidate.get("parameter_fingerprint", {}))
        candidate_fingerprint = dict(fingerprint.get("candidate", {}))
        parameter_hash = str(candidate_fingerprint.get("overall_sha256", "-"))
        self.revision_detail_var.set(
            f"候选 {candidate.get('revision_id', '-')}；输出层 33 参数；哈希 {parameter_hash}；"
            f"反馈特征 {int(candidate.get('feature_count', 0))}；阈值 {threshold:.4f}；"
            f"独立验证召回率 {float(validation_metrics['recall']):.1%} / "
            f"F1 {float(validation_metrics['f1']):.1%}"
        )
        if self.detection is not None:
            score = float(self.detection["result"]["fusion_score"])
            old_label = "异常" if score >= parent_threshold else "正常"
            new_label = "异常" if score >= threshold else "正常"
            self.sample_comparison_var.set(
                f"分数 {score:.4f} · 当前版本 {old_label} → 候选版本 {new_label}"
            )
        else:
            self.sample_comparison_var.set("完成一次图片检测后，可显示当前样本的新旧判断")
        if presentation["mode"] == "relative_improvement":
            self.status_var.set("反馈候选已实现相对改善但未达目标门槛，等待审核启用")
        else:
            self.status_var.set("完整反馈候选已达到目标门槛，等待审核启用")

    def _activate_candidate(self) -> None:
        if self.busy:
            return
        if self.candidate is None:
            self._showwarning("没有候选版本", "请先生成完整反馈候选。")
            return
        presentation = _feedback_validation_presentation(self.candidate)
        if presentation["mode"] == "relative_improvement":
            confirmation = (
                "该候选在独立验证上确有提升，但仍未达到设定的目标召回率。\n\n"
                f"{presentation['improvement']}\n\n"
                "仍要整体启用输出层叠加层、DINO 缺陷特征库和融合阈值吗？"
            )
        else:
            confirmation = (
                "将整体启用输出层叠加层、DINO 缺陷特征库和融合阈值。"
                "确认已检查独立验证指标？"
            )
        if not self._askyesno(
            "启用完整反馈候选",
            confirmation,
        ):
            return
        category = self.category_var.get()
        version = self._model_version(category)
        try:
            entry = self.store.activate_revision(
                self.candidate,
                category=category,
                run_id=version,
                bundle_hash=str(self.candidate["bundle_hash"]),
            )
            for (cached_category, _device), (runtime, cached_version) in self.runtimes.items():
                if cached_category == category and cached_version == version:
                    apply_active_feedback(runtime, self.store, category, model_version=version)
        except Exception as error:
            self._showerror("禁止启用", str(error))
            return
        self.detection = None
        self._reset_result()
        self._refresh_all()
        self.candidate_state_var.set("ACTIVE")
        if presentation["mode"] == "relative_improvement":
            self.improvement_var.set("相对改善版本已通过审核并整体启用")
        else:
            self.improvement_var.set("目标门槛通过版本已审核并整体启用")
        self._showinfo(
            "启用成功",
            f"完整反馈版本已启用：{entry['revision_id']}\n"
            f"{presentation['status']}\n"
            "后续检测将同时使用新的输出层、特征库和阈值。",
        )

    def _rollback(self) -> None:
        if self.busy:
            return
        category = self.category_var.get()
        version = self._model_version(category)
        try:
            active = self.store.active_revision(category, version, verify=True)
        except Exception as error:
            self._showerror("无法读取当前反馈版本", str(error))
            return
        if active is None:
            self._showinfo("无需回滚", "当前产品没有启用完整反馈版本。")
            return
        if not self._askyesno("回滚完整版本", "确认整体回滚输出层、特征库和阈值到上一版本吗？"):
            return
        try:
            restored = self.store.rollback_revision(
                category, version, str(active["bundle_hash"])
            )
            for (cached_category, _device), (runtime, cached_version) in self.runtimes.items():
                if cached_category == category and cached_version == version:
                    apply_active_feedback(runtime, self.store, category, model_version=version)
        except Exception as error:
            self._showerror("回滚失败", str(error))
            return
        self.candidate = None
        self.detection = None
        self._reset_result()
        self._refresh_all()
        self._showinfo(
            "回滚成功",
            "已恢复到上一完整反馈版本。" if restored is not None else "已恢复模型冻结模型包。",
        )

    def _reset_result(self) -> None:
        self.result_var.set("WAITING")
        self.score_var.set("-")
        self.threshold_var.set("-")
        self.branch_var.set("-")
        self.margin_var.set("-")
        self.defect_types_result_var.set("-")

    def _records_selection_changed(self, _event: tk.Event | None = None) -> None:
        state = "normal" if self.records_tree.selection() else "disabled"
        self.delete_records_button.configure(state=state)

    def _records_hover(self, event: tk.Event) -> None:
        item = self.records_tree.identify_row(event.y)
        if item == self._records_hover_item:
            return
        self._records_hover_clear()
        if item:
            tags = tuple(tag for tag in self.records_tree.item(item, "tags") if tag != "hover")
            self.records_tree.item(item, tags=(*tags, "hover"))
            self._records_hover_item = item

    def _records_hover_clear(self, _event: tk.Event | None = None) -> None:
        item = self._records_hover_item
        if item and self.records_tree.exists(item):
            tags = tuple(tag for tag in self.records_tree.item(item, "tags") if tag != "hover")
            self.records_tree.item(item, tags=tags)
        self._records_hover_item = ""

    def _invalidate_feedback_candidate(self) -> None:
        self.candidate = None
        self.base_metrics_var.set("反馈记录已变更，请重新生成完整反馈候选")
        self.candidate_metrics_var.set("暂无数据")
        self.validation_metrics_var.set("尚未执行独立验证")
        self.sample_comparison_var.set("暂无数据")
        self.improvement_var.set("暂无数据")
        self.candidate_state_var.set("WAITING")
        self.revision_detail_var.set(
            "候选已失效；输出层、特征库和阈值将于下次生成时共同验证。"
        )

    def _delete_selected_records(self) -> str:
        if self.busy:
            self._showwarning("操作进行中", "请等待当前操作完成后再删除记录。")
            return "break"
        feedback_ids = list(self.records_tree.selection())
        if not feedback_ids:
            self._showinfo("未选择记录", "请先选择要删除的反馈记录。")
            return "break"
        category = self.category_var.get()
        if not self._askyesno(
            "删除反馈记录",
            f"确定删除 {category} 的 {len(feedback_ids)} 条反馈记录及其对应图片吗？\n\n"
            "此操作无法从界面恢复；已启用的阈值和缺陷特征库不会自动回滚。",
        ):
            return "break"
        try:
            summary = self.store.delete_feedback(
                feedback_ids,
                category=category,
                model_version=self._model_version(category),
            )
        except PermissionError as error:
            self._showwarning(
                "证据受版本保护",
                "这些反馈已被当前版本或其回滚链引用，不能删除。"
                "请先回滚到不引用它们的版本后再处理。\n\n"
                f"详情：{error}",
            )
            return "break"
        except (OSError, TypeError, ValueError) as error:
            self._showerror("删除失败", str(error))
            return "break"
        self._invalidate_feedback_candidate()
        self._refresh_all()
        deleted = int(summary["deleted_records"])
        failed = len(summary["failed_images"])
        self.status_var.set(f"已删除 {deleted} 条反馈记录")
        if failed:
            failed_paths = "\n".join(
                str(item["path"]) for item in summary["failed_images"][:3]
            )
            self._showwarning(
                "记录已删除",
                f"已删除 {deleted} 条记录，但有 {failed} 个图片文件无法清理。\n"
                f"请关闭占用文件的程序后手动删除：\n{failed_paths}",
            )
        else:
            self._showinfo(
                "删除完成",
                f"已删除 {deleted} 条记录和 {summary['deleted_images']} 个图片文件。",
            )
        return "break"

    def _clear_current_category_records(self) -> None:
        if self.busy:
            self._showwarning("操作进行中", "请等待当前操作完成后再清空记录。")
            return
        category = self.category_var.get()
        version = self._model_version(category)
        rows = self.store.records(category, model_version=version)
        if not rows:
            self._showinfo("没有记录", f"{category} 当前没有反馈记录。")
            return
        if not self._askyesno(
            "清空当前产品记录",
            f"确定清空 {category} 的全部 {len(rows)} 条反馈记录及其对应图片吗？\n\n"
            "其他产品不受影响；此操作无法从界面恢复；已启用的阈值和缺陷特征库不会自动回滚。",
        ):
            return
        try:
            summary = self.store.clear_feedback(
                category=category,
                model_version=version,
            )
        except PermissionError as error:
            self._showwarning(
                "证据受版本保护",
                "当前产品含有被完整反馈版本或回滚链引用的记录，不能清空。"
                "请先回滚到不引用这些记录的版本后再处理。\n\n"
                f"详情：{error}",
            )
            return
        except (OSError, TypeError, ValueError) as error:
            self._showerror("清空失败", str(error))
            return
        self._invalidate_feedback_candidate()
        self._refresh_all()
        deleted = int(summary["deleted_records"])
        failed = len(summary["failed_images"])
        self.status_var.set(f"已清空 {category} 的 {deleted} 条反馈记录")
        if failed:
            failed_paths = "\n".join(
                str(item["path"]) for item in summary["failed_images"][:3]
            )
            self._showwarning(
                "记录已清空",
                f"已删除 {deleted} 条记录，但有 {failed} 个图片文件无法清理。\n"
                f"请关闭占用文件的程序后手动删除：\n{failed_paths}",
            )
        else:
            self._showinfo(
                "清空完成",
                f"已删除 {deleted} 条记录和 {summary['deleted_images']} 个图片文件。",
            )

    def _refresh_feedback_records(self) -> None:
        """Refresh only feedback counters and the records Treeview."""
        category = self.category_var.get()
        version = self._model_version(category)
        rows = self.store.records(category, model_version=version)
        optimization_rows = [
            row
            for row in rows
            if str(row.get("optimization_target", "appearance")) != "audit_only"
        ]
        training_rows = [
            row for row in optimization_rows if row["dataset_role"] == "training"
        ]
        validation_rows = [
            row for row in optimization_rows if row["dataset_role"] == "validation"
        ]
        normal = sum(int(row["operator_label"]) == 0 for row in training_rows)
        self.total_var.set(str(len(rows)))
        self.normal_var.set(str(normal))
        self.defect_var.set(str(len(training_rows) - normal))
        self.validation_count_var.set(str(len(validation_rows)))

        model_versions = sorted(
            {str(row.get("model_version", "-")) for row in rows}, reverse=True
        )
        self.record_model_combo.configure(values=("全部版本", *model_versions))
        if (
            self.record_model_filter_var.get() != "全部版本"
            and self.record_model_filter_var.get() not in model_versions
        ):
            self.record_model_filter_var.set("全部版本")
            return
        type_names = {
            "confirmed": "正确",
            "false_positive": "误检",
            "false_negative": "漏检",
        }
        query = self.record_search_var.get().strip().casefold()
        role_filter = self.record_role_filter_var.get()
        source_filter = self.record_source_filter_var.get()
        type_filter = self.record_type_filter_var.get()
        model_filter = self.record_model_filter_var.get()
        display_rows: list[dict[str, object]] = []
        for row in rows:
            role_text = "验证" if row.get("dataset_role") == "validation" else "训练"
            source_text = "视频" if row.get("source_type") == "video" else "图片"
            type_text = type_names.get(
                str(row.get("feedback_type")), str(row.get("feedback_type"))
            )
            searchable = " ".join(
                str(row.get(key, ""))
                for key in (
                    "created_at",
                    "source_name",
                    "feedback_type",
                    "defect_type",
                    "model_version",
                    "note",
                )
            ).casefold()
            if query and query not in searchable:
                continue
            if role_filter != "全部角色" and role_filter != role_text:
                continue
            if source_filter != "全部来源" and source_filter != source_text:
                continue
            if type_filter != "全部结果" and type_filter != type_text:
                continue
            if model_filter != "全部版本" and model_filter != str(
                row.get("model_version")
            ):
                continue
            display_rows.append(row)
        self.records_count_var.set(f"{len(display_rows)} / {len(rows)} 条记录")
        if display_rows:
            self.records_empty_label.place_forget()
        else:
            self.records_empty_label.place(relx=0.5, rely=0.5, anchor="center")
        self._records_hover_item = ""
        for item in self.records_tree.get_children():
            self.records_tree.delete(item)
        self.delete_records_button.configure(state="disabled")
        self.clear_records_button.configure(state="normal" if rows else "disabled")
        for row_index, row in enumerate(reversed(display_rows)):
            self.records_tree.insert(
                "",
                "end",
                iid=str(row["feedback_id"]),
                values=(
                    row["created_at"],
                    "验证" if row["dataset_role"] == "validation" else "训练",
                    (
                        f"视频帧 {row['frame_index']}"
                        if row["source_type"] == "video"
                        else "图片"
                    ),
                    type_names.get(row["feedback_type"], row["feedback_type"]),
                    "异常" if row["operator_label"] else "正常",
                    f"{row['fusion_score']:.4f}",
                    f"{row['fusion_threshold']:.4f}",
                    str(row.get("defect_region") or "-"),
                    row["defect_type"],
                    row["model_version"],
                ),
                tags=("odd" if row_index % 2 else "even",),
            )

    def _refresh_all(self) -> None:
        category = self.category_var.get()
        if not self._has_ready_model(category):
            pending = (
                isinstance(self.pointer.get("models"), dict)
                and str(self.pointer.get("models", {}).get(category, "")).strip()
                == "pending"
            )
            self.header_model_var.set(
                f"{category} PENDING" if pending else "NO PRODUCT"
            )
            self.feedback_status_var.set(
                "等待完成 100+30 适配" if pending else "请新建或加载产品任务"
            )
            self.active_revision_status_var.set("WAITING")
            self.feature_bank_var.set("完整版本：未启用")
            self.revision_detail_var.set(
                "完成产品适配后，反馈候选、独立验证、启用与回滚才会开放。"
            )
            self.candidate_state_var.set("WAITING")
            self._refresh_feedback_records()
            return
        version = self._model_version(category)
        rows = self.store.records(category, model_version=version)
        self.header_model_var.set(f"{category} Ready")
        optimization_rows = [
            row
            for row in rows
            if str(row.get("optimization_target", "appearance")) != "audit_only"
        ]
        training_rows = [
            row for row in optimization_rows if row["dataset_role"] == "training"
        ]
        validation_rows = [
            row for row in optimization_rows if row["dataset_role"] == "validation"
        ]
        normal = sum(int(row["operator_label"]) == 0 for row in training_rows)
        self.total_var.set(str(len(rows)))
        self.normal_var.set(str(normal))
        self.defect_var.set(str(len(training_rows) - normal))
        self.validation_count_var.set(str(len(validation_rows)))
        revision_error: Exception | None = None
        try:
            active_revision = self.store.active_revision(category, version, verify=True)
        except Exception as error:
            active_revision = None
            revision_error = error
        if active_revision is None:
            if revision_error is not None:
                self.active_revision_status_var.set("ERROR")
                self.feedback_status_var.set(f"反馈版本校验失败：{revision_error}")
                self.feature_bank_var.set("完整版本：校验失败，未加载")
                self.revision_detail_var.set(
                    "反馈版本未加载；请检查候选收据、工件和证据绑定后再继续。"
                )
            else:
                self.active_revision_status_var.set("WAITING")
                self.feedback_status_var.set("当前使用模型冻结模型包")
                self.feature_bank_var.set("完整版本：未启用")
                self.revision_detail_var.set(
                    "输出层参数：原始模型包；反馈特征：0；反馈阈值：原始模型包"
                )
        else:
            self.active_revision_status_var.set("ACTIVE")
            payload = active_revision.get("payload", {})
            feature = payload.get("feature_bank", {}) if isinstance(payload, dict) else {}
            fingerprint = payload.get("head_parameter_fingerprint", {}) if isinstance(payload, dict) else {}
            candidate_fingerprint = fingerprint.get("candidate", {}) if isinstance(fingerprint, dict) else {}
            revision_id = str(active_revision.get("revision_id", "-"))
            threshold = float(active_revision.get("threshold", 0.0))
            count = int(feature.get("sample_count", 0)) if isinstance(feature, dict) else 0
            parameter_hash = str(candidate_fingerprint.get("overall_sha256", "-")) if isinstance(candidate_fingerprint, dict) else "-"
            self.feedback_status_var.set(f"完整反馈版本已启用：{revision_id}")
            self.feature_bank_var.set(
                f"完整版本：{revision_id} · 缺陷特征 {count} · 阈值 {threshold:.4f}"
            )
            self.revision_detail_var.set(
                f"输出层 33 参数；哈希 {parameter_hash}；反馈特征 {count}；阈值 {threshold:.4f}"
            )
        model_versions = sorted(
            {str(row.get("model_version", "-")) for row in rows}, reverse=True
        )
        self.record_model_combo.configure(values=("全部版本", *model_versions))
        if (
            self.record_model_filter_var.get() != "全部版本"
            and self.record_model_filter_var.get() not in model_versions
        ):
            self.record_model_filter_var.set("全部版本")
            return
        type_names = {"confirmed": "正确", "false_positive": "误检", "false_negative": "漏检"}
        query = self.record_search_var.get().strip().casefold()
        role_filter = self.record_role_filter_var.get()
        source_filter = self.record_source_filter_var.get()
        type_filter = self.record_type_filter_var.get()
        model_filter = self.record_model_filter_var.get()
        display_rows: list[dict[str, object]] = []
        for row in rows:
            role_text = "验证" if row.get("dataset_role") == "validation" else "训练"
            source_text = "视频" if row.get("source_type") == "video" else "图片"
            type_text = type_names.get(str(row.get("feedback_type")), str(row.get("feedback_type")))
            searchable = " ".join(
                str(row.get(key, ""))
                for key in (
                    "created_at", "source_name", "feedback_type", "defect_type",
                    "model_version", "note",
                )
            ).casefold()
            if query and query not in searchable:
                continue
            if role_filter != "全部角色" and role_filter != role_text:
                continue
            if source_filter != "全部来源" and source_filter != source_text:
                continue
            if type_filter != "全部结果" and type_filter != type_text:
                continue
            if model_filter != "全部版本" and model_filter != str(row.get("model_version")):
                continue
            display_rows.append(row)
        self.records_count_var.set(f"{len(display_rows)} / {len(rows)} 条记录")
        if display_rows:
            self.records_empty_label.place_forget()
        else:
            self.records_empty_label.place(relx=0.5, rely=0.5, anchor="center")
        self._records_hover_item = ""
        for item in self.records_tree.get_children():
            self.records_tree.delete(item)
        self.delete_records_button.configure(state="disabled")
        self.clear_records_button.configure(
            state="normal" if rows else "disabled"
        )
        for row_index, row in enumerate(reversed(display_rows)):
            self.records_tree.insert("", "end", iid=str(row["feedback_id"]), values=(
                row["created_at"], "验证" if row["dataset_role"] == "validation" else "训练",
                (f"视频帧 {row['frame_index']}" if row["source_type"] == "video" else "图片"),
                type_names.get(row["feedback_type"], row["feedback_type"]),
                "异常" if row["operator_label"] else "正常", f"{row['fusion_score']:.4f}",
                f"{row['fusion_threshold']:.4f}", str(row.get("defect_region") or "-"),
                row["defect_type"], row["model_version"],
            ), tags=("odd" if row_index % 2 else "even",))


def main() -> int:
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    try:
        FeedbackApp(root)
    except Exception as error:
        root.withdraw()
        messagebox.showerror("AOI 用户反馈优化台", str(error), parent=root)
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
