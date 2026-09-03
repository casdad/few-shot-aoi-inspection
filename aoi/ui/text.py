"""Chinese display mappings for backend values used by the desktop UI.

The functions in this module never mutate a backend value.  They only convert
it immediately before presentation, keeping persisted enum/key values intact.
"""

from __future__ import annotations

from typing import Literal


StatusTone = Literal["waiting", "running", "success", "warning", "error", "neutral"]

STATUS_TEXT = {
    "WAITING": "等待中",
    "READY": "就绪",
    "RUNNING": "运行中",
    "PROCESSING": "处理中",
    "WARNING": "警告",
    "ERROR": "错误",
    "PASS": "已通过",
    "PASSED": "已通过",
    "FAIL": "未通过",
    "FAILED": "未通过",
    "SUCCESS": "成功",
    "NORMAL": "正常",
    "DEFECT": "缺陷",
    "CURRENT": "当前版本",
    "CANDIDATE": "候选版本",
    "ACTIVE": "已启用",
    "COMPLETE": "已完成",
    "DIAGNOSTIC": "诊断结果",
    "ATTENTION": "需关注",
    "MODEL LOADED": "已加载",
    "DEVICE DETECTING": "检测中",
    "UNAVAILABLE": "不可用",
    "READY FOR REVIEW": "待审核",
    "NOT CONFIGURED": "未启用",
    "NG": "异常",
    "ANOMALY": "异常",
    "REVIEW": "待复核",
    "AUTO": "自动",
    "UNKNOWN": "等待测试",
    "FAIL_HARDWARE_MISMATCH": "未通过 · 硬件不匹配",
    "FAIL_RESOLUTION_MISMATCH": "未通过 · 分辨率不匹配",
    "FAIL_SAMPLE_PROTOCOL": "未通过 · 样本协议不满足",
    "FAIL_GPU_NOT_EXCLUSIVE": "未通过 · GPU 非独占",
    "FAIL_P50_LATENCY": "未通过 · P50 延迟超标",
}

QUALITY_STATUS_TEXT = {
    "WAITING": "等待检测",
    "READY": "等待检测",
    "RUNNING": "检测中",
    "PROCESSING": "检测中",
    "WARNING": "警告",
    "ERROR": "异常",
    "PASS": "正常",
    "PASSED": "正常",
    "NORMAL": "正常",
    "NG": "异常",
    "FAIL": "异常",
    "FAILED": "异常",
    "DEFECT": "异常",
    "NOT CONFIGURED": "未启用",
    "DISABLED": "未启用",
    "REVIEW": "异常",
}

UI_TEXT = {
    "DETECTION STATUS": "检测状态",
    "ACCEPTANCE STATUS": "验收状态",
    "P50 LATENCY": "P50 延迟",
    "GPU PREFLIGHT": "GPU 预检",
    "RESOLUTION": "输入分辨率",
    "Hardware Match": "硬件匹配",
    "Resolution Match": "分辨率匹配",
    "Latency Target": "延迟达标",
    "GPU Available": "GPU 可用",
    "Accuracy": "准确率",
    "Recall": "召回率",
    "F1": "F1 分数",
    "FPR": "假阳性率",
    "Gate": "验收门槛",
    "NORMAL · 100": "正常样本 · 100",
    "DEFECT · 30": "缺陷样本 · 30",
    "MASK · 30": "像素掩码 · 30",
    "CURRENT": "当前版本",
    "CANDIDATE": "候选版本",
    "AUTO": "自动",
}


def display_text(value: object) -> str:
    """Return a user-facing Chinese label without changing the source value."""

    text = str(value)
    return UI_TEXT.get(text, text.replace("bundle", "模型包").replace("Bundle", "模型包"))


def _split_demo(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.split("·", maxsplit=1)]
    if len(parts) == 2 and parts[1].upper() == "DEMO":
        return parts[0], " · 演示"
    return value.strip(), ""


def display_gate(value: object) -> str:
    """Translate acceptance gate values while preserving their internal key."""

    raw = str(value).strip()
    upper = raw.upper()
    if upper.startswith("GO_STAGE2"):
        return "第二阶段已通过"
    if upper.startswith("GO_STAGE1"):
        return "第一阶段已通过"
    if upper.startswith("GO_") or upper == "GO":
        return "已通过"
    if upper.startswith("NO_GO"):
        return "未通过"
    return display_status(raw)


def display_status(value: object, *, context: str = "general") -> str:
    """Translate an enum-like status for display only."""

    raw = str(value).strip()
    if not raw:
        return "等待中"
    base, suffix = _split_demo(raw)
    upper = base.upper()
    if upper.startswith(("GO_", "NO_GO")):
        return f"{display_gate(base)}{suffix}"
    mapping = QUALITY_STATUS_TEXT if context == "quality" else STATUS_TEXT
    translated = mapping.get(upper)
    if translated is not None:
        return f"{translated}{suffix}"
    if upper.endswith(" READY"):
        return f"{base[:-6].strip()} 就绪{suffix}"
    if " · " in raw:
        head, tail = raw.split(" · ", maxsplit=1)
        translated_head = mapping.get(head.upper(), STATUS_TEXT.get(head.upper(), head))
        return f"{translated_head} · {display_text(tail)}"
    return display_text(raw)


def status_tone(value: object, *, context: str = "general") -> StatusTone:
    """Map a raw status to a restrained semantic color role."""

    upper = str(value).strip().upper()
    head = upper.split(" · ", maxsplit=1)[0]
    if upper.startswith("NO_GO"):
        return "error"
    if upper.startswith("GO_"):
        return "success"
    if head in {"FAIL", "FAILED", "ERROR", "NG", "DEFECT"} or head.startswith(
        ("FAIL_", "ERROR_")
    ):
        return "error"
    if context == "quality" and head == "REVIEW":
        return "error"
    if head in {"WARNING", "DIAGNOSTIC", "ATTENTION", "READY FOR REVIEW"} or head.startswith("WARN_"):
        return "warning"
    if head in {"RUNNING", "PROCESSING", "DEVICE DETECTING"}:
        return "running"
    if head in {"PASS", "PASSED", "SUCCESS", "READY", "ACTIVE", "COMPLETE", "NORMAL"}:
        return "success"
    if head.endswith(" READY"):
        return "success"
    if head in {"WAITING", "NOT CONFIGURED", "DISABLED", "UNKNOWN"}:
        return "waiting"
    if context == "quality":
        return "waiting"
    return "neutral"


def display_boolean(value: object) -> str:
    """Present a tri-state backend validation result without inventing data."""

    if value is True:
        return "是"
    if value is False:
        return "否"
    return "等待测试"
