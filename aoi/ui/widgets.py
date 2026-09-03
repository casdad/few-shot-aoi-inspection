"""Reusable Tkinter presentation components for competition demonstrations.

These widgets are intentionally read-only with respect to business state.  They
render values supplied by the existing application and invoke only navigation
callbacks provided by the caller.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Iterable, Sequence
from tkinter import ttk

from .theme import (
    APP_BG,
    BORDER,
    ERROR_BG,
    ERROR_TEXT,
    PRIMARY,
    PRIMARY_HOVER,
    PRIMARY_LIGHT,
    SUCCESS_BG,
    SUCCESS_TEXT,
    SURFACE,
    SURFACE_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
)


COMPETITION_STEPS = (
    ("adapt", "① 少样本适配", "100 正常 + 30 缺陷"),
    ("detect", "② 在线检测", "图片 / 视频"),
    ("feedback", "③ 人工反馈", "误检 / 漏检复核"),
    ("optimize", "④ 反馈优化", "候选模型 / 独立验证 / 可回滚"),
    ("acceptance", "⑤ 性能测试", "GPU 验收 / CPU < 2s"),
)


class Tooltip:
    """A restrained text tooltip whose content can be resolved on demand."""

    def __init__(
        self,
        widget: tk.Misc,
        text_provider: Callable[[], str],
        *,
        delay_ms: int = 450,
    ) -> None:
        self.widget = widget
        self.text_provider = text_provider
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        text = self.text_provider().strip()
        if not text or not self.widget.winfo_exists():
            return
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.attributes("-topmost", True)
        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 16
        window.wm_geometry(f"+{x}+{y}")
        tk.Label(
            window,
            text=text,
            justify="left",
            wraplength=720,
            bg=TEXT_PRIMARY,
            fg=SURFACE,
            font=("Microsoft YaHei UI", 9),
            padx=10,
            pady=7,
            relief="solid",
            borderwidth=1,
        ).pack()
        self._window = window

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
            self._window = None


def elide_middle(value: str, *, max_chars: int = 72) -> str:
    """Shorten long display text while retaining both context and filename."""

    text = str(value)
    if len(text) <= max_chars:
        return text
    left = max(12, max_chars // 3)
    right = max(20, max_chars - left - 3)
    return f"{text[:left]}...{text[-right:]}"


def elide_multiline(value: str, *, max_chars: int = 84) -> str:
    return "\n".join(elide_middle(line, max_chars=max_chars) for line in str(value).splitlines())


class ElidedTextLabel(ttk.Label):
    """Label that abbreviates long text but exposes the full value on hover."""

    def __init__(
        self,
        parent: tk.Misc,
        source: tk.StringVar,
        *,
        max_chars: int = 84,
        **kwargs: object,
    ) -> None:
        self._source = source
        self._max_chars = max_chars
        self._display = tk.StringVar(master=parent)
        super().__init__(parent, textvariable=self._display, **kwargs)
        self._source.trace_add("write", self._sync)
        Tooltip(self, lambda: self._source.get())
        self._sync()

    def _sync(self, *_args: object) -> None:
        try:
            self._display.set(
                elide_multiline(self._source.get(), max_chars=self._max_chars)
            )
        except tk.TclError:
            pass


def bind_path_entry(entry: ttk.Entry, source: tk.StringVar) -> None:
    """Keep a full path in an Entry while displaying its informative tail."""

    Tooltip(entry, lambda: source.get())

    def show_tail(*_args: object) -> None:
        def move_to_tail() -> None:
            try:
                entry.xview_moveto(1.0)
            except tk.TclError:
                pass

        try:
            entry.after_idle(move_to_tail)
            entry.after(80, move_to_tail)
            entry.after(240, move_to_tail)
        except tk.TclError:
            pass

    source.trace_add("write", show_tail)
    entry.bind("<Map>", show_tail, add="+")
    entry.bind("<FocusOut>", show_tail, add="+")
    show_tail()


class ScrollablePage(ttk.Frame):
    """Width-responsive page wrapper with an on-demand vertical scrollbar."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            self,
            background=APP_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.body = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.body, anchor="nw"
        )
        self._sync_pending = False
        self.body.bind("<Configure>", self._schedule_sync, add="+")
        self.canvas.bind("<Configure>", self._schedule_sync, add="+")
        self.canvas.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.body.bind("<MouseWheel>", self._on_mousewheel, add="+")

    def _schedule_sync(self, _event: tk.Event | None = None) -> None:
        if self._sync_pending:
            return
        self._sync_pending = True
        self.after_idle(self._sync_layout)

    def _sync_layout(self) -> None:
        self._sync_pending = False
        try:
            canvas_width = max(1, self.canvas.winfo_width())
            canvas_height = max(1, self.canvas.winfo_height())
            requested_height = max(1, self.body.winfo_reqheight())
            content_height = max(canvas_height, requested_height)
            self.canvas.itemconfigure(
                self._window_id, width=canvas_width, height=content_height
            )
            self.canvas.configure(
                scrollregion=(0, 0, canvas_width, content_height)
            )
            if requested_height > canvas_height + 2:
                self.scrollbar.grid()
            else:
                self.scrollbar.grid_remove()
                self.canvas.yview_moveto(0.0)
        except tk.TclError:
            return

    def _on_mousewheel(self, event: tk.Event) -> str:
        if self.scrollbar.winfo_ismapped():
            delta = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(delta * 3, "units")
            return "break"
        return ""

    def bind_mousewheel_tree(self) -> None:
        """Route wheel input from descendants to this local scroll viewport."""

        bindtag = f"scroll-panel-{id(self)}"
        self.bind_class(bindtag, "<MouseWheel>", self._on_mousewheel)

        def attach(widget: tk.Misc) -> None:
            tags = widget.bindtags()
            if bindtag not in tags:
                widget.bindtags((bindtag, *tags))
            for child in widget.winfo_children():
                attach(child)

        attach(self.canvas)
        attach(self.body)


class AdaptivePage(ttk.Frame):
    """A page whose body receives exactly the Notebook's remaining viewport."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.body = ttk.Frame(self)
        self.body.grid(row=0, column=0, sticky="nsew")


class VerticalScrollPanel(ScrollablePage):
    """A deliberately local vertical overflow panel, never a whole page."""


class CompetitionHint(tk.Label):
    """One-line competition positioning statement below a page title."""

    def __init__(self, parent: tk.Misc, text: str) -> None:
        super().__init__(
            parent,
            text=text,
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        )


class CompetitionFlowBar(tk.Frame):
    """Compact five-stage AOI loop; active/completed state is caller supplied."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        active_stage: str | None,
        on_navigate: Callable[[str], None] | None = None,
        completed_stages: Iterable[str] = (),
    ) -> None:
        super().__init__(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=10,
            pady=7,
            height=64,
        )
        self.grid_propagate(False)
        self.rowconfigure(0, weight=1)
        self._active_stage = active_stage
        self._completed_stages = set(completed_stages)
        self._on_navigate = on_navigate
        self._nodes: dict[str, tk.Label] = {}
        self._title_label = tk.Label(
            self,
            text="自学习\nAOI 闭环",
            bg=SURFACE,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 9, "bold"),
            justify="left",
        )
        self._title_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self._arrows: list[tk.Label] = []
        for index, (key, title, detail) in enumerate(COMPETITION_STEPS):
            node_column = index * 2 + 1
            self.columnconfigure(node_column, weight=1, uniform="competition_flow")
            node = tk.Label(
                self,
                text=f"{title}\n{detail}",
                justify="center",
                font=("Microsoft YaHei UI", 8),
                padx=5,
                pady=4,
                cursor="hand2" if on_navigate is not None else "arrow",
                relief="flat",
                borderwidth=0,
                highlightthickness=1,
            )
            node.grid(row=0, column=node_column, sticky="ew")
            if on_navigate is not None:
                node.bind(
                    "<Button-1>",
                    lambda _event, stage=key: self._on_navigate(stage),
                )
            self._nodes[key] = node
            if index < len(COMPETITION_STEPS) - 1:
                arrow = tk.Label(
                    self,
                    text="→",
                    bg=SURFACE,
                    fg=TEXT_TERTIARY,
                    font=("Microsoft YaHei UI", 9, "bold"),
                    padx=3,
                )
                arrow.grid(row=0, column=node_column + 1)
                self._arrows.append(arrow)
        self._refresh()

    def set_compact_height(self, compact: bool) -> None:
        """Reallocate vertical space without scaling text or changing content."""

        self.configure(
            height=56 if compact else 64,
            padx=8 if compact else 10,
            pady=4 if compact else 7,
        )
        self._title_label.grid_configure(padx=(0, 8 if compact else 10))
        for node in self._nodes.values():
            node.configure(pady=2 if compact else 4)
        for arrow in self._arrows:
            arrow.configure(padx=2 if compact else 3)

    def set_state(
        self,
        *,
        active_stage: str | None,
        completed_stages: Iterable[str] = (),
    ) -> None:
        self._active_stage = active_stage
        self._completed_stages = set(completed_stages)
        self._refresh()

    def _refresh(self) -> None:
        for key, node in self._nodes.items():
            if key in self._completed_stages:
                background, foreground, border = (
                    SUCCESS_BG,
                    SUCCESS_TEXT,
                    SUCCESS_TEXT,
                )
            elif key == self._active_stage:
                background, foreground, border = (
                    PRIMARY_LIGHT,
                    PRIMARY_HOVER,
                    PRIMARY,
                )
            else:
                background, foreground, border = (
                    SURFACE_SUBTLE,
                    TEXT_SECONDARY,
                    BORDER,
                )
            node.configure(
                bg=background,
                fg=foreground,
                highlightbackground=border,
                highlightcolor=border,
            )


class CompactProcessStrip(tk.Frame):
    """Small status-aware workflow used by the feedback optimization page."""

    def __init__(self, parent: tk.Misc, steps: Sequence[str]) -> None:
        super().__init__(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=8,
        )
        self._nodes: list[tk.Label] = []
        for index, step in enumerate(steps):
            node_column = index * 2
            self.columnconfigure(node_column, weight=1, uniform="process_strip")
            node = tk.Label(
                self,
                text=step,
                bg=SURFACE_SUBTLE,
                fg=TEXT_SECONDARY,
                font=("Microsoft YaHei UI", 9, "bold"),
                padx=9,
                pady=6,
                highlightthickness=1,
                highlightbackground=BORDER,
            )
            node.grid(row=0, column=node_column, sticky="ew")
            self._nodes.append(node)
            if index < len(steps) - 1:
                tk.Label(
                    self,
                    text="→",
                    bg=SURFACE,
                    fg=TEXT_TERTIARY,
                    font=("Microsoft YaHei UI", 9, "bold"),
                    padx=6,
                ).grid(row=0, column=node_column + 1)

    def set_state(
        self,
        *,
        active_index: int | None,
        completed_indexes: Iterable[int] = (),
        error_index: int | None = None,
    ) -> None:
        completed = set(completed_indexes)
        for index, node in enumerate(self._nodes):
            if index == error_index:
                background, foreground, border = ERROR_BG, ERROR_TEXT, ERROR_TEXT
            elif index in completed:
                background, foreground, border = (
                    SUCCESS_BG,
                    SUCCESS_TEXT,
                    SUCCESS_TEXT,
                )
            elif index == active_index:
                background, foreground, border = (
                    PRIMARY_LIGHT,
                    PRIMARY_HOVER,
                    PRIMARY,
                )
            else:
                background, foreground, border = (
                    SURFACE_SUBTLE,
                    TEXT_SECONDARY,
                    BORDER,
                )
            node.configure(
                bg=background,
                fg=foreground,
                highlightbackground=border,
            )


class MetricTargetCard(tk.Frame):
    """Compact title/value list for fixed targets or measured backend values."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        rows: Sequence[tuple[str, str | tk.Variable]],
        accent: bool = False,
    ) -> None:
        background = PRIMARY_LIGHT if accent else SURFACE_SUBTLE
        border = PRIMARY if accent else BORDER
        super().__init__(
            parent,
            bg=background,
            highlightthickness=1,
            highlightbackground=border,
            padx=10,
            pady=7,
        )
        tk.Label(
            self,
            text=title,
            bg=background,
            fg=PRIMARY_HOVER if accent else TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 3))
        for index, (name, value) in enumerate(rows):
            row = index // 3 + 1
            pair = index % 3
            name_column = pair * 2
            value_column = name_column + 1
            tk.Label(
                self,
                text=f"{name}：",
                bg=background,
                fg=TEXT_SECONDARY,
                font=("Microsoft YaHei UI", 8),
            ).grid(row=row, column=name_column, sticky="w", padx=(0, 2), pady=1)
            value_kwargs: dict[str, object] = {"text": value}
            if isinstance(value, tk.Variable):
                value_kwargs = {"textvariable": value}
            tk.Label(
                self,
                bg=background,
                fg=TEXT_PRIMARY,
                font=("Microsoft YaHei UI", 8, "bold"),
                **value_kwargs,
            ).grid(row=row, column=value_column, sticky="w", padx=(0, 10), pady=1)


__all__ = [
    "AdaptivePage",
    "COMPETITION_STEPS",
    "CompactProcessStrip",
    "CompetitionFlowBar",
    "CompetitionHint",
    "ElidedTextLabel",
    "MetricTargetCard",
    "ScrollablePage",
    "Tooltip",
    "VerticalScrollPanel",
    "bind_path_entry",
    "elide_middle",
    "elide_multiline",
]
