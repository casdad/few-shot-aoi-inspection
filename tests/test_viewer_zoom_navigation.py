from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from feedback_ui import FeedbackApp


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


def _viewer_app() -> FeedbackApp:
    app = FeedbackApp.__new__(FeedbackApp)
    app.busy = False
    app.source_type = "image"
    app.folder_images = []
    app.image_index = -1
    app.viewer_fit_mode = True
    app.viewer_zoom = 1.0
    app.viewer_zoom_text_var = FakeVar("适应")
    app.viewer_file_var = FakeVar()
    app.viewer_meta_var = FakeVar()
    app.status_var = FakeVar()
    app._reset_result = lambda: None
    app._render_preview = lambda: None
    app._update_navigation_buttons = lambda: None
    return app


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (32, 24), color).save(path)
    return path


def test_previous_next_preserves_manual_zoom(tmp_path) -> None:
    first = _image(tmp_path / "first.png", (255, 0, 0))
    second = _image(tmp_path / "second.png", (0, 255, 0))
    app = _viewer_app()
    app.folder_images = [first, second]
    app.image_index = 0
    app._load_image(first)
    app._zoom_viewer(0.15)

    app._navigate_image(1)

    assert app.image_path == second
    assert app.image_index == 1
    assert app.viewer_fit_mode is False
    assert app.viewer_zoom == 1.15
    assert app.viewer_zoom_text_var.get() == "115%"


def test_previous_next_preserves_fit_mode(tmp_path) -> None:
    first = _image(tmp_path / "first.png", (255, 0, 0))
    second = _image(tmp_path / "second.png", (0, 255, 0))
    app = _viewer_app()
    app.folder_images = [first, second]
    app.image_index = 0
    app._load_image(first)

    app._navigate_image(1)

    assert app.viewer_fit_mode is True
    assert app.viewer_zoom == 1.0
    assert app.viewer_zoom_text_var.get() == "适应"


class FakeCapture:
    def __init__(self) -> None:
        self.index = 0

    def set(self, _property: int, value: int) -> None:
        self.index = int(value)

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.full((24, 32, 3), self.index * 40, dtype=np.uint8)
        return True, frame


def test_previous_next_video_frame_preserves_manual_zoom() -> None:
    app = _viewer_app()
    app.source_type = "video"
    app.video_capture = FakeCapture()
    app.video_frame_count = 3
    app.video_frame_index = 0
    app.video_path = Path("sample.mp4")
    app._load_video_frame(0)
    app._zoom_viewer(0.15)

    app._navigate_image(1)

    assert app.video_frame_index == 1
    assert app.viewer_fit_mode is False
    assert app.viewer_zoom == 1.15
    assert app.viewer_zoom_text_var.get() == "115%"
