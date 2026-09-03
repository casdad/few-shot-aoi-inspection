from __future__ import annotations

from feedback_ui import FeedbackApp


def test_competition_flow_routes_to_content_pages() -> None:
    app = FeedbackApp.__new__(FeedbackApp)
    app.adapt_tab = object()
    app.detect_tab = object()
    app.optimize_tab = object()
    app.acceptance_tab = object()
    selected: list[object] = []
    app._switch_tab = selected.append

    expected = {
        "adapt": app.adapt_tab,
        "detect": app.detect_tab,
        "feedback": app.detect_tab,
        "optimize": app.optimize_tab,
        "acceptance": app.acceptance_tab,
    }
    for stage, tab in expected.items():
        app._navigate_competition_stage(stage)
        assert selected.pop() is tab


def test_unknown_competition_stage_does_not_switch_page() -> None:
    app = FeedbackApp.__new__(FeedbackApp)
    app.adapt_tab = object()
    app.detect_tab = object()
    app.optimize_tab = object()
    app.acceptance_tab = object()
    selected: list[object] = []
    app._switch_tab = selected.append

    app._navigate_competition_stage("unknown")

    assert selected == []
