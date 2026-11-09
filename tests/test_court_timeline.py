"""The timeline seeks exact markers and keeps its zoom independent of decoding."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from court_annotation.timeline import LANES, CourtTimeline, frame_at_x, x_at_frame


@pytest.fixture
def timeline(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    widget = CourtTimeline()
    widget.resize(800, 145)
    widget.set_data(18001, 30, 9000, [9030], [9060], [9090])
    widget.show()
    app.processEvents()
    yield app, widget
    widget.close()


def test_zoom_and_exact_marker_click(timeline):
    app, widget = timeline
    seeks = []
    widget.seekRequested.connect(seeks.append)
    visits = []
    widget.savedRequested.connect(lambda frame: visits.append((frame, 'saved')))
    widget.candidateRequested.connect(lambda frame: visits.append((frame, 'candidate')))
    widget.markerRequested.connect(lambda frame: visits.append((frame, 'marker')))
    widget.zoom_in.click()
    app.processEvents()
    assert (widget.start, widget.end) == (4500, 13500)
    assert not seeks
    for index, (frame, (lane, _label, _colour)) in enumerate(zip((9030, 9060, 9090), LANES, strict=True)):
        point = QPoint(round(x_at_frame(frame, widget.canvas.width(), widget.start, widget.end)), lane)
        QTest.mousePress(widget.canvas, Qt.MouseButton.LeftButton, pos=point)
        assert len(visits) == index
        QTest.mouseRelease(widget.canvas, Qt.MouseButton.LeftButton, pos=point)
        assert visits[-1] == (frame, ('saved', 'candidate', 'marker')[index])
    assert not seeks
    QTest.mouseClick(widget.canvas, Qt.MouseButton.LeftButton, pos=QPoint(400, 20))
    assert seeks == [frame_at_x(400, widget.canvas.width(), widget.start, widget.end)]
    widget.setEnabled(False)
    QTest.mouseClick(widget.canvas, Qt.MouseButton.LeftButton, pos=point)
    assert len(visits) == 3
    assert len(seeks) == 1


def test_zoom_tracks_navigation_and_full_range(timeline):
    _app, widget = timeline
    widget.zoom(0.5)
    widget.set_data(18001, 30, 17000, [9030], [9060], [9090])
    assert widget.start <= 17000 <= widget.end
    assert widget.end - widget.start == 9000
    widget.whole_video.click()
    assert (widget.start, widget.end) == (0, 18000)
    assert frame_at_x(-100, 800, 0, 18000) == 0
    assert frame_at_x(1000, 800, 0, 18000) == 18000


def test_single_frame_video(timeline):
    _app, widget = timeline
    widget.set_data(1, 30, 0, [0], [], [])
    widget.zoom(0.5)
    assert (widget.start, widget.end) == (0, 0)
    assert widget.target_at(200, 20, 800) == (0, None)
