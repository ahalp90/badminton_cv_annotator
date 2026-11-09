"""Zoomable frame navigation with separate annotation and camera-change lanes."""
from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

LABEL_WIDTH = 126
RIGHT_MARGIN = 14
LANES = ((40, "Saved courts", "#56B4E9"), (58, "Suspected", "#B7BDC5"), (76, "Camera changed", "#E69F00"))
# Wong palette colours; each kind also has its own labelled lane.
PERIOD_LANES = {"play": (94, "Play", "#009E73"), "jitter": (112, "Jitter", "#D55E00"),
                "transition": (130, "Transition", "#CC79A7")}
LANE_HALF_HEIGHT = 9


def frame_at_x(x: float, width: int, start: int, end: int) -> int:
    fraction = (x - LABEL_WIDTH) / max(1, width - LABEL_WIDTH - RIGHT_MARGIN)
    return round(start + min(1.0, max(0.0, fraction)) * (end - start))


def x_at_frame(frame: int, width: int, start: int, end: int) -> float:
    return LABEL_WIDTH + (frame - start) / max(1, end - start) * max(1, width - LABEL_WIDTH - RIGHT_MARGIN)


def time_label(seconds: float) -> str:
    minutes, remainder = divmod(max(0, seconds), 60)
    return f"{int(minutes):02d}:{int(remainder):02d}"


class CourtTimeline(QWidget):
    """Seek on release; zoom around the current frame without decoding on drag."""

    seekRequested = Signal(int)
    savedRequested = Signal(int)
    candidateRequested = Signal(int)
    markerRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.total = 1
        self.fps = 1.0
        self.current = 0
        self.start, self.end = 0, 0
        self.points: tuple[tuple[int, ...], ...] = ((), (), ())
        self.periods: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = QHBoxLayout()
        self.zoom_in = QPushButton("Zoom in")
        self.zoom_out = QPushButton("Zoom out")
        self.whole_video = QPushButton("Whole video")
        for button in (self.zoom_in, self.zoom_out, self.whole_video):
            toolbar.addWidget(button)
        self.zoom_in.clicked.connect(lambda: self.zoom(0.5))
        self.zoom_out.clicked.connect(lambda: self.zoom(2.0))
        self.whole_video.clicked.connect(self.fit)
        self.range_label = QLabel()
        toolbar.addWidget(self.range_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.canvas = TimelineCanvas(self)
        layout.addWidget(self.canvas)
        self.setToolTip("Click a marker to visit its exact frame. Click the ruler to seek. Wheel to zoom around the current frame.")

    def set_data(self, total: int, fps: float, current: int, annotations: Sequence[int],
                 candidates: Sequence[int], markers: Sequence[int], periods: Sequence[dict] = ()) -> None:
        changed_video = total != self.total or fps != self.fps
        self.total, self.fps, self.current = total, fps, current
        self.points = (tuple(annotations), tuple(candidates), tuple(markers))
        self.periods = list(periods)
        if changed_video:
            self.start, self.end = 0, total - 1
        elif not self.start <= current <= self.end:
            self.set_span(self.end - self.start)
        self.refresh()

    def set_span(self, span: int) -> None:
        span = min(self.total - 1, max(1, span))
        self.start = max(0, min(self.total - 1 - span, self.current - span // 2))
        self.end = self.start + span

    def zoom(self, factor: float) -> None:
        minimum = max(1, round(self.fps * 2))
        self.set_span(max(minimum, round((self.end - self.start) * factor)))
        self.refresh()

    def fit(self) -> None:
        self.start, self.end = 0, self.total - 1
        self.refresh()

    def refresh(self) -> None:
        self.range_label.setText(f"{time_label(self.start / self.fps)} – {time_label(self.end / self.fps)}")
        self.zoom_in.setEnabled(self.end - self.start > round(self.fps * 2))
        self.zoom_out.setEnabled(self.end - self.start < self.total - 1)
        self.canvas.update()

    def target_at(self, x: float, y: float, width: int) -> tuple[int, int | None] | None:
        if x < LABEL_WIDTH or x > width - RIGHT_MARGIN:
            return None
        for lane, ((lane_y, _label, _colour), points) in enumerate(zip(LANES, self.points, strict=True)):
            if abs(y - lane_y) <= LANE_HALF_HEIGHT:
                visible = [frame for frame in points if self.start <= frame <= self.end]
                if visible:
                    closest = min(visible, key=lambda frame: abs(x_at_frame(frame, width, self.start, self.end) - x))
                    if abs(x_at_frame(closest, width, self.start, self.end) - x) <= 8:
                        return closest, lane
        return frame_at_x(x, width, self.start, self.end), None


class TimelineCanvas(QWidget):
    def __init__(self, timeline: CourtTimeline) -> None:
        super().__init__(timeline)
        self.timeline = timeline
        self.setMinimumWidth(400)
        self.setFixedHeight(142)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event: object) -> None:
        del event
        timeline = self.timeline
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#20252b"))
        if not self.isEnabled():
            painter.setOpacity(0.4)
        width = self.width()
        text_colour = QColor("#E3E8ED")
        duration = max(1 / timeline.fps, (timeline.end - timeline.start) / timeline.fps)
        raw_step = duration / 5
        magnitude = 10 ** math.floor(math.log10(raw_step))
        step = next(value * magnitude for value in (1, 2, 5, 10) if value * magnitude >= raw_step)
        tick = math.ceil(timeline.start / timeline.fps / step) * step
        while tick <= timeline.end / timeline.fps:
            x = x_at_frame(round(tick * timeline.fps), width, timeline.start, timeline.end)
            painter.setPen(QPen(QColor("#46505a"), 1))
            painter.drawLine(QPointF(x, 25), QPointF(x, self.height() - 7))
            label = time_label(tick) if step >= 1 else f"{tick:.1f}s"
            painter.setPen(text_colour)
            text_width = painter.fontMetrics().horizontalAdvance(label)
            painter.drawText(round(min(width - text_width - 3, max(LABEL_WIDTH, x - text_width / 2))), 18, label)
            tick += step
        for index, ((y, label, colour), points) in enumerate(zip(LANES, timeline.points, strict=True)):
            painter.setPen(text_colour)
            painter.drawText(7, y + 4, label)
            painter.setPen(QColor("#46505a"))
            painter.drawLine(LABEL_WIDTH, y, width - RIGHT_MARGIN, y)
            painter.setPen(QPen(QColor(colour), 2))
            painter.setBrush(QColor(colour))
            for frame in points:
                if not timeline.start <= frame <= timeline.end:
                    continue
                x = x_at_frame(frame, width, timeline.start, timeline.end)
                if index == 0:
                    painter.drawRect(round(x) - 3, y - 4, 6, 8)
                elif index == 1:
                    painter.drawEllipse(QPointF(x, y), 3, 3)
                else:
                    painter.drawPolygon(QPolygonF([QPointF(x, y - 5), QPointF(x - 4, y + 4), QPointF(x + 4, y + 4)]))
        for y, label, _colour in PERIOD_LANES.values():
            painter.setPen(text_colour)
            painter.drawText(7, y + 4, label)
            painter.setPen(QColor("#46505a"))
            painter.drawLine(LABEL_WIDTH, y, width - RIGHT_MARGIN, y)
        painter.setPen(Qt.PenStyle.NoPen)
        for period in timeline.periods:
            y, _label, colour = PERIOD_LANES[period["kind"]]
            # An open period runs to the displayed frame so its extent is visible while scrubbing.
            end = period["end"] if period["end"] is not None else max(period["start"], timeline.current)
            if end < timeline.start or period["start"] > timeline.end:
                continue
            left = x_at_frame(max(period["start"], timeline.start), width, timeline.start, timeline.end)
            right = x_at_frame(min(end, timeline.end), width, timeline.start, timeline.end)
            painter.setOpacity(0.45 if period["end"] is None else 1.0)
            painter.setBrush(QColor(colour))
            painter.drawRect(round(left), y - 4, max(3, round(right - left)), 8)
        painter.setOpacity(1.0 if self.isEnabled() else 0.4)
        if timeline.start <= timeline.current <= timeline.end:
            x = x_at_frame(timeline.current, width, timeline.start, timeline.end)
            painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
            painter.drawLine(QPointF(x, 26), QPointF(x, self.height() - 4))
        painter.end()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            target = self.timeline.target_at(event.position().x(), event.position().y(), self.width())
            if target is not None:
                frame, lane = target
                signals = (self.timeline.savedRequested, self.timeline.candidateRequested, self.timeline.markerRequested)
                signal = self.timeline.seekRequested if lane is None else signals[lane]
                signal.emit(frame)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.isEnabled() and event.angleDelta().y():
            self.timeline.zoom(0.5 if event.angleDelta().y() > 0 else 2.0)
            event.accept()
