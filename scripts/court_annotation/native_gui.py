"""Single-window Qt workspace using direct OpenCV seeking and the court state machine."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np
from court_annotation import court_landmarks as court
from court_annotation.annotate_court_corners_offframe import (
    ActionKind,
    CaptureState,
    OffframeSession,
    SessionAction,
    clock_label,
    load_prefill,
    previous_court,
    read_saved_courts,
)
from court_annotation.image_helpers import (
    CORNER_COLOUR,
    draw_outline,
    loupe_origin,
    render_loupe,
)
from court_annotation.review_queue import PERIOD_LABELS, ReviewQueue
from court_annotation.timeline import PERIOD_LANES, CourtTimeline
from PySide6 import QtCore, QtGui, QtWidgets

# Shift-clicking a one-frame button, or Shift+Left/Right, steps this many frames.
BIG_FRAME_STEP = 10
# Gap between a browsing zoom inset and the video edge, in logical pixels.
INSET_MARGIN = 8
# Overlay choice that follows the displayed frame; other choices are fixed saved frames.
OVERLAY_PREVIOUS = 'previous'


def point_label(name: str) -> str:
    row, column = name.split("_x_")
    row_labels = {"far_baseline": "Far baseline", "far_long_service": "Far doubles service line",
                  "far_short_service": "Far short service line", "near_short_service": "Near short service line",
                  "near_long_service": "Near doubles service line", "near_baseline": "Near baseline"}
    column_labels = {"left_doubles": "left doubles sideline", "left_singles": "left singles sideline",
                     "centre": "centre line", "right_singles": "right singles sideline",
                     "right_doubles": "right doubles sideline"}
    return f"{row_labels[row]} · {column_labels[column]}"


@dataclass(frozen=True)
class ImagePlacement:
    """Actual displayed image rectangle, in Qt logical pixels."""

    source_width: int
    source_height: int
    width: int
    height: int
    left: int
    top: int

    def source_point(self, x: float, y: float) -> tuple[float, float] | None:
        if not (self.left <= x < self.left + self.width and self.top <= y < self.top + self.height):
            return None
        return ((x - self.left) * self.source_width / self.width,
                (y - self.top) * self.source_height / self.height)


def image_placement(source_width: int, source_height: int, width: int, height: int) -> ImagePlacement:
    scale = min(max(1, width) / source_width, max(1, height) / source_height)
    fitted_width, fitted_height = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    return ImagePlacement(source_width, source_height, fitted_width, fitted_height,
                          (width - fitted_width) // 2, (height - fitted_height) // 2)


def to_qimage(image: np.ndarray) -> QtGui.QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                        QtGui.QImage.Format.Format_RGB888).copy()


def add_key_hint(button: QtWidgets.QPushButton, hint: str) -> None:
    """Show a small muted modifier hint after the label, matching the rally labeller's buttons."""
    label = QtWidgets.QLabel(f'{button.text()} <span style="color: gray;">{hint}</span>')
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    button.setAccessibleName(button.text())
    button.setText('')
    layout = QtWidgets.QHBoxLayout(button)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.addWidget(label)
    button.setMinimumWidth(label.sizeHint().width() + 12)


class ImageCanvas(QtWidgets.QWidget):
    clicked = QtCore.Signal(float, float)
    rightPressed = QtCore.Signal(float, float)
    # Left click inside the zoom inset, as fractions of the inset's width and height.
    insetClicked = QtCore.Signal(float, float)
    panStarted = QtCore.Signal()
    panned = QtCore.Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.pan_anchor: QtCore.QPointF | None = None
        self.image = QtGui.QImage()
        self.inset = QtGui.QImage()
        self.inset_on_right = False
        self.inset_rect = QtCore.QRect()
        self.placement: ImagePlacement | None = None
        self.setMinimumSize(100, 100)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def set_image(self, image: np.ndarray | None, inset: np.ndarray | None = None, inset_on_right: bool = False) -> None:
        if image is None:
            self.image = QtGui.QImage()
            self.placement = None
        else:
            self.image = to_qimage(image)
        self.inset = QtGui.QImage() if inset is None else to_qimage(inset)
        self.inset_on_right = inset_on_right
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#171c22'))
        self.inset_rect = QtCore.QRect()
        if not self.image.isNull():
            self.placement = image_placement(self.image.width(), self.image.height(), self.width(), self.height())
            box = self.placement
            painter.drawImage(QtCore.QRect(box.left, box.top, box.width, box.height), self.image)
            if not self.inset.isNull():
                side = min(self.inset.width(), box.width // 2 - INSET_MARGIN, box.height - 2 * INSET_MARGIN)
                left = box.left + box.width - side - INSET_MARGIN if self.inset_on_right else box.left + INSET_MARGIN
                self.inset_rect = QtCore.QRect(left, box.top + INSET_MARGIN, side, side)
                painter.drawImage(self.inset_rect, self.inset)
                painter.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), 2))
                painter.drawRect(self.inset_rect)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.setFocus()
        # The inset hides the video beneath it, so clicks there must never map to full-video coordinates.
        if not self.inset.isNull() and self.inset_rect.contains(event.position().toPoint()):
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                box = self.inset_rect
                self.insetClicked.emit((event.position().x() - box.left()) / box.width(),
                                       (event.position().y() - box.top()) / box.height())
            return
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.pan_anchor = event.position()
            self.panStarted.emit()
            self.emit_right(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.placement is not None:
            point = self.placement.source_point(event.position().x(), event.position().y())
            if point is not None:
                self.clicked.emit(*point)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.pan_anchor is not None and self.placement is not None:
            delta = event.position() - self.pan_anchor
            self.panned.emit(delta.x() * self.placement.source_width / self.placement.width,
                             delta.y() * self.placement.source_height / self.placement.height)
            self.emit_right(event)

    def emit_right(self, event: QtGui.QMouseEvent) -> None:
        if not self.inset.isNull() and self.inset_rect.contains(event.position().toPoint()):
            return
        if self.placement is not None:
            point = self.placement.source_point(event.position().x(), event.position().y())
            if point is not None:
                self.rightPressed.emit(*point)

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:
        self.pan_anchor = None


class CourtPicker(QtWidgets.QWidget):
    selected = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(220, 320)
        self.current: str | None = None
        self.assigned: set[str] = set()
        self.setToolTip('Click a court intersection to select its identity. Filled circles are confirmed; the diamond is selected.')

    def diagram_point(self, x: float, y: float) -> QtCore.QPointF:
        scale = min((self.width() - 40) / court.COURT_WIDTH_M, (self.height() - 40) / court.COURT_LENGTH_M)
        return QtCore.QPointF((self.width() - court.COURT_WIDTH_M * scale) / 2 + x * scale, 20 + y * scale)

    def point_positions(self) -> dict[str, QtCore.QPointF]:
        return {name: self.diagram_point(*point) for name, point in court.LANDMARKS.items()}

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#17232c'))
        painter.setPen(QtGui.QPen(QtGui.QColor('#dbe5ec'), 1.4))
        for x in court.X_LINES.values():
            if x == court.X_LINES['centre']:
                for lower, upper in ((0, court.Y_LINES['far_short_service']),
                                     (court.Y_LINES['near_short_service'], court.COURT_LENGTH_M)):
                    painter.drawLine(self.diagram_point(x, lower), self.diagram_point(x, upper))
            else:
                painter.drawLine(self.diagram_point(x, 0), self.diagram_point(x, court.COURT_LENGTH_M))
        for y in court.Y_LINES.values():
            painter.drawLine(self.diagram_point(0, y), self.diagram_point(court.COURT_WIDTH_M, y))
        painter.setPen(QtGui.QPen(QtGui.QColor('#8c9caa'), 1, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(self.diagram_point(0, court.NET_Y_M), self.diagram_point(court.COURT_WIDTH_M, court.NET_Y_M))
        for name, point in self.point_positions().items():
            painter.setPen(QtGui.QPen(QtGui.QColor('#07121c'), 1))
            painter.setBrush(QtGui.QColor('#45ddff' if name in self.assigned else '#627582'))
            painter.drawEllipse(point, 4 if name in self.assigned else 2, 4 if name in self.assigned else 2)
            if name == self.current:
                painter.setPen(QtGui.QPen(QtGui.QColor('#ff1eb4'), 2))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawPolygon(QtGui.QPolygonF([point + QtCore.QPointF(0, -7), point + QtCore.QPointF(7, 0),
                                                    point + QtCore.QPointF(0, 7), point + QtCore.QPointF(-7, 0)]))
        painter.setPen(QtGui.QColor('#dbe5ec'))
        painter.drawText(QtCore.QRect(0, 0, self.width(), 18), QtCore.Qt.AlignmentFlag.AlignCenter, 'Far end')
        painter.drawText(QtCore.QRect(0, self.height() - 18, self.width(), 18), QtCore.Qt.AlignmentFlag.AlignCenter, 'Near end')

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            name, point = min(self.point_positions().items(),
                              key=lambda item: (item[1] - event.position()).manhattanLength())
            if (point - event.position()).manhattanLength() <= 14:
                self.selected.emit(name)


class CourtWorkspace(QtWidgets.QMainWindow):
    """Keep review decisions separate from capture state and decoded frame selection."""

    def __init__(self, capture: cv2.VideoCapture, session: OffframeSession, review: ReviewQueue,
                 total: int, fps: float, start_frame: int, nudge_step: float,
                 videos: Sequence[str] = (), video_index: int = 0) -> None:
        super().__init__()
        self.capture, self.session, self.review = capture, session, review
        # Picker labels for every video in the batch; the caller opens chosen_video after close.
        self.videos = list(videos) or [session.video_basename]
        self.video_index = video_index
        self.chosen_video: int | None = None
        self.total, self.fps = total, fps
        self.nudge_step = nudge_step
        self.frame_index = -1
        self.frame: np.ndarray | None = None
        self.queue_active = True
        self.aim_started = 0.0
        self.loupe_origin: tuple[int, int] | None = None
        # Zoom crop origin shown as an inset over the video while browsing; kept while stepping frames.
        self.browse_zoom: tuple[int, int] | None = None
        self.pan_origin: tuple[int, int] | None = None
        self.annotations = read_saved_courts(session.csv_path, session.video_basename)
        self.setWindowTitle(f'Court annotation — {session.video_basename}')
        self.resize(1320, 740)
        self.setMinimumSize(960, 620)
        self.build_widgets()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.refresh_lists()
        self.load(start_frame)
        self.video.setFocus()

    @property
    def browsing(self) -> bool:
        return self.session.state == CaptureState.SCRUB

    def button(self, layout: QtWidgets.QLayout, text: str, callback: object) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def build_widgets(self) -> None:
        centre = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(centre)
        self.setCentralWidget(centre)
        header = QtWidgets.QHBoxLayout()
        # A menu button rather than a dropdown: arrow keys and the mouse wheel cannot switch videos.
        self.video_picker = QtWidgets.QPushButton(self.session.video_basename)
        self.video_picker.setAccessibleName('Video')
        self.video_picker.setToolTip('Choose another video. Saved work stays saved.')
        self.video_picker.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        video_menu = QtWidgets.QMenu(self.video_picker)
        current_marker = QtGui.QActionGroup(video_menu)
        for index, label in enumerate(self.videos):
            action = video_menu.addAction(label)
            current_marker.addAction(action)
            action.setCheckable(True)
            action.setChecked(index == self.video_index)
            action.triggered.connect(lambda _checked=False, index=index: self.choose_video(index))
        self.video_picker.setMenu(video_menu)
        header.addWidget(self.video_picker)
        header.addStretch()
        self.position = QtWidgets.QLabel()
        header.addWidget(self.position)
        self.next_video_button = self.button(header, "Next video →", self.next_video)
        self.next_video_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.next_video_button.setVisible(len(self.videos) > 1)
        layout.addLayout(header)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        left, right = QtWidgets.QWidget(), QtWidgets.QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([820, 480])
        left_layout, right_layout = QtWidgets.QVBoxLayout(left), QtWidgets.QVBoxLayout(right)
        left_layout.setContentsMargins(0, 0, 4, 0)
        right_layout.setContentsMargins(4, 0, 0, 0)
        self.video = ImageCanvas()
        self.video.clicked.connect(self.click_video)
        self.video.rightPressed.connect(self.look_at)
        self.video.insetClicked.connect(self.click_inset)
        self.video.setToolTip('Left click marks a court point. Right click or right-drag zooms there without marking; '
                              'Escape closes the zoom.')
        left_layout.addWidget(self.video, 1)
        nav = QtWidgets.QHBoxLayout()
        left_layout.addLayout(nav)
        self.back = self.button(nav, '−1 frame', lambda: self.step_frames(-1))
        self.forward = self.button(nav, '+1 frame', lambda: self.step_frames(1))
        for button, sign in ((self.back, '−'), (self.forward, '+')):
            button.setToolTip(f'Shift-click (or Shift+arrow) to step {BIG_FRAME_STEP} frames.')
            add_key_hint(button, f'⇧{sign}{BIG_FRAME_STEP}')
        nav.addSpacing(8)
        self.back_second = self.button(nav, '−1 sec', lambda: self.load(self.frame_index - round(self.fps)))
        self.forward_second = self.button(nav, '+1 sec', lambda: self.load(self.frame_index + round(self.fps)))
        nav.addSpacing(8)
        nav.addWidget(QtWidgets.QLabel('Frame'))
        self.jump_entry = QtWidgets.QLineEdit()
        self.jump_entry.setAccessibleName('Frame number')
        self.jump_entry.setToolTip('Enter a frame number and press Enter. Values outside the video go to its first or last frame.')
        nav.addWidget(self.jump_entry)
        self.jump_entry.returnPressed.connect(self.jump_to_frame)
        self.timeline = CourtTimeline()
        self.timeline.seekRequested.connect(self.load)
        self.timeline.candidateRequested.connect(self.inspect_candidate)
        self.timeline.savedRequested.connect(self.inspect_saved)
        self.timeline.markerRequested.connect(self.inspect_marker)
        left_layout.addWidget(self.timeline)
        capture_bar = QtWidgets.QHBoxLayout()
        left_layout.addLayout(capture_bar)
        self.mark_button = self.button(capture_bar, 'Mark court', lambda: self.act(self.session.begin_capture(self.frame_index)))
        self.copy_button = self.button(capture_bar, 'Copy saved court', self.copy_court)
        self.overlay = QtWidgets.QComboBox()
        self.overlay.setAccessibleName('Court overlay')
        self.overlay.setToolTip('Draw a saved court over the displayed frame to check whether it still matches.')
        # Mouse-only, so keyboard focus passed on from Mark court cannot land here.
        self.overlay.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.overlay.currentIndexChanged.connect(self.draw)
        capture_bar.addWidget(self.overlay)
        capture_bar.addStretch()
        self.prior_label = QtWidgets.QLabel()
        left_layout.addWidget(self.prior_label)
        self.prompt = QtWidgets.QLabel()
        self.prompt.setWordWrap(True)
        left_layout.addWidget(self.prompt)
        self.editor = QtWidgets.QGroupBox('Refine court point')
        editor_layout = QtWidgets.QVBoxLayout(self.editor)
        self.zoom = ImageCanvas()
        self.zoom.setFixedSize(320, 320)
        self.zoom.clicked.connect(self.click_zoom)
        self.zoom.panStarted.connect(self.start_pan)
        self.zoom.panned.connect(self.pan_zoom)
        self.zoom.setToolTip('Left click to refine; drag with the right button to pan. Arrow keys nudge the point.')
        previews = QtWidgets.QHBoxLayout()
        previews.addWidget(self.zoom)
        self.diagram = CourtPicker()
        self.diagram.selected.connect(self.choose_diagram)
        previews.addWidget(self.diagram)
        editor_layout.addLayout(previews)
        edit_controls = QtWidgets.QVBoxLayout()
        editor_layout.addLayout(edit_controls)
        editor_layout.addStretch()
        self.landmark = QtWidgets.QComboBox()
        self.landmark.addItem("Choose which court point this is…")
        self.landmark.addItems([point_label(name) for name in court.LANDMARK_NAMES])
        self.landmark.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.landmark.setMinimumContentsLength(24)
        self.landmark.currentIndexChanged.connect(self.choose_landmark)
        self.landmark.activated.connect(lambda _index: QtCore.QTimer.singleShot(0, self.video.setFocus))
        edit_controls.addWidget(self.landmark)
        confirm_row = QtWidgets.QHBoxLayout()
        edit_controls.addLayout(confirm_row)
        self.confirm_button = self.button(confirm_row, 'Confirm point', self.confirm)
        self.save_button = self.button(confirm_row, 'Save court', lambda: self.act(self.session.commit()))
        repair_row = QtWidgets.QHBoxLayout()
        edit_controls.addLayout(repair_row)
        self.undo_button = self.button(repair_row, 'Undo point', lambda: self.act(self.session.undo_last()))
        self.cancel_button = self.button(repair_row, 'Cancel court', lambda: self.act(self.session.abort()))
        self.fit_button = self.button(repair_row, 'Fit court', lambda: self.act(self.session.do_fit()))
        right_layout.addWidget(self.editor, 1)
        self.review_panel = QtWidgets.QWidget()
        right_layout.addWidget(self.review_panel, 1)
        right_layout = QtWidgets.QVBoxLayout(self.review_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QtWidgets.QLabel('Saved courts · click a row to inspect'))
        self.saved_tree = self.make_tree(('Time', 'Frame', 'TL', 'TR', 'BR', 'BL'))
        self.saved_tree.itemSelectionChanged.connect(self.select_saved)
        right_layout.addWidget(self.saved_tree, 1)
        right_layout.addWidget(QtWidgets.QLabel('Suspected camera movement'))
        self.queue_tree = self.make_tree(('Time', 'Decision', 'Reason'))
        self.queue_tree.itemSelectionChanged.connect(self.select_candidate)
        right_layout.addWidget(self.queue_tree, 2)
        queue_nav = QtWidgets.QHBoxLayout()
        right_layout.addLayout(queue_nav)
        self.previous_button = self.button(queue_nav, 'Previous', lambda: self.move_queue(-1))
        self.next_button = self.button(queue_nav, 'Next pending', lambda: self.move_queue(1))
        self.marker_next = self.button(queue_nav, 'Next confirmed', self.next_marker)
        self.marker_next.setToolTip('Visit each saved camera-change marker, including standalone markers.')
        self.selection = QtWidgets.QLabel()
        self.selection.setWordWrap(True)
        right_layout.addWidget(self.selection)
        decisions = QtWidgets.QHBoxLayout()
        right_layout.addLayout(decisions)
        self.change_button = self.button(decisions, 'Camera changed here', self.mark_change)
        self.unchanged_button = self.button(decisions, 'Unchanged', lambda: self.decide('unchanged'))
        self.unchanged_button.setToolTip('The camera view did not change at this stop.')
        self.shake_button = self.button(decisions, 'Shake', self.mark_shake)
        self.shake_button.setToolTip('A momentary jolt (about one frame): saves a one-frame jitter period at this frame '
                                     'and marks the selected stop Unchanged. For longer jitter, use Start jitter.')
        other = QtWidgets.QHBoxLayout()
        right_layout.addLayout(other)
        self.uncertain_button = self.button(other, 'Revisit later', lambda: self.decide('uncertain'))
        self.reopen_button = self.button(other, 'Clear decision', lambda: self.decide('pending'))
        self.reopen_button.setToolTip('Reset this stop to pending and detach its camera-change marker. A marker shared with another stop or saved standalone is retained.')
        self.remove_button = self.button(other, 'Remove marker', self.remove_marker)
        right_layout.addWidget(QtWidgets.QLabel('Periods · click a start or end to visit it'))
        self.period_tree = self.make_tree(('Kind', 'Start', 'End'))
        self.period_tree.itemSelectionChanged.connect(self.select_period)
        self.period_tree.itemClicked.connect(self.visit_period)
        right_layout.addWidget(self.period_tree, 1)
        period_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(period_row)
        self.period_buttons = {kind: self.button(period_row, f'Start {label.lower()}',
                                                 lambda _checked=False, kind=kind: self.toggle_period(kind))
                               for kind, label in PERIOD_LABELS.items()}
        self.remove_period_button = self.button(period_row, 'Remove period', self.remove_period)
        # Mouse-only: a focused button would re-fire on Space and save a stray period.
        for button in (*self.period_buttons.values(), self.remove_period_button):
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        notes_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(notes_row)
        notes_row.addWidget(QtWidgets.QLabel('Notes'))
        self.notes = QtWidgets.QLineEdit(self.review.notes)
        self.notes.setAccessibleName('Video notes')
        self.notes.setToolTip('Notes for this whole video. Saved as you type; Enter returns to the video.')
        self.notes.setPlaceholderText('e.g. doubles match: exclude from benchmarking')
        self.notes.textEdited.connect(self.review.set_notes)
        self.notes.returnPressed.connect(self.finish_notes)
        notes_row.addWidget(self.notes)
        self.status = self.prompt

    def make_tree(self, columns: tuple[str, ...]) -> QtWidgets.QTreeWidget:
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(columns)
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        tree.setMinimumWidth(250)
        tree.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        tree.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        return tree

    def jump_to_frame(self) -> None:
        try:
            frame = int(self.jump_entry.text())
        except ValueError:
            self.status.setText('Enter a whole frame number, then press Enter.')
            return
        self.load(frame)
        self.video.setFocus()

    def load(self, frame: int) -> None:
        if not self.browsing:
            return
        frame = min(self.total - 1, max(0, frame))
        if frame != self.frame_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, decoded = self.capture.read()
            if not ok:
                self.status.setText(f'Could not decode frame {frame}; the previous frame remains displayed.')
                return
            self.frame, self.frame_index = decoded, frame
            self.loupe_origin = None
            self.zoom.set_image(None)
        self.jump_entry.setText(str(frame))
        self.position.setText(f'{clock_label(frame, self.fps)} / {clock_label(self.total - 1, self.fps)}   ·   frame {frame}')
        self.refresh_state()
        self.draw()

    def step_frames(self, direction: int) -> None:
        shifted = QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        self.load(self.frame_index + direction * (BIG_FRAME_STEP if shifted else 1))

    def act(self, action: SessionAction, show_message: bool = True) -> None:
        if action.kind == ActionKind.OPEN_LOUPE:
            self.aim_started = monotonic()
            self.loupe_origin = self.session.pending_origin
        message = action.message if show_message else None
        if action.kind == ActionKind.COMMITTED:
            if self.queue_active and not self.review.annotated(self.frame_index):
                message = f'{action.message}. Saved independently: this court precedes the confirmed change. Mark a court at or after the change to complete this stop.'
            self.annotations = read_saved_courts(self.session.csv_path, self.session.video_basename)
            self.refresh_lists()
        if action.kind in (ActionKind.COMMITTED, ActionKind.ABORTED):
            self.video.setFocus()
        if not self.browsing:
            self.browse_zoom = None
        self.refresh_state()
        if message:
            self.status.setText(message)
        self.draw()

    def refresh_state(self) -> None:
        self.timeline.set_data(self.total, self.fps, max(0, self.frame_index), list(self.annotations),
                               [item.frame for item in self.review.items], self.review.markers, self.review.periods)
        browsing, adjusting = self.browsing, self.session.adjusting
        for widget in (self.back, self.forward, self.back_second, self.forward_second,
                       self.jump_entry, self.timeline, self.mark_button, self.previous_button, self.next_button,
                       self.marker_next, self.change_button, self.remove_button, self.saved_tree, self.queue_tree,
                       self.video_picker, self.shake_button, *self.period_buttons.values()):
            widget.setEnabled(browsing)
        self.next_video_button.setEnabled(browsing and self.video_index + 1 < len(self.videos))
        self.remove_period_button.setEnabled(browsing and bool(self.period_tree.selectedItems()))
        for kind, button in self.period_buttons.items():
            open_period = self.review.open_period(kind)
            label = PERIOD_LABELS[kind].lower()
            # A coloured End button makes an open period hard to forget; a missed End
            # would otherwise flip every later Start and End.
            if open_period is None:
                button.setText(f'Start {label}')
                button.setStyleSheet('')
            else:
                button.setText(f'End {label} · {clock_label(open_period["start"], self.fps)[:5]}')
                button.setStyleSheet(f'background: {PERIOD_LANES[kind][2]}; color: black; font-weight: bold')
            button.setToolTip(f'Start a {PERIOD_LABELS[kind].lower()} period at this frame.' if open_period is None
                              else f'End the {PERIOD_LABELS[kind].lower()} period from frame {open_period["start"]} '
                                   'at this frame (inclusive).')
        self.mark_button.setEnabled(browsing and self.frame is not None)
        self.copy_button.setEnabled(browsing and bool(self.annotations))
        source = min(self.annotations, key=lambda frame: abs(frame - self.frame_index)) if self.annotations else None
        self.copy_button.setText(f'Copy court · frame {source}' if source is not None else 'Copy court')
        self.copy_button.setToolTip(f'Copy the nearest saved court: frame {source}, {clock_label(source, self.fps)}.'
                                   if source is not None else 'Save a court before copying one.')
        for widget in (self.unchanged_button, self.uncertain_button, self.reopen_button):
            widget.setEnabled(browsing and self.queue_active)
        self.landmark.setEnabled(not browsing)
        with QtCore.QSignalBlocker(self.landmark):
            selected = self.session.selected_name
            self.landmark.setCurrentIndex(court.LANDMARK_NAMES.index(selected) + 1 if selected else 0)
        self.fit_button.setEnabled(not adjusting and len(self.session.landmarks) >= court.FIT_MIN_POINTS)
        self.confirm_button.setEnabled(adjusting and self.session.selected_name is not None)
        self.save_button.setEnabled(self.session.can_save)
        self.editor.setVisible(not browsing)
        self.review_panel.setVisible(browsing)
        self.timeline.setVisible(browsing)
        placed = any(name == self.session.selected_name for name, _x, _y in self.session.landmarks)
        if browsing:
            self.prompt.setText('Click any court intersection to start. Left / Right steps frames.')
        elif self.session.can_save:
            self.prompt.setText('Court ready. Choose Save court, or select a point to revise it.')
        elif self.session.selected_name is None:
            self.prompt.setText('Select this intersection on the court diagram.' if adjusting
                                else 'Choose a point identity, or click any visible intersection first.')
        else:
            target = point_label(self.session.selected_name)
            suffix = 'refine, then Confirm point.' if adjusting else 'click the video to aim.'
            self.prompt.setText(f'{target}: {suffix} {len(self.session.landmarks)} points confirmed.')
        if not browsing and placed:
            self.prompt.setText(self.prompt.text() + ' Selected identity is already marked: Confirm point will replace it; Undo restores it.')
        item = self.review.current
        if self.queue_active:
            pending = sum(self.review.needs_review(entry) for entry in self.review.items)
            self.selection.setText(f'Stop {self.review.index + 1}/{len(self.review.items)}: {item.status}. '
                                   f'{pending} pending. Change: {item.change_frame}; court: {item.annotation_frame}.')
        else:
            self.selection.setText('Saved-frame inspection. Select a suspected movement to make a review decision.')

    def draw(self) -> None:
        if self.frame is None:
            return
        image = self.frame.copy()
        outline = None
        self.prior_label.setText('')
        choice = self.overlay.currentData()
        if choice is not None:
            prior = (previous_court(self.annotations, self.frame_index) if choice == OVERLAY_PREVIOUS
                     else (choice, self.annotations[choice]))
            if prior is None:
                self.prior_label.setText('No saved court at or before this frame.')
            else:
                source, outline = prior
                self.prior_label.setText(f'Overlay: saved court from frame {source}, {clock_label(source, self.fps)}')
        if not self.browsing:
            if self.session.fit is not None:
                outline = self.session.fit.corners_px
            elif all(point is not None for point in self.session.slots):
                outline = np.asarray(self.session.slots, dtype=float)
            if self.session.fit is not None or all(point is not None for point in self.session.slots):
                self.prior_label.setText('Showing the court being marked.')
            for _name, x, y in self.session.landmarks:
                cv2.circle(image, (round(x), round(y)), 5, (0, 0, 0), -1)
                cv2.circle(image, (round(x), round(y)), 3, CORNER_COLOUR, -1)
        if outline is not None:
            draw_outline(image, outline)
        if not self.browsing and self.session.pending_point is not None:
            x, y = self.session.pending_point
            centre = (round(x), round(y))
            cv2.drawMarker(image, centre, (0, 0, 0), cv2.MARKER_CROSS, 41, 6)
            cv2.drawMarker(image, centre, CORNER_COLOUR, cv2.MARKER_CROSS, 41, 4)
        inset = None
        if self.browsing and self.browse_zoom is not None:
            inset = render_loupe(self.frame, self.browse_zoom, None, self.session.half, self.session.zoom, outline)
        # The inset goes on the side away from the zoomed spot so it never hides it.
        self.video.set_image(image, inset, inset_on_right=self.browse_zoom is not None
                             and self.browse_zoom[0] + self.session.half < self.session.width / 2)
        if self.loupe_origin is not None:
            self.zoom.set_image(render_loupe(self.frame, self.loupe_origin,
                                             self.session.pending_point, self.session.half, self.session.zoom, outline))
        else:
            self.zoom.set_image(None)
        self.diagram.current = self.session.selected_name
        self.diagram.assigned = {name for name, _x, _y in self.session.landmarks}
        self.diagram.update()

    def start_pan(self) -> None:
        self.pan_origin = self.loupe_origin

    def pan_zoom(self, dx: float, dy: float) -> None:
        if self.pan_origin is None or self.browsing:
            return
        width, height = self.session.width, self.session.height
        side = self.session.half * 2
        self.loupe_origin = (min(max(round(self.pan_origin[0] - dx / self.session.zoom), 0), max(0, width - side)),
                             min(max(round(self.pan_origin[1] - dy / self.session.zoom), 0), max(0, height - side)))
        if self.session.adjusting:
            self.session.set_loupe_origin(self.loupe_origin)
        self.draw()

    def choose_diagram(self, name: str) -> None:
        self.act(self.session.select_point(name), show_message=False)
        self.video.setFocus()

    def click_video(self, x: float, y: float) -> None:
        if self.browsing:
            self.session.begin_capture(self.frame_index)
        self.act(self.session.coarse_click(x, y), show_message=False)

    def look_at(self, x: float, y: float) -> None:
        if self.frame is None:
            return
        origin = loupe_origin(x, y, self.session.half, self.session.width, self.session.height)
        if self.browsing:
            self.browse_zoom = origin
        else:
            self.loupe_origin = origin
            if self.session.adjusting:
                self.session.set_loupe_origin(origin)
        self.draw()

    def click_inset(self, fraction_x: float, fraction_y: float) -> None:
        if self.browsing and self.browse_zoom is not None:
            side = 2 * self.session.half
            self.click_video(self.browse_zoom[0] + fraction_x * side, self.browse_zoom[1] + fraction_y * side)

    def click_zoom(self, x: float, y: float) -> None:
        if not self.browsing and self.loupe_origin is not None and monotonic() - self.aim_started >= 0.35:
            origin = self.loupe_origin
            if not self.session.adjusting:
                self.session.coarse_click(origin[0] + x / self.session.zoom, origin[1] + y / self.session.zoom)
                self.session.set_loupe_origin(origin)
            self.act(self.session.reposition_from_loupe(x, y), show_message=False)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() != QtCore.QEvent.Type.KeyPress or not isinstance(event, QtGui.QKeyEvent):
            return False
        if not isinstance(watched, QtWidgets.QWidget) or watched.window() is not self:
            return False
        focus = QtWidgets.QApplication.focusWidget()
        while focus is not None and focus is not self:
            if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
                return False
            focus = focus.parentWidget()
        if isinstance(watched, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            return False
        directions = {QtCore.Qt.Key.Key_Left: (-1, 0), QtCore.Qt.Key.Key_Right: (1, 0),
                      QtCore.Qt.Key.Key_Up: (0, -1), QtCore.Qt.Key.Key_Down: (0, 1)}
        if event.key() in directions:
            dx, dy = directions[event.key()]
            if self.browsing and dx:
                shifted = event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
                self.load(self.frame_index + dx * (BIG_FRAME_STEP if shifted else 1))
            elif self.session.adjusting:
                self.nudge(dx, dy)
            else:
                return False
            return True
        if (event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space)
                and not self.browsing):
            self.confirm()
            return True
        if event.key() == QtCore.Qt.Key.Key_Escape and self.browsing and self.browse_zoom is not None:
            self.browse_zoom = None
            self.draw()
            return True
        if event.key() == QtCore.Qt.Key.Key_Escape and not self.browsing:
            if self.session.adjusting:
                self.act(self.session.cancel_aim())
            elif self.session.landmarks:
                self.status.setText(f'{len(self.session.landmarks)} confirmed points kept. '
                                    'Choose Cancel court to discard them.')
            else:
                self.act(self.session.abort())
            return True
        return False

    def nudge(self, dx: int, dy: int) -> None:
        self.act(self.session.nudge(dx * self.nudge_step, dy * self.nudge_step), show_message=False)

    def confirm(self) -> None:
        if self.session.adjusting:
            self.act(self.session.confirm_point(), show_message=False)
        else:
            self.act(self.session.commit())

    def choose_landmark(self, index: int) -> None:
        if not self.browsing:
            name = court.LANDMARK_NAMES[index - 1] if index > 0 else None
            self.act(self.session.select_point(name), show_message=False)

    def copy_court(self) -> None:
        prefill = load_prefill(self.session.csv_path, self.session.landmark_csv_path,
                               self.session.video_basename, self.frame_index)
        if prefill is not None:
            source, slots, landmarks = prefill
            self.act(self.session.begin_capture_prefilled(self.frame_index, slots, landmarks, source))

    def refresh_lists(self) -> None:
        for tree in (self.saved_tree, self.queue_tree, self.period_tree):
            tree.blockSignals(True)
            tree.clear()
        for frame, points in sorted(self.annotations.items()):
            values = [clock_label(frame, self.fps), str(frame), *(f'{x:.1f}, {y:.1f}' for x, y in points)]
            item = QtWidgets.QTreeWidgetItem(values)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, frame)
            self.saved_tree.addTopLevelItem(item)
        for index, entry in enumerate(self.review.items):
            item = QtWidgets.QTreeWidgetItem([clock_label(entry.frame, self.fps), entry.status, entry.reason])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, index)
            self.queue_tree.addTopLevelItem(item)
            if index == self.review.index and self.queue_active:
                item.setSelected(True)
        for index, period in sorted(enumerate(self.review.periods), key=lambda entry: entry[1]['start']):
            end = 'open' if period['end'] is None else self.frame_text(period['end'])
            item = QtWidgets.QTreeWidgetItem([PERIOD_LABELS[period['kind']], self.frame_text(period['start']), end])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, index)
            self.period_tree.addTopLevelItem(item)
        for tree in (self.saved_tree, self.queue_tree, self.period_tree):
            tree.blockSignals(False)
        choice = self.overlay.currentData()
        with QtCore.QSignalBlocker(self.overlay):
            self.overlay.clear()
            self.overlay.addItem('No court overlay', None)
            self.overlay.addItem('Overlay previous saved court', OVERLAY_PREVIOUS)
            for frame in sorted(self.annotations):
                self.overlay.addItem(f'Overlay court from frame {frame} · {clock_label(frame, self.fps)}', frame)
            self.overlay.setCurrentIndex(max(0, self.overlay.findData(choice)))

    def frame_text(self, frame: int) -> str:
        return f'{clock_label(frame, self.fps)} · frame {frame}'

    def inspect_saved(self, frame: int) -> None:
        if not self.browsing:
            return
        self.queue_active = False
        self.queue_tree.clearSelection()
        # "Previous saved court" already shows this court here and keeps following later ones.
        if self.overlay.currentData() != OVERLAY_PREVIOUS:
            self.overlay.setCurrentIndex(self.overlay.findData(frame))
        self.load(frame)

    def inspect_candidate(self, frame: int) -> None:
        if not self.browsing:
            return
        self.review.index = next(index for index, item in enumerate(self.review.items) if item.frame == frame)
        self.queue_active = True
        self.refresh_lists()
        self.queue_tree.scrollToItem(self.queue_tree.topLevelItem(self.review.index))
        self.load(frame)

    def inspect_marker(self, frame: int) -> None:
        if not self.browsing:
            return
        self.queue_active = False
        self.queue_tree.clearSelection()
        self.saved_tree.clearSelection()
        self.load(frame)
        self.status.setText(f'Confirmed camera change at {clock_label(frame, self.fps)}, frame {frame}.')

    def select_saved(self) -> None:
        selected = self.saved_tree.selectedItems()
        if selected:
            self.inspect_saved(selected[0].data(0, QtCore.Qt.ItemDataRole.UserRole))

    def select_candidate(self) -> None:
        selected = self.queue_tree.selectedItems()
        if selected:
            self.inspect_candidate(self.review.items[selected[0].data(0, QtCore.Qt.ItemDataRole.UserRole)].frame)

    def move_queue(self, direction: int) -> None:
        if not self.browsing:
            return
        self.review.move(direction, pending_only=direction > 0)
        self.queue_active = True
        self.refresh_lists()
        self.queue_tree.scrollToItem(self.queue_tree.topLevelItem(self.review.index))
        self.load(self.review.current.frame)
        self.report_all_reviewed()

    def report_all_reviewed(self) -> None:
        if not any(self.review.needs_review(entry) for entry in self.review.items):
            self.status.setText('All stops reviewed. Choose Next video to continue.'
                                if self.next_video_button.isEnabled()
                                else 'All stops reviewed. Close the workspace when finished, or select a stop to revisit.')

    def mark_change(self) -> None:
        self.review.mark_change(self.frame_index, associate=self.queue_active)
        self.refresh_lists()
        self.refresh_state()
        self.status.setText('Saved camera change for this stop. Mark a clear court frame after it.' if self.queue_active
                            else 'Saved standalone camera-change marker.')

    def decide(self, status: str) -> None:
        if not self.queue_active:
            self.status.setText('Select a suspected movement first.')
            return
        self.review.decide(status)
        self.refresh_lists()
        self.refresh_state()
        self.status.setText(f'Saved: {status}. Choose Next pending to continue.')
        self.report_all_reviewed()

    def next_marker(self) -> None:
        frame = self.review.next_change(self.frame_index)
        if frame is None:
            self.status.setText('No confirmed camera changes yet.')
            return
        self.inspect_marker(frame)

    def remove_marker(self) -> None:
        removed = self.review.remove_change(self.frame_index)
        self.refresh_lists()
        self.refresh_state()
        self.status.setText('Removed marker here.' if removed else 'No marker on this exact frame.')

    def select_period(self) -> None:
        # Keyboard selection visits the start; a mouse click is handled by visit_period, which knows the column.
        selected = self.period_tree.selectedItems()
        if selected and QtWidgets.QApplication.mouseButtons() == QtCore.Qt.MouseButton.NoButton:
            self.load(self.review.periods[selected[0].data(0, QtCore.Qt.ItemDataRole.UserRole)]['start'])
        self.refresh_state()

    def visit_period(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        period = self.review.periods[item.data(0, QtCore.Qt.ItemDataRole.UserRole)]
        end_column = 2
        self.load(period['end'] if column == end_column and period['end'] is not None else period['start'])

    def toggle_period(self, kind: str) -> None:
        label = PERIOD_LABELS[kind]
        try:
            if self.review.open_period(kind) is None:
                self.review.start_period(kind, self.frame_index)
                message = (f'{label} starts at {self.frame_text(self.frame_index)}. '
                           f'Go to its last frame, then choose End {label.lower()}.')
            else:
                self.review.end_period(kind, self.frame_index)
                message = f'{label} period saved, ending at {self.frame_text(self.frame_index)}.'
        except ValueError as error:
            self.status.setText(f'{error}. Go to a later frame first.')
            return
        self.refresh_lists()
        self.refresh_state()
        self.status.setText(message)

    def mark_shake(self) -> None:
        try:
            self.review.add_shake(self.frame_index)
        except ValueError as error:
            self.status.setText(f'{error}.')
            return
        if self.queue_active:
            self.review.decide('unchanged')
        self.refresh_lists()
        self.refresh_state()
        stop_note = '; stop marked Unchanged' if self.queue_active else ''
        self.status.setText(f'Shake saved as a one-frame jitter at {self.frame_text(self.frame_index)}{stop_note}.')

    def remove_period(self) -> None:
        selected = self.period_tree.selectedItems()
        if not selected:
            return
        period = self.review.periods[selected[0].data(0, QtCore.Qt.ItemDataRole.UserRole)]
        self.review.remove_period(period)
        self.refresh_lists()
        self.refresh_state()
        self.status.setText(f'Removed {PERIOD_LABELS[period["kind"]].lower()} period from {self.frame_text(period["start"])}.')

    def finish_notes(self) -> None:
        self.status.setText('Notes saved.')
        self.video.setFocus()

    def choose_video(self, index: int) -> None:
        if index == self.video_index or not self.browsing:
            return
        self.chosen_video = index
        self.close()

    def next_video(self) -> None:
        if self.browsing and self.video_index + 1 < len(self.videos):
            self.chosen_video = self.video_index + 1
            self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.capture.release()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        event.accept()
