"""Native widget actions and source-coordinate mapping at real UI boundaries."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6 import QtCore, QtTest, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from court_annotation import court_landmarks as court
from court_annotation.annotate_court_corners_offframe import OffframeSession
from court_annotation.native_gui import CourtWorkspace, image_placement
from court_annotation.review_queue import ReviewItem, ReviewQueue


def mark_outer_points(window, points):
    for name, point in zip(court.CORNER_LANDMARK_NAMES, points, strict=True):
        window.landmark.setCurrentIndex(court.LANDMARK_NAMES.index(name) + 1)
        window.click_video(*point)
        window.confirm_button.click()
    window.save_button.click()


def test_offscreen_corner_through_landmark_widgets(workspace):
    from court_annotation import court_landmarks as court
    from court_annotation.annotate_court_corners_offframe import CaptureState

    app, window = workspace
    expected = np.float32([(100, 60), (500, 60), (550, 320), (-20, 370)])
    transform = cv2.getPerspectiveTransform(np.float32(court.CORNER_COURT_M), expected)
    window.mark_button.click()
    for name, (x, y) in zip(court.CORNER_LANDMARK_NAMES[:3], expected[:3], strict=True):
        window.landmark.setCurrentIndex(court.LANDMARK_NAMES.index(name) + 1)
        window.click_video(float(x), float(y))
        window.confirm_button.click()
    assert window.session.state == CaptureState.POINT_COARSE
    for name in ('near_short_service_x_left_doubles', 'far_short_service_x_centre'):
        window.landmark.setCurrentIndex(court.LANDMARK_NAMES.index(name) + 1)
        source = np.float32([[court.LANDMARKS[name]]])
        x, y = cv2.perspectiveTransform(source, transform)[0, 0]
        window.click_video(float(x), float(y))
        window.confirm_button.click()
    window.fit_button.click()
    assert window.session.fit is not None
    assert window.session.fit.corners_px == pytest.approx(expected, abs=0.01)
    window.save_button.click()
    app.processEvents()
    assert window.browsing
    assert window.saved_tree.topLevelItemCount() == 1


@pytest.mark.parametrize('source,size', [((1920, 1080), (800, 600)), ((960, 720), (1000, 400)),
                                        ((640, 360), (321, 201)), ((1024, 1024), (200, 200))])
def test_letterbox_mapping(source, size):
    placement = image_placement(*source, *size)
    point = placement.source_point(placement.left + placement.width / 2,
                                   placement.top + placement.height / 2)
    assert point == pytest.approx((source[0] / 2, source[1] / 2))
    assert placement.source_point(placement.left - 1, placement.top) is None
    assert placement.source_point(placement.left + placement.width, placement.top) is None


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    video = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25, (640, 360))
    for index in range(30):
        writer.write(np.full((360, 640, 3), index * 3, dtype=np.uint8))
    writer.release()
    session = OffframeSession(str(video), 640, 360, 'portrait', tmp_path / 'corners.csv', 20, 8)
    queue = ReviewQueue(tmp_path / 'review.json.gz', video.name,
                        [ReviewItem(0, 'Initial'), ReviewItem(10, 'Movement')])
    window = CourtWorkspace(cv2.VideoCapture(str(video)), session, queue, 30, 25, 0, .5)
    window.show()
    app.processEvents()
    yield app, window
    window.close()
    app.processEvents()


def test_arrows_buttons_and_point_refinement(workspace):
    app, window = workspace
    QtTest.QTest.keyClick(window.video, QtCore.Qt.Key.Key_Right)
    assert window.frame_index == 1
    QtTest.QTest.keyClick(window.queue_tree, QtCore.Qt.Key.Key_Right)
    assert window.frame_index == 2
    window.mark_button.click()
    app.processEvents()
    assert not window.forward.isEnabled()
    window.load(20)
    assert window.frame_index == 2
    box = window.video.placement
    QtTest.QTest.mouseClick(window.video, QtCore.Qt.MouseButton.LeftButton,
                           pos=QtCore.QPoint(box.left + box.width // 2, box.top + box.height // 2))
    point = window.session.pending_point
    assert point == pytest.approx((320, 180), abs=2)
    QtTest.QTest.keyClick(window.video, QtCore.Qt.Key.Key_Left)
    assert window.session.pending_point == pytest.approx((point[0] - .5, point[1]))
    window.aim_started = 0
    app.processEvents()
    zoom_box = window.zoom.placement
    QtTest.QTest.mouseClick(window.zoom, QtCore.Qt.MouseButton.LeftButton,
                           pos=QtCore.QPoint(zoom_box.left + zoom_box.width // 2,
                                            zoom_box.top + zoom_box.height // 2))
    assert window.session.adjusting
    assert not window.confirm_button.isEnabled()
    original_aim = window.session.pending_point
    window.landmark.setCurrentIndex(1)
    assert window.session.pending_point == original_aim
    window.confirm_button.click()
    assert len(window.session.landmarks) == 1
    window.cancel_button.click()
    assert window.browsing


def test_video_click_starts_capture_at_clicked_point(workspace):
    app, window = workspace
    box = window.video.placement
    QtTest.QTest.mouseClick(window.video, QtCore.Qt.MouseButton.LeftButton,
                           pos=QtCore.QPoint(box.left + box.width // 4, box.top + box.height // 3))
    app.processEvents()
    assert window.session.adjusting
    assert window.session.pending_point == pytest.approx((160, 120), abs=2)
    assert window.editor.isVisible()
    assert not window.session.csv_path.exists()
    window.cancel_button.click()
    assert window.browsing


def test_queue_decision_marker_and_court_remain_distinct(workspace):
    app, window = workspace
    window.next_button.click()
    assert window.frame_index == 10
    window.change_button.click()
    window.load(14)
    window.mark_button.click()
    mark_outer_points(window, [(100, 80), (500, 80), (550, 300), (60, 300)])
    assert window.review.current.change_frame == 10
    assert window.review.current.annotation_frame == 14
    assert window.saved_tree.topLevelItemCount() == 1
    window.saved_tree.setCurrentItem(window.saved_tree.topLevelItem(0))
    app.processEvents()
    assert not window.queue_active
    assert not window.unchanged_button.isEnabled()
    window.load(20)
    assert 'frame 14' in window.prior_label.text()
    window.marker_next.click()
    assert window.frame_index == 10
    assert not window.queue_active


def test_copy_court_requires_explicit_save_and_enter_can_save(workspace):
    app, window = workspace
    window.mark_button.click()
    mark_outer_points(window, [(100, 80), (500, 80), (550, 300), (60, 300)])
    window.load(5)
    window.copy_button.setFocus()
    window.copy_button.click()
    app.processEvents()
    assert not window.session.adjusting
    assert window.save_button.isEnabled()
    QtTest.QTest.keyClick(window.copy_button, QtCore.Qt.Key.Key_Return)
    assert window.browsing
    assert 5 in window.annotations


def test_capture_keeps_video_large_and_browsing_drops_capture_overlay(workspace, monkeypatch):
    app, window = workspace
    window.resize(1320, 740)
    app.processEvents()
    browsing_height = window.video.height()
    window.mark_button.click()
    app.processEvents()
    assert not window.review_panel.isVisible()
    assert not window.timeline.isVisible()
    assert window.video.height() >= browsing_height
    mark_outer_points(window, [(100, 80), (500, 80), (550, 300), (60, 300)])
    app.processEvents()
    assert window.review_panel.isVisible()
    assert window.timeline.isVisible()
    rendered = []
    monkeypatch.setattr(window.video, 'set_image', lambda image, *_inset, **_side: rendered.append(image.copy()))
    window.load(5)
    assert window.overlay.currentData() is None
    assert np.array_equal(rendered[-1], window.frame)


def test_frame_jump_restores_browsing_focus_and_zoom_is_eight_times(workspace):
    app, window = workspace
    window.activateWindow()
    window.jump_entry.setFocus()
    window.jump_entry.setText('5')
    QtTest.QTest.keyClick(window.jump_entry, QtCore.Qt.Key.Key_Return)
    app.processEvents()
    assert window.frame_index == 5
    assert app.focusWidget() is window.video
    QtTest.QTest.keyClick(window.video, QtCore.Qt.Key.Key_Left)
    assert window.frame_index == 4
    window.mark_button.click()
    window.click_video(200, 150)
    app.processEvents()
    assert len(window.session.landmarks) == 0
    assert window.zoom.image.width() == 320
    assert window.zoom.width() / (2 * window.session.half) == 8


def test_frame_text_enter_clamps_and_navigation_buttons_are_paired(workspace):
    app, window = workspace
    for text, expected in [('12', 12), ('-500', 0), ('999999999999', window.total - 1)]:
        window.jump_entry.setFocus()
        window.jump_entry.selectAll()
        QtTest.QTest.keyClicks(window.jump_entry, text)
        QtTest.QTest.keyClick(window.jump_entry, QtCore.Qt.Key.Key_Return)
        app.processEvents()
        assert window.frame_index == expected
        assert window.jump_entry.text() == str(expected)
    assert window.back.x() < window.forward.x() < window.back_second.x() < window.forward_second.x()


def test_vertical_arrows_navigate_review_table(workspace):
    _app, window = workspace
    window.queue_tree.setCurrentItem(window.queue_tree.topLevelItem(0))
    window.queue_tree.setFocus()
    QtTest.QTest.keyClick(window.queue_tree, QtCore.Qt.Key.Key_Down)
    assert window.review.index == 1
    assert window.frame_index == 10


def test_earlier_court_does_not_complete_later_camera_change(workspace):
    _app, window = workspace
    window.next_button.click()
    window.change_button.click()
    window.load(5)
    window.mark_button.click()
    mark_outer_points(window, [(100, 80), (500, 80), (550, 300), (60, 300)])
    assert 5 in window.annotations
    assert window.review.current.change_frame == 10
    assert window.review.current.annotation_frame is None
    assert window.review.needs_review(window.review.current)
    assert 'Saved independently' in window.status.text()


def test_timeline_candidate_click_changes_review_target_without_erasing_previous_marker(workspace):
    from court_annotation.timeline import LANES, x_at_frame

    app, window = workspace
    window.load(3)
    window.change_button.click()
    app.processEvents()
    timeline = window.timeline
    x = x_at_frame(10, timeline.canvas.width(), timeline.start, timeline.end)
    QtTest.QTest.mouseClick(timeline.canvas, QtCore.Qt.MouseButton.LeftButton,
                           pos=QtCore.QPoint(round(x), LANES[1][0]))
    assert window.review.index == 1
    assert window.frame_index == 10
    window.unchanged_button.click()
    assert window.review.items[0].status == 'changed'
    assert window.review.items[0].change_frame == 3
    assert window.review.markers == [3]
    assert window.review.items[1].status == 'unchanged'


def test_dropdown_activation_returns_focus_to_video_for_nudging(workspace):
    app, window = workspace
    window.activateWindow()
    window.click_video(200, 150)
    window.landmark.setFocus()
    app.processEvents()
    window.landmark.setCurrentIndex(1)
    assert app.focusWidget() is window.landmark
    window.landmark.activated.emit(1)
    app.processEvents()
    assert app.focusWidget() is window.video
    chosen = window.session.selected_name
    QtTest.QTest.keyClick(window.video, QtCore.Qt.Key.Key_Right)
    assert window.session.pending_point == (200.5, 150)
    assert window.session.selected_name == chosen


def test_advance_warns_about_existing_identity_and_queue_completion_is_visible(workspace):
    _app, window = workspace
    window.click_video(200, 100)
    window.landmark.setCurrentIndex(2)
    window.confirm_button.click()
    window.click_video(100, 100)
    window.landmark.setCurrentIndex(1)
    window.confirm_button.click()
    assert window.session.selected_name == court.LANDMARK_NAMES[1]
    assert 'already marked' in window.prompt.text()
    assert 'replace' in window.prompt.text()
    window.cancel_button.click()
    window.unchanged_button.click()
    window.next_button.click()
    window.unchanged_button.click()
    assert 'All stops reviewed' in window.status.text()
    window.next_button.click()
    assert 'All stops reviewed' in window.status.text()


def test_diagram_reselect_pan_confirm_and_save_preserve_coordinates(workspace):
    import csv

    app, window = workspace
    expected = [(100, 80), (500, 80), (550, 300), (60, 300)]
    window.click_video(*expected[2])
    app.processEvents()
    for index in (2, 0, 3, 1):
        name = court.CORNER_LANDMARK_NAMES[index]
        point = window.diagram.point_positions()[name]
        QtTest.QTest.mouseClick(window.diagram, QtCore.Qt.MouseButton.LeftButton, pos=point.toPoint())
        assert window.session.selected_name == name
        window.click_video(*expected[index])
        before = window.session.pending_point
        origin = window.loupe_origin
        window.aim_started = 0
        app.processEvents()
        QtTest.QTest.mousePress(window.zoom, QtCore.Qt.MouseButton.RightButton, pos=QtCore.QPoint(160, 160))
        QtTest.QTest.mouseMove(window.zoom, QtCore.QPoint(184, 176))
        QtTest.QTest.mouseRelease(window.zoom, QtCore.Qt.MouseButton.RightButton, pos=QtCore.QPoint(184, 176))
        assert window.session.pending_point == before
        assert window.loupe_origin == (origin[0] - 3, origin[1] - 2)
        assert window.session.pending_origin == window.loupe_origin
        # Clicking the same source pixel through the panned viewport preserves it.
        px = round((before[0] - window.loupe_origin[0]) * window.session.zoom)
        py = round((before[1] - window.loupe_origin[1]) * window.session.zoom)
        QtTest.QTest.mouseClick(window.zoom, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(px, py))
        assert window.session.pending_point == before
        window.confirm_button.click()
        assert not window.zoom.image.isNull()
        assert name in window.diagram.assigned
    window.save_button.click()
    with window.session.csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(float(row['x_px']), float(row['y_px'])) for row in rows] == expected
    assert [int(row['corner_idx']) for row in rows] == [0, 1, 2, 3]
    window.load(5)
    assert window.loupe_origin is None
    assert window.zoom.image.isNull()
    assert window.copy_button.text() == 'Copy court · frame 0'
    assert 'frame 0' in window.copy_button.toolTip()


def test_diagram_click_reopens_already_selected_confirmed_identity(workspace):
    app, window = workspace
    window.click_video(200, 150)
    window.choose_diagram(court.LANDMARK_NAMES[1])
    window.confirm_button.click()
    window.choose_diagram(court.LANDMARK_NAMES[0])
    window.click_video(100, 150)
    window.confirm_button.click()
    assert window.session.selected_name == court.LANDMARK_NAMES[1]
    assert not window.session.adjusting
    app.processEvents()
    point = window.diagram.point_positions()[court.LANDMARK_NAMES[1]]
    QtTest.QTest.mouseClick(window.diagram, QtCore.Qt.MouseButton.LeftButton, pos=point.toPoint())
    assert window.session.adjusting
    assert window.session.pending_point == (200, 150)


def test_persistent_loupe_click_starts_next_point_and_fit_errors_remain_visible(workspace):
    app, window = workspace
    window.click_video(200, 150)
    window.choose_diagram(court.LANDMARK_NAMES[0])
    window.confirm_button.click()
    origin = window.loupe_origin
    window.start_pan()
    window.pan_zoom(80000, 80000)
    assert window.loupe_origin == (0, 0)
    assert window.session.landmarks == ((court.LANDMARK_NAMES[0], 200, 150),)
    window.start_pan()
    window.pan_zoom(-origin[0] * 8, -origin[1] * 8)
    window.aim_started = 0
    app.processEvents()
    QtTest.QTest.mouseClick(window.zoom, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(164, 160))
    assert window.session.pending_point == (200.5, 150)
    assert window.session.pending_origin == origin
    assert window.session.selected_name == court.LANDMARK_NAMES[1]
    window.confirm_button.click()
    for index in (2, 3, 4):
        window.click_video(200 + index * .5, 150)
        window.choose_diagram(court.LANDMARK_NAMES[index])
        window.confirm_button.click()
    window.fit_button.click()
    assert 'fit failed' in window.prompt.text()
    assert window.status is window.prompt


def test_ten_frame_steps_and_right_click_zoom_keep_browsing(workspace):
    app, window = workspace
    window.load(15)
    QtTest.QTest.mouseClick(window.windowHandle(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ShiftModifier,
                            window.forward.mapTo(window, QtCore.QPoint(5, 5)))
    app.processEvents()
    assert window.frame_index == 25
    # Quick unshifted taps step one frame each.
    for _tap in range(4):
        QtTest.QTest.mouseClick(window.windowHandle(), QtCore.Qt.MouseButton.LeftButton,
                                pos=window.back.mapTo(window, QtCore.QPoint(5, 5)))
    app.processEvents()
    assert window.frame_index == 21
    window.load(25)
    box = window.video.placement
    QtTest.QTest.mousePress(window.video, QtCore.Qt.MouseButton.RightButton,
                            pos=QtCore.QPoint(box.left + box.width // 2, box.top + box.height // 2))
    QtTest.QTest.mouseRelease(window.video, QtCore.Qt.MouseButton.RightButton)
    QtTest.QTest.keyClick(window.video, QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.ShiftModifier)
    app.processEvents()
    assert window.browsing and window.frame_index == 15
    assert window.browse_zoom == pytest.approx((300, 160), abs=2)
    assert not window.video.inset.isNull()
    QtTest.QTest.keyClick(window.video, QtCore.Qt.Key.Key_Escape)
    assert window.video.inset.isNull()
    # A click in the inset places the point at the zoomed spot, not the video hidden beneath the inset.
    window.look_at(500, 60)
    app.processEvents()
    QtTest.QTest.mouseClick(window.video, QtCore.Qt.MouseButton.LeftButton, pos=window.video.inset_rect.center())
    assert window.session.adjusting
    assert window.session.pending_point == pytest.approx((500, 60), abs=.5)
    window.cancel_button.click()
    window.load(25)
    window.period_buttons['play'].click()
    assert window.period_buttons['play'].text().startswith('End play')
    window.load(29)
    window.period_buttons['play'].click()
    assert window.review.periods == [{'kind': 'play', 'start': 25, 'end': 29}]


def test_switching_videos_writes_each_court_under_its_own_video(tmp_path, monkeypatch):
    import argparse
    import csv

    from court_annotation import annotate_court_corners_offframe as tool

    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sizes = {'first.avi': (640, 360), 'second.avi': (320, 240)}
    for name, (width, height) in sizes.items():
        writer = cv2.VideoWriter(str(tmp_path / name), cv2.VideoWriter_fourcc(*'MJPG'), 25, (width, height))
        for index in range(30):
            writer.write(np.full((height, width, 3), index * 3, dtype=np.uint8))
        writer.release()
    out_csv = tmp_path / 'corners.csv'
    corners = [(100, 80), (300, 80), (310, 200), (20, 200)]

    def rows_for(name):
        with out_csv.open() as handle:
            return [row for row in csv.DictReader(handle) if Path(row['video']).name == name]

    def open_window():
        return next(widget for widget in app.topLevelWidgets()
                    if isinstance(widget, CourtWorkspace) and widget.isVisible())

    def save_then_switch(target):
        window = open_window()
        window.load(7)
        window.mark_button.click()
        mark_outer_points(window, corners)
        window.choose_video(target)

    first_rows = []
    steps = [lambda: save_then_switch(1),
             lambda: (first_rows.extend(rows_for('first.avi')), save_then_switch(0)),
             lambda: open_window().close()]
    monkeypatch.setattr(QtWidgets.QApplication, 'exec', lambda _app: steps.pop(0)())
    args = argparse.Namespace(
        video=[tmp_path / name for name in sizes], out_csv=out_csv, start_video=None, start_frame=None,
        frames=2, nudge_step=.5, loupe_size=20, loupe_zoom=8, candidate_dir=None, include_uncertain=False,
        orientation='portrait')
    assert tool.run_annotation_tool(args) == 0
    assert not steps
    assert rows_for('first.avi') == first_rows
    for name, (width, height) in sizes.items():
        rows = rows_for(name)
        assert [row['video'] for row in rows] == [str(tmp_path / name)] * 4
        assert [int(row['frame']) for row in rows] == [7] * 4
        assert [(float(row['x_norm']), float(row['y_norm'])) for row in rows] == [
            (x / width, y / height) for x, y in corners]
