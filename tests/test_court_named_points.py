"""Arbitrary point identities retain canonical court output and geometric provenance."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from court_annotation import court_landmarks as court
from court_annotation.annotate_court_corners_offframe import (
    ActionKind,
    OffframeSession,
    load_prefill,
)


def session_at(tmp_path):
    session = OffframeSession('match.mp4', 640, 360, 'portrait', tmp_path / 'corners.csv', 20, 8)
    session.begin_capture(10)
    return session


def confirm_named(session, name, point):
    session.select_point(name)
    session.coarse_click(*point)
    return session.confirm_point()


def rows(session):
    with session.csv_path.open() as source:
        return list(csv.DictReader(source))


def test_four_outer_corners_any_order_have_canonical_rows(tmp_path):
    session = session_at(tmp_path)
    points = [(100, 80), (500, 80), (550, 300), (60, 300)]
    for index in (2, 0, 3, 1):
        confirm_named(session, court.CORNER_LANDMARK_NAMES[index], points[index])
    assert session.can_save
    assert not session.csv_path.exists()
    assert session.commit().kind == ActionKind.COMMITTED
    saved = rows(session)
    assert [(float(row['x_px']), float(row['y_px'])) for row in saved] == points
    assert [int(row['corner_idx']) for row in saved] == [0, 1, 2, 3]
    assert {row['source'] for row in saved} == {'click'}


def test_inner_points_fit_visible_corners_without_false_offscreen_flags(tmp_path):
    session = session_at(tmp_path)
    expected = np.float32([(100, 60), (500, 60), (550, 320), (30, 320)])
    transform = cv2.getPerspectiveTransform(np.float32(court.CORNER_COURT_M), expected)
    names = ['far_baseline_x_left_singles', 'far_long_service_x_right_singles',
             'near_short_service_x_left_singles', 'near_baseline_x_right_singles',
             'far_short_service_x_centre', 'near_long_service_x_centre']
    for name in names:
        point = court.project_points(transform, np.array([court.LANDMARKS[name]]))[0]
        confirm_named(session, name, point)
    assert not session.can_save
    session.do_fit()
    assert session.fit is not None
    assert session.can_save
    assert session.commit().kind == ActionKind.COMMITTED
    saved = rows(session)
    assert {row['source'] for row in saved} == {'extrapolated'}
    assert {row['visible'] for row in saved} == {'1'}
    assert np.array([(float(row['x_px']), float(row['y_px'])) for row in saved]) == pytest.approx(expected, abs=.01)
    prefill = load_prefill(session.csv_path, session.landmark_csv_path, 'match.mp4', 11)
    assert prefill is not None
    source, slots, landmarks = prefill
    assert slots == [None] * 4
    session.begin_capture_prefilled(11, slots, landmarks, source)
    assert len(session.landmarks) == len(names)
    session.do_fit()
    session.commit()
    assert len(rows(session)) == 8


def test_identity_changes_preserve_aim_and_undo_restores_replaced_point(tmp_path):
    session = session_at(tmp_path)
    session.coarse_click(200, 100)
    assert session.confirm_point().kind == ActionKind.ADJUSTED
    assert session.pending_point == (200, 100)
    first, second = court.LANDMARK_NAMES[:2]
    session.select_point(first)
    session.select_point(second)
    assert session.pending_point == (200, 100)
    session.confirm_point()
    assert session.landmarks == ((second, 200, 100),)
    assert session.selected_name == court.LANDMARK_NAMES[2]
    session.select_point(second)
    assert session.pending_point == (200, 100)
    session.nudge(2, 1)
    session.confirm_point()
    assert session.landmarks == ((second, 202, 101),)
    session.undo_last()
    assert session.landmarks == ((second, 200, 100),)


def test_reading_order_advances_across_row_and_stops_at_end(tmp_path):
    session = session_at(tmp_path)
    confirm_named(session, court.LANDMARK_NAMES[4], (300, 50))
    assert session.selected_name == court.LANDMARK_NAMES[5]
    action = confirm_named(session, court.LANDMARK_NAMES[-1], (500, 300))
    assert session.selected_name is None
    assert 'Last catalogue point' in action.message


def test_relabelling_reopened_point_removes_old_identity_and_undo_restores_both(tmp_path):
    session = session_at(tmp_path)
    first, second = court.LANDMARK_NAMES[:2]
    confirm_named(session, first, (100, 50))
    confirm_named(session, second, (200, 50))
    original = session.landmarks
    session.select_point(first)
    session.select_point(second)
    assert session.pending_point == (100, 50)
    session.confirm_point()
    assert session.landmarks == ((second, 100, 50),)
    session.undo_last()
    assert session.landmarks == original
    session.select_point(first)
    session.select_point(second)
    session.cancel_aim()
    assert session.landmarks == original


@pytest.mark.parametrize('order', [(1, 0, 2, 3), (1, 0, 3, 2)])
def test_direct_crossed_and_mirrored_courts_cannot_save(tmp_path, order):
    session = session_at(tmp_path)
    corners = [(100, 80), (500, 80), (550, 300), (60, 300)]
    for name, index in zip(court.CORNER_LANDMARK_NAMES, order, strict=True):
        confirm_named(session, name, corners[index])
    assert session.can_save
    assert 'crossed, mirrored' in session.commit().message
    assert not session.csv_path.exists()


def test_mirrored_landmark_fit_is_refused(tmp_path):
    session = session_at(tmp_path)
    target = np.float32([(500, 60), (100, 60), (30, 320), (550, 320)])
    transform = cv2.getPerspectiveTransform(np.float32(court.CORNER_COURT_M), target)
    for name in court.LANDMARK_NAMES:
        point = court.project_points(transform, np.array([court.LANDMARKS[name]]))[0]
        confirm_named(session, name, point)
    assert 'fit failed' in session.do_fit().message
    assert session.fit is None


def test_mixed_clicked_and_fitted_final_quad_is_checked(tmp_path):
    session = session_at(tmp_path)
    target = np.float32([(200, 50), (202, 50), (500, 330), (20, 330)])
    transform = cv2.getPerspectiveTransform(np.float32(court.CORNER_COURT_M), target)
    for name in court.LANDMARK_NAMES:
        if name in court.CORNER_LANDMARK_NAMES[2:]:
            continue
        point = court.project_points(transform, np.array([court.LANDMARKS[name]]))[0]
        if name == court.CORNER_LANDMARK_NAMES[1]:
            point[0] -= 2.2
        confirm_named(session, name, point)
    session.do_fit()
    assert session.fit is not None
    assert court.quad_is_camera_valid(session.fit.corners_px)
    assert 'crossed, mirrored' in session.commit().message
    assert not session.csv_path.exists()


def test_clicked_off_image_corner_keeps_click_source_but_not_visible(tmp_path):
    session = session_at(tmp_path)
    for name, point in zip(court.CORNER_LANDMARK_NAMES,
                           [(-40, 80), (500, 80), (550, 300), (-60, 300)], strict=True):
        confirm_named(session, name, point)
    session.commit()
    saved = rows(session)
    assert [row['visible'] for row in saved] == ['0', '1', '1', '0']
    assert {row['source'] for row in saved} == {'click'}
