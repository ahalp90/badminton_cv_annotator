"""Saved coordinates remain canonical through the existing benchmark reader."""
import csv
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from court_annotation import court_landmarks as court
from court_annotation.annotate_court_corners_offframe import OffframeSession

from experiments.annotator.independent_court.boundary_metrics import _rows_to_metadata

VIDEO = 'zYqgJo1L5uM.mp4'
FRAME = 54
WIDTH, HEIGHT = 640, 360
CASE = 'am1_window_00_frame_54'


def add_point(session, name, point):
    session.coarse_click(*point)
    session.select_point(name)
    session.confirm_point()


def read_output(session, expected, clicked_indices):
    with session.csv_path.open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert [int(row['corner_idx']) for row in rows] == list(range(4))
    assert [row['corner_label'] for row in rows] == list(court.CORNER_NAMES)
    assert {row['video'] for row in rows} == {VIDEO}
    assert {int(row['frame']) for row in rows} == {FRAME}
    normalised = np.array([(float(row['x_norm']), float(row['y_norm'])) for row in rows])
    assert normalised == pytest.approx(expected / [WIDTH, HEIGHT], abs=1e-5)
    metadata = _rows_to_metadata(rows, 0.0)[CASE]
    assert np.array(metadata['corners_px']) == pytest.approx(expected, abs=0.005)
    assert metadata['indices'] == clicked_indices
    return rows


def test_all_outer_click_orders_round_trip_through_consumer(tmp_path):
    expected = np.array([(100.25, 60.5), (500.75, 65.25), (550.5, 320.25), (30.25, 315.5)])
    for order in itertools.permutations(range(4)):
        session = OffframeSession(VIDEO, WIDTH, HEIGHT, 'portrait', tmp_path / 'corners.csv', 20, 8)
        session.begin_capture(FRAME)
        for index in order:
            add_point(session, court.CORNER_LANDMARK_NAMES[index], expected[index])
        session.commit()
        rows = read_output(session, expected, [0, 1, 2, 3])
        assert {row['source'] for row in rows} == {'click'}
        with session.landmark_csv_path.open() as source:
            labels = {row['landmark']: (float(row['x_px']), float(row['y_px'])) for row in csv.DictReader(source)}
        assert labels == dict(zip(court.CORNER_LANDMARK_NAMES, map(tuple, expected), strict=True))


def test_fitted_outputs_keep_identity_and_provenance_in_different_orders(tmp_path):
    expected = np.float32([(100, 60), (500, 60), (550, 320), (-20, 370)])
    transform = cv2.getPerspectiveTransform(np.float32(court.CORNER_COURT_M), expected)
    inner = ['far_baseline_x_left_singles', 'far_long_service_x_right_singles',
             'near_short_service_x_left_singles', 'near_baseline_x_right_singles',
             'far_short_service_x_centre', 'near_long_service_x_centre']
    for names in (inner, [court.CORNER_LANDMARK_NAMES[0], court.CORNER_LANDMARK_NAMES[2], *inner]):
        clicked = [index for index, name in enumerate(court.CORNER_LANDMARK_NAMES) if name in names]
        for ordered in (names, names[::-1], names[2:] + names[:2]):
            session = OffframeSession(VIDEO, WIDTH, HEIGHT, 'portrait', tmp_path / 'corners.csv', 20, 8)
            session.begin_capture(FRAME)
            for name in ordered:
                point = court.project_points(transform, np.array([court.LANDMARKS[name]]))[0]
                add_point(session, name, point)
            session.do_fit()
            assert session.fit is not None
            session.commit()
            rows = read_output(session, expected, clicked)
            assert [row['source'] for row in rows] == [
                'click' if index in clicked else 'extrapolated' for index in range(4)
            ]
            assert [row['visible'] for row in rows] == ['1', '1', '1', '0']
