"""Contract tests for the annotator migration fixture manifest."""

import os

import numpy as np
import pytest

from annotator.replay_mask import perspective_shift_signal
from annotator.calibration.fixtures import (
    FIXTURES,
    REPO_ROOT,
    SHARED_FILES,
    verify_file,
    verify_fixture,
)
from annotator.calibration.gt_scoring import build_run_video_inputs


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_fixture_files_are_present_and_pinned(fixture):
    if not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")
    verify_fixture(fixture)


@pytest.mark.parametrize("pin", SHARED_FILES, ids=lambda pin: str(pin.path))
def test_shared_files_are_present_and_pinned(pin):
    if pin.root == "fixtures" and not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")
    verify_file(pin)


def test_gt_repo_paths_exist():
    for fixture in FIXTURES:
        assert (REPO_ROOT / fixture.gt_set_dir).is_dir(), fixture.gt_set_dir
    for pin in SHARED_FILES:
        if pin.root == "repo":
            assert (REPO_ROOT / pin.path).is_file(), pin.path


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_fixture_manifest_includes_calibration_inputs_once(fixture):
    paths = tuple(pin.path for pin in fixture.files)
    assert paths.count(fixture.court_present_path) == 1
    assert paths.count(fixture.scene_rows_path) == 1


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_calibration_inputs_have_expected_shapes_and_rows(fixture):
    if not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")

    inputs = build_run_video_inputs(fixture)
    track = inputs.positional[0]
    court_present = inputs.keyword["court_present"]
    homography_rows = inputs.keyword["homography_rows"]

    assert isinstance(court_present, np.ndarray)
    assert court_present.shape == (len(track),)
    assert court_present.dtype == np.bool_
    assert isinstance(homography_rows, list) and homography_rows
    starts = [int(row["start_frame"]) for row in homography_rows]
    ends = [int(row["end_frame"]) for row in homography_rows]
    assert starts[0] == 0
    assert all(end == next_start for end, next_start in zip(ends, starts[1:]))
    assert ends[-1] == len(track)
    assert not perspective_shift_signal(homography_rows, len(track)).any()

    corners = np.array(
        [[float(row[column]) for column in (
            "upleft_x", "upleft_y", "upright_x", "upright_y",
            "downleft_x", "downleft_y", "downright_x", "downright_y",
        )] for row in homography_rows]
    )
    corner_points = corners.reshape(-1, 4, 2)
    span = corner_points[0].max(axis=0) - corner_points[0].min(axis=0)
    assert np.hypot(*span) > 0
