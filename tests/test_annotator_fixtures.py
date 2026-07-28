"""Contract tests for the annotator migration fixture manifest."""

import os

import numpy as np
import pandas as pd
import pytest

from annotator.replay_mask import perspective_shift_signal
from annotator.calibration.fixtures import (
    FIXTURES,
    _HOMOGRAPHY_SOURCE,
    _RESOLUTION_SOURCE,
    _load_calibration_geometry,
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


@pytest.mark.parametrize(
    ("fixture", "expected_court_box", "expected_net_band"),
    [
        (
            "pilot",
            ((460.8, 1459.5), (461.1, 1006.8), (84.0, 336.0), (664.6, 703.7)),
            (664.6, 703.7),
        ),
        (
            "vid15",
            ((439.5, 1472.1), (378.0, 994.2), (84.0, 336.0), (583.9, 626.6)),
            (583.9, 626.6),
        ),
        (
            "sset21",
            ((434.1, 1480.2), (453.3, 988.5), (84.0, 336.0), (644.6, 682.5)),
            (644.6, 682.5),
        ),
    ],
)
def test_calibration_geometry_matches_tracked_sources(fixture, expected_court_box, expected_net_band):
    selected_fixture = next(item for item in FIXTURES if item.name == fixture)

    assert selected_fixture.court_box == expected_court_box
    assert selected_fixture.net_band == expected_net_band
    assert selected_fixture.court_box[3] is selected_fixture.net_band
    assert selected_fixture.resolution == (1920.0, 1080.0)


def _write_calibration_sources(tmp_path, homography_frame, resolution_frame):
    homography_path = tmp_path / "homography.csv"
    resolution_path = tmp_path / "resolution.csv"
    homography_frame.to_csv(homography_path, index=False)
    resolution_frame.to_csv(resolution_path, index=False)
    return homography_path, resolution_path


@pytest.mark.parametrize("source", ["homography", "resolution"])
def test_calibration_derivation_rejects_duplicate_source_rows(tmp_path, source):
    homography_frame = pd.read_csv(_HOMOGRAPHY_SOURCE)
    resolution_frame = pd.read_csv(_RESOLUTION_SOURCE)
    if source == "homography":
        homography_frame = pd.concat([homography_frame, homography_frame.iloc[[0]]], ignore_index=True)
    else:
        resolution_frame = pd.concat([resolution_frame, resolution_frame.iloc[[0]]], ignore_index=True)
    homography_path, resolution_path = _write_calibration_sources(
        tmp_path, homography_frame, resolution_frame,
    )

    with pytest.raises(ValueError, match=f"{source} source has duplicate rows"):
        _load_calibration_geometry(homography_path, resolution_path)


@pytest.mark.parametrize("source", ["homography", "resolution"])
def test_calibration_derivation_rejects_missing_source_rows(tmp_path, source):
    homography_frame = pd.read_csv(_HOMOGRAPHY_SOURCE)
    resolution_frame = pd.read_csv(_RESOLUTION_SOURCE)
    if source == "homography":
        homography_frame = homography_frame.loc[homography_frame["id"] != 1]
    else:
        resolution_frame = resolution_frame.loc[resolution_frame["id"] != 1]
    homography_path, resolution_path = _write_calibration_sources(
        tmp_path, homography_frame, resolution_frame,
    )

    with pytest.raises(ValueError, match=f"{source} source row missing for id 1"):
        _load_calibration_geometry(homography_path, resolution_path)


@pytest.mark.parametrize(
    ("source", "column", "bad_value"),
    [("homography", "upleft_x", "not-a-number"), ("resolution", "width", "not-a-number")],
)
def test_calibration_derivation_rejects_malformed_source_values(tmp_path, source, column, bad_value):
    homography_frame = pd.read_csv(_HOMOGRAPHY_SOURCE)
    resolution_frame = pd.read_csv(_RESOLUTION_SOURCE)
    frame = homography_frame if source == "homography" else resolution_frame
    frame[column] = frame[column].astype(object)
    frame.loc[frame["id"] == 1, column] = bad_value
    homography_path, resolution_path = _write_calibration_sources(
        tmp_path, homography_frame, resolution_frame,
    )

    with pytest.raises(ValueError, match=f"{source} value '{column}' is malformed"):
        _load_calibration_geometry(homography_path, resolution_path)


def test_calibration_derivation_rejects_malformed_homography_matrix(tmp_path):
    homography_frame = pd.read_csv(_HOMOGRAPHY_SOURCE)
    homography_frame.loc[homography_frame["id"] == 1, "homography_matrix"] = "not-a-matrix"
    homography_path, resolution_path = _write_calibration_sources(
        tmp_path, homography_frame, pd.read_csv(_RESOLUTION_SOURCE),
    )

    with pytest.raises(ValueError, match="homography matrix is malformed for id 1"):
        _load_calibration_geometry(homography_path, resolution_path)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_calibration_inputs_have_expected_shapes_and_rows(fixture):
    if not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")

    inputs = build_run_video_inputs(fixture)
    track = inputs.positional[0]
    inpaint_codes = inputs.keyword["inpaint_codes"]
    court_present = inputs.keyword["court_present"]
    homography_rows = inputs.keyword["homography_rows"]

    assert isinstance(court_present, np.ndarray)
    assert court_present.shape == (len(track),)
    assert court_present.dtype == np.bool_
    assert isinstance(inpaint_codes, np.ndarray)
    assert inpaint_codes.shape == (len(track),)
    assert inpaint_codes.dtype == np.uint8
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
