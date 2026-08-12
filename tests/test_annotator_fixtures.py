"""Contract tests for the annotator migration fixture manifest."""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from annotator.replay_mask import perspective_shift_signal
from annotator.calibration.fixtures import (
    FIXTURES,
    SSET_01,
    SSET_15,
    SSET_21,
    _HOMOGRAPHY_SOURCE,
    _RESOLUTION_SOURCE,
    _load_calibration_geometry,
    REPO_ROOT,
    SHARED_FILES,
    verify_file,
    verify_fixture,
    verify_run_video_fixture,
)
from annotator.calibration.gt_scoring import build_run_video_inputs

# The complete nine-file external pin set per fixture (track, five pose-role
# arrays including kp_scores, dead mask, court-present mask, scene rows),
# keyed by the canonical stem-plus-role relative path.
EXPECTED_PIN_MD5S = {
    "sset_01": {
        "sset_01_track_npy/sset_01_track.npy": "08c5afced66b561517a43571df567b2f",
        "sset_01_pose_raw/sset_01_pose_raw_bboxes.npy": "4c9525949d1c79f0161f81b2bb63d5ef",
        "sset_01_pose_raw/sset_01_pose_raw_scores.npy": "03e655b3429f9482c5a3f4df766a3534",
        "sset_01_pose_raw/sset_01_pose_raw_kps.npy": "621427713fc617d81d4081db15613b06",
        "sset_01_pose_raw/sset_01_pose_raw_kp_scores.npy": "deb1ab46efcbe34a19bd4590b2f1b384",
        "sset_01_pose_raw/sset_01_pose_raw_ndet.npy": "5cc366f2cd459ea9be44876bc07e74ea",
        "sset_01_results/sset_01_dead_mask.npy": "a5043d329752a4e202c8566515b37231",
        "sset_01_results/sset_01_court_present.npy": "095f6ee3a3a3042c06f42e6e4467e88d",
        "sset_01_results/sset_01_scene_rows.csv": "378cfeb29a44e90ef9f9694344cca649",
    },
    "sset_15": {
        "sset_15_track_npy/sset_15_track.npy": "0b9c0966ffc58a36c65f97a5a9a78deb",
        "sset_15_pose_raw/sset_15_pose_raw_bboxes.npy": "031d4f61f71f7e3f2e18a0af5e52b138",
        "sset_15_pose_raw/sset_15_pose_raw_scores.npy": "5c3c7895312abbd28045968426fc21c4",
        "sset_15_pose_raw/sset_15_pose_raw_kps.npy": "1d74ceef0fdd53dab60e3afd64e4a6fc",
        "sset_15_pose_raw/sset_15_pose_raw_kp_scores.npy": "088d65a108d3be668dc71c93f4b9beb0",
        "sset_15_pose_raw/sset_15_pose_raw_ndet.npy": "71f7f8a9e7f270fc0ffea868da437e08",
        "sset_15_results/sset_15_dead_mask.npy": "c01914b9788afef3bca6e0b5bd88dc7f",
        "sset_15_results/sset_15_court_present.npy": "8268eeed2c48914d165c31899ce9417b",
        "sset_15_results/sset_15_scene_rows.csv": "a893afaf12920658338586e4b9b0d6d6",
    },
    "sset_21": {
        "sset_21_track_npy/sset_21_track.npy": "ad00846dc78b08de728cf59ea773ad61",
        "sset_21_pose_raw/sset_21_pose_raw_bboxes.npy": "3ee48b9637a49157ed494cbc0fbfab9a",
        "sset_21_pose_raw/sset_21_pose_raw_scores.npy": "86ba65b4e902067853a51308db864a69",
        "sset_21_pose_raw/sset_21_pose_raw_kps.npy": "6f5b60e0b2ae04ead4a3523aad744fa4",
        "sset_21_pose_raw/sset_21_pose_raw_kp_scores.npy": "014561d30e74bd6811933d68dfd19525",
        "sset_21_pose_raw/sset_21_pose_raw_ndet.npy": "1844e00ffd6cddfa1dd52e26442fef14",
        "sset_21_results/sset_21_dead_mask.npy": "9a6b43bc14f795d8c5e4d62e86005798",
        "sset_21_results/sset_21_court_present.npy": "93f5cbea19f8b7e65e272df9a5d0b252",
        "sset_21_results/sset_21_scene_rows.csv": "f9fb06285637076c5817301ae7a7b41b",
    },
}


def test_fixture_names_are_canonical():
    assert [fixture.name for fixture in FIXTURES] == ["sset_01", "sset_15", "sset_21"]
    assert FIXTURES == (SSET_01, SSET_15, SSET_21)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_fixture_paths_follow_the_stem_plus_role_pattern(fixture):
    name = fixture.name
    assert fixture.track_path == Path(f"{name}_track_npy/{name}_track.npy")
    assert fixture.pose_dir == Path(f"{name}_pose_raw")
    assert fixture.mask_path == Path(f"{name}_results/{name}_dead_mask.npy")
    assert fixture.court_present_path == Path(f"{name}_results/{name}_court_present.npy")
    assert fixture.scene_rows_path == Path(f"{name}_results/{name}_scene_rows.csv")


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_fixture_pin_set_has_nine_files_with_expected_md5s(fixture):
    expected = EXPECTED_PIN_MD5S[fixture.name]
    assert len(fixture.files) == 9
    actual = {str(pin.path): pin.md5 for pin in fixture.files}
    assert actual == expected
    assert all(pin.root == "fixtures" for pin in fixture.files)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_run_video_pin_set_excludes_only_unused_keypoint_scores(fixture):
    assert len(fixture.run_video_files) == 8
    assert fixture.pose_path("kp_scores") not in {
        pin.path for pin in fixture.run_video_files
    }
    assert set(fixture.files) - set(fixture.run_video_files) == {fixture.files[4]}


def test_replacing_a_fixtures_name_moves_every_operational_and_pin_path():
    """A copied fixture's paths must follow its own ``name``, not the original's."""
    renamed = dataclasses.replace(SSET_01, name="sset_99")

    assert renamed.track_path == Path("sset_99_track_npy/sset_99_track.npy")
    assert renamed.pose_dir == Path("sset_99_pose_raw")
    assert renamed.pose_path("kps") == Path("sset_99_pose_raw/sset_99_pose_raw_kps.npy")
    assert renamed.mask_path == Path("sset_99_results/sset_99_dead_mask.npy")
    assert renamed.court_present_path == Path("sset_99_results/sset_99_court_present.npy")
    assert renamed.scene_rows_path == Path("sset_99_results/sset_99_scene_rows.csv")

    assert len(renamed.files) == 9
    assert all(str(pin.path).startswith("sset_99") for pin in renamed.files)
    assert {pin.path for pin in renamed.files}.isdisjoint({pin.path for pin in SSET_01.files})
    assert {pin.md5 for pin in renamed.files} == {pin.md5 for pin in SSET_01.files}


def test_verify_fixture_names_the_first_missing_canonical_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNOTATOR_FIXTURES_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match=r"fixture file missing: sset_01_track_npy/sset_01_track\.npy"):
        verify_fixture(SSET_01)


def test_run_video_verification_does_not_require_unused_keypoint_scores(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ANNOTATOR_FIXTURES_ROOT", str(tmp_path))
    payload = b"run-video-input"
    digest = "d3d8d205907b04a807903e3dfa31ceba"
    fixture = dataclasses.replace(
        SSET_01,
        digests=dataclasses.replace(
            SSET_01.digests,
            track=digest,
            bboxes=digest,
            scores=digest,
            kps=digest,
            ndet=digest,
            dead_mask=digest,
            court_present=digest,
            scene_rows=digest,
        ),
    )
    for pin in fixture.run_video_files:
        path = tmp_path / pin.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    verify_run_video_fixture(fixture)
    with pytest.raises(ValueError, match="pose_raw_kp_scores.npy"):
        verify_fixture(fixture)


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
    ("fixture", "expected_court_geo", "expected_net_band"),
    [
        (
            "sset_01",
            ((460.8, 1459.5), (461.1, 1006.8), (664.6, 703.7)),
            (664.6, 703.7),
        ),
        (
            "sset_15",
            ((439.5, 1472.1), (378.0, 994.2), (583.9, 626.6)),
            (583.9, 626.6),
        ),
        (
            "sset_21",
            ((434.1, 1480.2), (453.3, 988.5), (644.6, 682.5)),
            (644.6, 682.5),
        ),
    ],
)
def test_calibration_geometry_matches_tracked_sources(fixture, expected_court_geo, expected_net_band):
    selected_fixture = next(item for item in FIXTURES if item.name == fixture)

    assert selected_fixture.court_geo == expected_court_geo
    assert selected_fixture.net_band == expected_net_band
    assert selected_fixture.court_geo[2] is selected_fixture.net_band
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
