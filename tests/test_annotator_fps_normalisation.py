"""Check experiment-only conversion of per-frame motion into base-30 units."""

from pathlib import Path

import numpy as np
import pytest

from annotator.config import BASE_FPS
from annotator.contacts.features import (
    BASE_MISSINGNESS_SIGNALS,
    REGION_FIELDS,
    WINDOW_OFFSETS_BASE30,
    ContactFeatures,
    build_contact_features,
)
from annotator.rally.contacts import FLOOR_EPS
from annotator.types import StickyResult
from experiments.annotator import development as experiment
from experiments.annotator.tree_simplification import input_stages
from experiments.annotator.tree_simplification.fps_normalisation import (
    SCHEMA,
    file_md5,
    fit_contacts,
    normalise_contact_features,
    preparation_receipt,
)

GEOMETRY_SIGNALS = (
    'wrist_gap_min',
    'wrist_gap_top',
    'wrist_gap_bot',
    'nearest_wrist_dx',
    'nearest_wrist_dy',
)
MOTION_SIGNALS = (
    'shuttle_vx',
    'shuttle_vy',
    'shuttle_speed',
    'shuttle_impulse',
    'shuttle_impulse_ratio',
    'ankle_speed_top',
    'ankle_speed_bot',
)


def build_motion_features(
    fps: float,
    *,
    acceleration: tuple[float, float] = (0.0, 0.0),
    include_missing_run: bool = False,
) -> tuple[np.ndarray, ContactFeatures]:
    """Build real contact rows from one physical trajectory sampled at ``fps``."""
    duration_seconds = 12
    frame_count = duration_seconds * int(fps)
    seconds = np.arange(frame_count, dtype=np.float64) / fps
    acceleration_xy = np.asarray(acceleration, dtype=np.float64)
    velocity_xy = np.asarray((0.012, -0.009), dtype=np.float64)
    start_xy = np.asarray((0.2, 0.4), dtype=np.float64)
    shuttle_xy = start_xy + seconds[:, None] * velocity_xy + 0.5 * seconds[:, None] ** 2 * acceleration_xy
    track = np.column_stack((shuttle_xy, np.ones(frame_count, dtype=np.float64)))

    ankle_pos = np.empty((frame_count, 2, 2), dtype=np.float64)
    ankle_pos[:, 0, 0] = 0.25 + 0.003 * seconds
    ankle_pos[:, 0, 1] = 0.3 + 0.004 * seconds
    ankle_pos[:, 1, 0] = 0.65 - 0.005 * seconds
    ankle_pos[:, 1, 1] = 0.7 + 0.012 * seconds
    picks = np.tile(np.asarray((0, 1), dtype=np.int32), (frame_count, 1))
    gaps = np.column_stack((0.5 + 0.01 * np.sin(seconds), 0.7 + 0.01 * np.cos(seconds)))
    if include_missing_run:
        missing = (seconds >= 2.0) & (seconds < 2.12)
        track[missing, 2] = 0
        picks[missing] = -1
        ankle_pos[missing] = np.nan
        gaps[missing] = np.nan

    distances = np.full(frame_count, np.nan, dtype=np.float64)
    finite_gaps = np.isfinite(gaps).any(axis=1)
    distances[finite_gaps] = np.nanmin(gaps[finite_gaps], axis=1)
    sticky = StickyResult(
        distances=distances,
        picks=picks,
        standing_count=np.full(frame_count, 2, dtype=np.int32),
        ankle_pos=ankle_pos,
        bbox_height=np.full((frame_count, 2), 100.0, dtype=np.float64),
        distances_per_slot=gaps,
        wrist_dist_px=gaps * 100.0,
        analysed=np.ones(frame_count, dtype=bool),
    )
    pose_kps = np.full((frame_count, 2, 17, 2), 500.0, dtype=np.float64)
    exclusion_mask = np.zeros(frame_count, dtype=bool)
    first_end = 4 * int(fps)
    second_start = 8 * int(fps)
    features = build_contact_features(
        track=track,
        pose_kps=pose_kps,
        sticky=sticky,
        tracker_intervals=((0, first_end), (second_start, frame_count)),
        exclusion_mask=exclusion_mask,
        heuristic_spans=((0, first_end), (second_start, frame_count)),
        raw_contact_frames=(2 * int(fps), 9 * int(fps)),
        scene_spans=((0, frame_count),),
        fps=fps,
        resolution=(1000.0, 1000.0),
    )
    return track, features


def row_at_frame(features: ContactFeatures, frame: int) -> np.void:
    matches = np.flatnonzero(features.rows['frame'] == frame)
    assert len(matches) == 1
    return features.rows[matches[0]]


def test_30_fps_conversion_is_exact_identity() -> None:
    track, raw = build_motion_features(BASE_FPS, include_missing_run=True)

    converted = normalise_contact_features(raw, track, BASE_FPS)

    assert converted is raw
    for field in converted.rows.dtype.names:
        actual = converted.rows[field]
        expected = raw.rows[field]
        if np.issubdtype(actual.dtype, np.floating):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('fps', (25.0, 60.0))
def test_velocity_ablation_leaves_impulse_and_ratio_raw(fps: float) -> None:
    track, raw = build_motion_features(fps, acceleration=(0.002, 0.001), include_missing_run=True)
    converted = normalise_contact_features(raw, track, fps, velocity_only=True)
    full = normalise_contact_features(raw, track, fps)
    for field in raw.rows.dtype.names:
        if field.startswith(('shuttle_vx_', 'shuttle_vy_', 'shuttle_speed_', 'ankle_speed_')):
            np.testing.assert_array_equal(converted.rows[field], full.rows[field])
        else:
            np.testing.assert_array_equal(converted.rows[field], raw.rows[field])


@pytest.mark.parametrize('fps', (25.0, 30.0, 60.0))
def test_same_linear_physical_motion_has_same_base30_velocity(fps: float) -> None:
    track, raw = build_motion_features(fps)
    converted = normalise_contact_features(raw, track, fps)
    row = row_at_frame(converted, int(2 * fps))

    expected_shuttle_vx = 0.012 / BASE_FPS
    expected_shuttle_vy = -0.009 / BASE_FPS
    expected_shuttle_speed = np.hypot(0.012, -0.009) / BASE_FPS
    expected_ankle_speed_top = np.hypot(0.003, 0.004) / BASE_FPS
    expected_ankle_speed_bot = np.hypot(-0.005, 0.012) / BASE_FPS
    np.testing.assert_allclose(row['shuttle_vx_t+0'], expected_shuttle_vx, rtol=1e-6)
    np.testing.assert_allclose(row['shuttle_vy_t+0'], expected_shuttle_vy, rtol=1e-6)
    np.testing.assert_allclose(row['shuttle_speed_t+0'], expected_shuttle_speed, rtol=1e-6)
    np.testing.assert_allclose(row['ankle_speed_top_t+0'], expected_ankle_speed_top, rtol=2e-4)
    np.testing.assert_allclose(row['ankle_speed_bot_t+0'], expected_ankle_speed_bot, rtol=2e-4)


@pytest.mark.parametrize('fps', (25.0, 30.0, 60.0))
def test_same_quadratic_motion_has_base30_impulse_and_clamped_ratio(fps: float) -> None:
    acceleration = (0.002, 0.001)
    track, raw = build_motion_features(fps, acceleration=acceleration)
    converted = normalise_contact_features(raw, track, fps)
    row = row_at_frame(converted, int(2 * fps))

    expected_impulse = np.hypot(*acceleration) / BASE_FPS**2
    assert expected_impulse < FLOOR_EPS
    np.testing.assert_allclose(row['shuttle_impulse_t+0'], expected_impulse, rtol=2e-5)
    np.testing.assert_allclose(row['shuttle_impulse_ratio_t+0'], expected_impulse / FLOOR_EPS, rtol=2e-5)


@pytest.mark.parametrize('fps', (25.0, 60.0))
def test_conversion_preserves_geometry_regions_missingness_and_boundaries(fps: float) -> None:
    track, raw = build_motion_features(fps, include_missing_run=True)
    converted = normalise_contact_features(raw, track, fps)

    for field in ('interval_id', 'frame', *REGION_FIELDS):
        np.testing.assert_array_equal(converted.rows[field], raw.rows[field], err_msg=field)

    for signal in GEOMETRY_SIGNALS + BASE_MISSINGNESS_SIGNALS:
        for offset in WINDOW_OFFSETS_BASE30:
            field = f'{signal}_t{offset:+d}'
            np.testing.assert_allclose(
                converted.rows[field], raw.rows[field], rtol=0.0, atol=0.0, equal_nan=True, err_msg=field,
            )

    for signal in MOTION_SIGNALS:
        for offset in WINDOW_OFFSETS_BASE30:
            field = f'{signal}_t{offset:+d}'
            np.testing.assert_array_equal(
                np.isnan(converted.rows[field]), np.isnan(raw.rows[field]), err_msg=field,
            )

    for interval_id, (start, end) in enumerate(raw.search_intervals):
        interval_rows = raw.rows[raw.rows['interval_id'] == interval_id]
        first = row_at_frame(converted, start)
        last = row_at_frame(converted, end - 1)
        assert np.isnan(first['shuttle_vx_t-10'])
        assert np.isnan(last['shuttle_vx_t+10'])
        assert np.isnan(first['ankle_speed_top_t-10'])
        assert np.isnan(last['ankle_speed_bot_t+10'])
        assert len(interval_rows) == end - start

    missing_frame = int(2 * fps)
    missing_row = row_at_frame(converted, missing_frame)
    assert missing_row['shuttle_visible_t+0'] == 0
    assert missing_row['pose_valid_top_t+0'] == 0
    assert np.isnan(missing_row['ankle_speed_top_t+0'])


@pytest.mark.parametrize('fps', (0.0, -1.0))
def test_conversion_rejects_nonpositive_fps(fps: float) -> None:
    features = ContactFeatures(np.empty(0, dtype=[]), [], [])

    with pytest.raises(ValueError, match='fps'):
        normalise_contact_features(features, np.empty((0, 3)), fps)


def test_partial_preparation_fails_before_creating_fit_outputs(tmp_path: Path) -> None:
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    experiment.write_json(prepared / 'preparation.json.gz', {
        'motion_units': SCHEMA, 'inputs': {'groups': {'sset_01': 'A', 'sset_18': 'V'}},
    })
    output = tmp_path / 'fit'
    with pytest.raises(ValueError, match='incomplete'):
        fit_contacts(prepared, output, seed_offset=0, threads=1)
    assert not output.exists()


def test_prepared_artifact_change_invalidates_receipt(tmp_path: Path) -> None:
    experiment.write_json(tmp_path / 'preparation.json.gz', {
        'motion_units': SCHEMA, 'inputs': {'groups': {'sset_01': 'A'}},
    })
    path = tmp_path / 'videos' / 'sset_01.joblib'
    path.parent.mkdir()
    path.write_bytes(b'original artifact')
    experiment.write_json(path.with_suffix('.json.gz'), {'motion_units': SCHEMA, 'prepared_md5': file_md5(path)})
    assert preparation_receipt(tmp_path)['artifacts'] == {'sset_01': file_md5(path)}
    path.write_bytes(b'replaced artifact')
    with pytest.raises(ValueError, match='differs from its receipt'):
        preparation_receipt(tmp_path)


def test_raw_training_refuses_normalised_prepared_features(tmp_path: Path) -> None:
    experiment.write_json(tmp_path / 'preparation.json.gz', {'motion_units': SCHEMA})
    with pytest.raises(ValueError, match='production bundles use raw units'):
        experiment.require_raw_features(tmp_path)


def test_export_refuses_normalised_units_before_writing_bundle(tmp_path: Path) -> None:
    prepared, run, output = (tmp_path / name for name in ('prepared', 'run', 'export'))
    prepared.mkdir()
    run.mkdir()
    experiment.write_json(prepared / 'preparation.json.gz', {'motion_units': SCHEMA})
    experiment.write_json(run / 'result.json.gz', {'complete': True})
    experiment.write_json(run / 'settings.json.gz', {
        'layers': 1, 'target_policy': {},
        'pool_settings': {
            'schema': input_stages.POOL_SCHEMA, 'prepared': str(prepared), 'contact_run': str(tmp_path),
        },
    })
    with pytest.raises(ValueError, match='production bundles use raw units'):
        input_stages.export_run(run, tmp_path / 'targets', output)
    assert not output.exists()
