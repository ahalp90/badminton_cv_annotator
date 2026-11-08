"""Contact feature rules and fixed numerical regression cases."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from annotator.contacts.build_features import (
    build_candidate_mask,
    build_contact_features,
    build_eligible_intervals,
    build_player_signals,
)
from annotator.contacts.score_candidates import CONTACT_FEATURE_NAMES
from annotator.types import TrackedPlayers, WristGaps


def feature_inputs() -> dict:
    frame_count = 180
    frames = np.arange(frame_count)
    track = np.column_stack((0.2 + frames * 0.001, 0.3 + np.sin(frames / 8) * 0.05, np.ones(frame_count)))
    track[30:34, 2] = 0
    pose = np.full((frame_count, 2, 17, 2), 100.0)
    picks = np.tile([0, 1], (frame_count, 1))
    picks[40:44] = -1
    gaps = np.column_stack((1 + np.sin(frames / 10), 2 + np.cos(frames / 9)))
    gaps[40:44] = np.nan
    gaps[0:3] = np.inf
    ankle = np.full((frame_count, 2, 2), 0.5)
    ankle[40:44] = np.nan
    players = TrackedPlayers(picks, ankle, np.full((frame_count, 2), 100.0))
    wrist_gaps = WristGaps(gaps.min(axis=1), gaps)
    mask = np.zeros(frame_count, dtype=bool)
    mask[80:83] = True
    return {
        'track': track, 'pose_kps': pose, 'players': players, 'wrist_gaps': wrist_gaps,
        'tracker_intervals': [(50, 160)], 'exclusion_mask': mask,
        'heuristic_spans': [(55, 75), (90, 150)], 'raw_contact_frames': [60, 110],
        'scene_spans': [(0, 100), (100, 180)], 'resolution': (1920.0, 1080.0),
    }


REFERENCE_DIR = Path(__file__).parent / 'fixtures' / 'annotator_reference'


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
def test_feature_values_match_reference(fps: float) -> None:
    with np.load(REFERENCE_DIR / 'contact_features.npz') as reference:
        expected = reference[f'fps_{int(fps)}']
    intervals = json.loads(gzip.decompress((REFERENCE_DIR / 'contact_intervals.json.gz').read_bytes()))[str(int(fps))]
    actual = build_contact_features(**feature_inputs(), fps=fps)
    expected_fields = ("interval_id", "frame", *CONTACT_FEATURE_NAMES)
    assert actual.rows.dtype.names == expected_fields
    for name in expected_fields:
        np.testing.assert_array_equal(actual.rows[name], expected[name], err_msg=name)
    assert actual.eligible_intervals == [tuple(interval) for interval in intervals['eligible_intervals']]
    assert actual.search_intervals == [tuple(interval) for interval in intervals['search_intervals']]


def test_empty_domain() -> None:
    inputs = feature_inputs()
    empty = build_contact_features(**(inputs | {'exclusion_mask': np.ones(180, dtype=bool)}), fps=30.0)
    assert len(empty.rows) == 0
    assert empty.search_intervals == []


def test_exclusion_splits_half_open_intervals() -> None:
    mask = np.array([True, False, False, True, False, True])
    assert build_eligible_intervals([(0, 6)], mask) == [(1, 3), (4, 5)]
    assert build_eligible_intervals([(2, 5)], mask) == [(2, 3), (4, 5)]
    assert build_eligible_intervals([(0, 1), (3, 3)], mask) == []
    with pytest.raises(ValueError, match='outside'):
        build_eligible_intervals([(0, 7)], mask)


def test_player_motion_keeps_players_separate_and_stops_at_missing_coordinates() -> None:
    inputs = feature_inputs()
    ankle = inputs['players'].ankle_xy.copy()
    ankle[:4] = [
        [[0, 0], [10, 20]],
        [[3, 4], [10, 22]],
        [[np.nan, 8], [13, 26]],
        [[9, 12], [13, 26]],
    ]
    players = inputs['players']._replace(ankle_xy=ankle)
    signals = build_player_signals(inputs['track'], inputs['pose_kps'], players, inputs['wrist_gaps'], inputs['resolution'])
    np.testing.assert_allclose(signals['ankle_speed_top'][:4], [np.nan, 5, np.nan, np.nan])
    np.testing.assert_allclose(signals['ankle_speed_bot'][:4], [np.nan, 2, 5, 0])


def test_invisible_motion_and_window_edges() -> None:
    rows = build_contact_features(**feature_inputs(), fps=30.0).rows
    invisible = np.isin(rows['frame'], [30, 31, 32, 33, 34])
    assert np.isnan(rows['shuttle_vx_t+0'][invisible]).all()
    assert np.isnan(rows['shuttle_vx_t-10'][rows['frame'] < 15]).all()
    assert np.isnan(rows['shuttle_vx_t+10'][rows['frame'] >= 150]).all()
    assert rows['shuttle_visible_t+0'][rows['frame'] == 30] == 0


def test_sparse_candidates_read_offsets_from_video_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = np.zeros(180, dtype=bool)
    candidates[[25, 60, 155]] = True
    monkeypatch.setattr('annotator.contacts.build_features.build_candidate_mask', lambda *_args: candidates)
    rows = build_contact_features(**feature_inputs(), fps=30.0).rows
    np.testing.assert_array_equal(rows['frame'], [25, 60, 155])
    np.testing.assert_array_equal(rows['shuttle_visible_t+5'], [0, 1, np.nan])


def test_search_interval_ids_survive_an_interval_without_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = np.zeros(180, dtype=bool)
    candidates[155] = True
    monkeypatch.setattr('annotator.contacts.build_features.build_candidate_mask', lambda *_args: candidates)
    inputs = feature_inputs() | {'tracker_intervals': [(50, 75), (140, 160)]}
    features = build_contact_features(**inputs, fps=30.0)
    assert features.search_intervals == [(5, 75), (95, 160)]
    np.testing.assert_array_equal(features.rows['interval_id'], [1])


@pytest.mark.parametrize(('prior', 'expected_start', 'expected_end'), [
    ('heuristic', 135, 166), ('impulse', 135, 166), ('wrist', 140, 161),
    ('visibility', 135, 166), ('rally_start', 105, 196), ('scene_start', 135, 166),
])
def test_each_prior_nominates_its_contact_window(prior: str, expected_start: int, expected_end: int) -> None:
    signals = {
        'shuttle_visible': np.ones(250),
        'shuttle_impulse_ratio': np.full(250, np.nan),
        'wrist_gap_min': np.full(250, np.nan),
    }
    raw_contacts, rallies, scenes = [], [], []
    if prior == 'heuristic':
        raw_contacts = [150]
    elif prior == 'impulse':
        signals['shuttle_impulse_ratio'][150] = 2
    elif prior == 'wrist':
        signals['wrist_gap_min'][150] = 1
    elif prior == 'visibility':
        signals['shuttle_visible'][150:] = 0
    elif prior == 'rally_start':
        rallies = [(150, 180)]
    else:
        scenes = [(150, 200)]
    candidates = build_candidate_mask(signals, [(100, 200)], rallies, raw_contacts, scenes, 30.0)
    expected = np.concatenate((np.arange(55, 100), np.arange(expected_start, expected_end)))
    np.testing.assert_array_equal(np.flatnonzero(candidates), expected)


def test_seed_regions_stop_at_eligible_interval_ends() -> None:
    signals = {
        'shuttle_visible': np.ones(250),
        'shuttle_impulse_ratio': np.full(250, np.nan),
        'wrist_gap_min': np.full(250, np.nan),
    }
    # Frame 148's seed region would reach 163, beyond the next interval's start.
    candidates = build_candidate_mask(signals, [(100, 150), (155, 250)], [], [148], [], 30.0)
    # Serve lookback covers 55:100 and 110:155; the seed stays inside 100:150.
    expected = np.concatenate((np.arange(55, 100), np.arange(110, 155)))
    np.testing.assert_array_equal(np.flatnonzero(candidates), expected)
