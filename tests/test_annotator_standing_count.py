"""Standing counts cover contact context without changing player selection."""

import numpy as np
import pytest

from annotator import player_tracking
from annotator.contacts.build_features import (
    CONTACT_FEATURE_NAMES,
    build_contact_features,
)
from annotator.courts.frame_geometry import SceneCourt
from annotator.types import TrackedPlayers, WristGaps


def test_counts_include_unselected_people_and_untracked_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_count = 9
    positions = np.array([[0.5, 0.25], [0.5, 0.75], [1.04, 0.5], [1.06, 0.5], [0.4, 0.5], [0.5, 0.5], [0.6, 0.5]])
    centres = positions * [1280.0, 720.0]
    boxes = np.column_stack((centres[:, 0] - 20, centres[:, 1] - 100, centres[:, 0] + 20, centres[:, 1]))
    boxes = np.tile(boxes, (frame_count, 1, 1))
    poses = np.zeros((frame_count, len(positions), 17, 2))
    poses[:, :, [5, 6], 1] = -1
    poses[:, :, [13, 14], 1] = 1
    poses[:, 4, [13, 14], 1] = 0  # Seated official inside the court.
    scores = np.ones((frame_count, len(positions)))
    scores[:, 5] = 0.2  # The detection threshold is strictly greater than 0.2.
    detection_counts = np.full(frame_count, 6)  # Last pose slot is padding.
    detection_counts[1] = 0
    court = {'H': np.eye(3), 'border_L': 0.0, 'border_R': 1280.0, 'border_U': 0.0, 'border_D': 720.0}
    scenes = (SceneCourt(1, 5, court, (0.0, 0.0), 0.0), SceneCourt(6, 8, court, (0.0, 0.0), 0.0))
    count_untracked = player_tracking.count_standing_in_court
    untracked_detection_counts = []

    def record_untracked_count(positions: np.ndarray, sitting: np.ndarray, margin: float) -> int:
        untracked_detection_counts.append(len(positions))
        return count_untracked(positions, sitting, margin)

    monkeypatch.setattr(player_tracking, 'count_standing_in_court', record_untracked_count)
    players = player_tracking.track_sticky_players(
        frame_count, [(2, 4)], boxes, scores, poses, detection_counts, (1280.0, 720.0), scenes,
    )

    np.testing.assert_array_equal(players.standing_count, [np.nan, 0, 3, 3, 3, np.nan, 3, 3, np.nan])
    np.testing.assert_array_equal(players.detection_indices[2:4], [[0, 1], [0, 1]])
    assert (players.detection_indices[[0, 1, 4, 5, 6, 7, 8]] == -1).all()
    assert untracked_detection_counts == [0, 5, 5, 5]


@pytest.mark.parametrize('fps,scaled_offsets', [
    (25.0, (-8, -4, 0, 4, 8)),
    (30.0, (-10, -5, 0, 5, 10)),
    (60.0, (-20, -10, 0, 10, 20)),
])
def test_count_features_sample_context_and_stop_at_search_bounds(fps: float, scaled_offsets: tuple[int, ...]) -> None:
    frame_count = 70
    standing_count = (np.arange(frame_count) % 4).astype(np.float32)
    standing_count[:5] = np.nan
    players = TrackedPlayers(
        np.full((frame_count, 2), -1), np.full((frame_count, 2, 2), np.nan),
        np.full((frame_count, 2), np.nan), standing_count,
    )
    features = build_contact_features(
        track=np.zeros((frame_count, 3)),
        pose_kps=np.zeros((frame_count, 1, 17, 2)),
        players=players,
        wrist_gaps=WristGaps(np.full(frame_count, np.inf), np.full((frame_count, 2), np.inf)),
        tracker_intervals=[(15, 60)],
        exclusion_mask=np.zeros(frame_count, dtype=bool),
        heuristic_spans=[(15, 60)],
        raw_contact_frames=list(range(frame_count)),
        scene_spans=[(5, frame_count)],
        fps=fps,
        resolution=(1280.0, 720.0),
    )
    names = ('standing_count_t-10', 'standing_count_t-5', 'standing_count_t+0',
             'standing_count_t+5', 'standing_count_t+10')
    assert CONTACT_FEATURE_NAMES[-5:] == names
    assert features.search_intervals == [(0, 60)]
    np.testing.assert_array_equal(features.rows['frame'], np.arange(60))
    for name, offset in zip(names, scaled_offsets, strict=True):
        expected = np.full(60, np.nan, dtype=np.float32)
        for frame in range(60):
            sample_frame = frame + offset
            if 0 <= sample_frame < 60:
                expected[frame] = standing_count[sample_frame]
        np.testing.assert_array_equal(features.rows[name], expected)
